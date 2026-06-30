import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.csgraph import connected_components
from anndata import AnnData


def build_spatial_graph(coords: np.ndarray):
    """
    Builds a spatial connectivity graph (Delaunay edge list).

    Args:
        coords: Spatial coordinates array shape (N, 2).

    Returns:
        edges: list of tuples (i, j) with i < j.
    """
    n_cells = coords.shape[0]
    edges = []

    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i+1]]:
            if i < j:  # Avoid duplicates
                edges.append((i, j))

    return edges


def _split_into_contiguous_clusters(coords: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Relabel a partition into spatially contiguous, consecutively numbered clusters.

    Each connected component of the Delaunay graph restricted to within-label
    edges becomes one cluster. This (1) splits the rare Voronoi cell that bridges
    a tissue gap into separate physical regions and (2) guarantees the labels are
    a dense ``0..C-1`` range, which the downstream hierarchical stage relies on
    (cluster ids are used directly as positional indices into ``Pi_cluster``).
    """
    n = coords.shape[0]
    labels = np.asarray(labels)

    try:
        edges = build_spatial_graph(coords)
    except Exception:
        edges = []

    if len(edges) == 0:
        # Cannot build a graph (degenerate geometry): just densify the labels.
        _, inv = np.unique(labels, return_inverse=True)
        return inv.astype(int)

    ei = np.fromiter((e[0] for e in edges), dtype=int, count=len(edges))
    ej = np.fromiter((e[1] for e in edges), dtype=int, count=len(edges))

    same = labels[ei] == labels[ej]
    ei, ej = ei[same], ej[same]

    rows = np.concatenate([ei, ej])
    cols = np.concatenate([ej, ei])
    data = np.ones(rows.shape[0], dtype=np.int8)
    graph = sp.csr_matrix((data, (rows, cols)), shape=(n, n))

    _, comp = connected_components(graph, directed=False, connection="weak")
    return comp.astype(int)


def _pca_canonical_coords(coords: np.ndarray):
    """
    Rotate coords into the tissue's PCA frame so the first principal component
    aligns with the X-axis.

    Sign convention: each axis is flipped so that the majority of cells project
    onto its positive half — this breaks the 180° ambiguity and gives a
    consistent orientation for the same tissue regardless of how it was rotated
    in the original coordinate system.

    Returns:
        canonical: (N, 2) coordinates in the PCA frame.
        Vt: (2, 2) rotation matrix (rows are principal axes in original space).
    """
    centered = coords - coords.mean(axis=0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    for i in range(2):
        if np.mean(centered @ Vt[i] > 0) < 0.5:
            Vt[i] = -Vt[i]

    return centered @ Vt.T, Vt


def _canonical_grid_labels(coords: np.ndarray, S: float) -> np.ndarray:
    """
    Assign each cell to the nearest vertex of a fixed grid of spacing ``S``
    anchored at the tissue centroid in PCA-canonical coordinates.

    Because the grid is defined relative to the tissue's own principal axes
    (not the global bounding box), the same physical cell always maps to the
    same grid vertex regardless of how the tissue was rotated, cropped, or
    had its border trimmed.  This is an O(N) rounding operation — no k-means
    iteration is needed.

    Args:
        coords: (N, 2) spatial coordinates in any frame.
        S: Grid spacing (same physical units as coords).

    Returns:
        Raw integer labels (not yet contiguity-enforced, not necessarily dense).
    """
    canonical, _ = _pca_canonical_coords(coords)

    # Round each cell's canonical position to the nearest grid vertex.
    vertex_ij = np.round(canonical / S).astype(np.int64)  # (N, 2)

    # Map unique (gx, gy) integer pairs to dense sequential IDs.
    _, labels = np.unique(vertex_ij, axis=0, return_inverse=True)
    return labels


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    *,
    coarsen_length: float,
) -> np.ndarray:
    """
    Partition cells into uniform, contiguous supercells (mesoregions) using a
    canonical grid-Voronoi tessellation.

    Each cell is assigned to the nearest vertex of a regular grid of spacing
    ``coarsen_length`` anchored at the tissue centroid in PCA-canonical
    coordinates (the tissue's own principal axes).  Because the grid is defined
    relative to the tissue's intrinsic geometry rather than the global bounding
    box, the same physical cell always receives the same grid-vertex assignment
    regardless of how the tissue was rotated, cropped, or had its border
    trimmed.  Interior cells (away from the crop boundary) therefore get
    identical cluster labels in a parent slice and any rotated/cropped child
    derived from it.

    Compared to the previous SLIC/k-means approach:
    - No bounding-box anchoring (grid origin = tissue centroid in PCA frame).
    - No k-means iteration (O(N) rounding instead of iterative Lloyd steps).
    - Rotation-invariant: PCA frame tracks the tissue's own axes.
    - Crop/border-invariant: interior cells are unaffected; only boundary
      clusters may differ after contiguity splitting.

    The labels are a dense ``0..C-1`` range (contiguity-enforced).

    Args:
        adata: AnnData object.
        spatial_key: Key in ``adata.obsm`` storing the (N, 2) coordinates.
        coarsen_length: Physical grid spacing ``S``.  The same value should be
            used for both slices of a pair so the tessellations are at one
            shared physical scale.

    Returns:
        Cluster labels from 0 to C-1.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=int)

    S = float(coarsen_length)
    if not np.isfinite(S) or S <= 0:
        raise ValueError("coarsen_length must be a positive, finite length scale.")

    if n_cells < 2:
        return np.zeros(n_cells, dtype=int)

    # 1. Assign each cell to its canonical grid vertex.
    raw_labels = _canonical_grid_labels(coords, S)

    # 2. Enforce contiguity and densify labels to 0..C-1.
    labels = _split_into_contiguous_clusters(coords, raw_labels)

    return labels
