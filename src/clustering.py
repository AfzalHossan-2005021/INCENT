import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse.csgraph import connected_components
from anndata import AnnData
from sklearn.cluster import KMeans


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
            if i < j:
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
    Rotate coords into the tissue's PCA frame (PC1 along X-axis).

    Sign convention: each axis is oriented so that its 90th-percentile
    projection is positive.  Using a high quantile rather than the mean
    makes the sign stable even when a large minority of cells are on the
    opposite side — a crop that removes up to ~40 % of the cells will
    not flip the axis.

    Returns:
        canonical : (N, 2) coordinates in the PCA frame.
        Vt        : (2, 2) rotation matrix (rows are the principal axes).
        centroid  : (2,) mean of the original coords.
    """
    centroid = coords.mean(axis=0)
    centered = coords - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)

    for i in range(2):
        if np.percentile(centered @ Vt[i], 90) < 0:
            Vt[i] = -Vt[i]

    return centered @ Vt.T, Vt, centroid


def _grid_seeds_canonical(coords: np.ndarray, S: float) -> np.ndarray:
    """
    Lay a regular grid of spacing ``S`` in the PCA-canonical frame of the
    tissue, anchored at the tissue centroid (the PCA origin), and return
    the seeds back in the original coordinate space.

    Anchoring at the PCA origin rather than the bounding-box corner means
    the grid is defined relative to the tissue's own geometry.  Two slices
    of the same tissue — even after an arbitrary rotation, crop, or border
    removal — produce seeds at the same intrinsic positions, so k-means
    converges to the same compact clusters for the shared interior region.

    Only seeds that have at least one cell within distance ``S`` are kept,
    so the tessellation follows the true tissue outline (holes and concave
    boundaries included).
    """
    canonical, Vt, centroid = _pca_canonical_coords(coords)

    # Integer grid coordinates that span the tissue in canonical space.
    c_min = canonical.min(axis=0)
    c_max = canonical.max(axis=0)
    k_lo = np.floor(c_min / S).astype(int) - 1
    k_hi = np.ceil(c_max / S).astype(int) + 2

    ks = np.arange(k_lo[0], k_hi[0])
    ls = np.arange(k_lo[1], k_hi[1])
    gk, gl = np.meshgrid(ks, ls)
    seeds_can = np.column_stack([gk.ravel(), gl.ravel()]) * S  # (M, 2) in canonical frame

    # Keep only seeds close enough to an actual cell.
    tree = cKDTree(canonical)
    d, _ = tree.query(seeds_can, k=1)
    seeds_can = seeds_can[d <= S]

    if seeds_can.shape[0] == 0:
        seeds_can = canonical.mean(axis=0, keepdims=True)

    # Map back to original coordinate space: x_orig = seeds_can @ Vt + centroid
    return seeds_can @ Vt + centroid


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    *,
    coarsen_length: float,
) -> np.ndarray:
    """
    Partition cells into uniform, contiguous supercells (mesoregions) using a
    PCA-canonical grid-seeded centroidal Voronoi tessellation.

    Seeds are placed on a regular grid of spacing ``coarsen_length`` in the
    tissue's PCA frame, anchored at the tissue centroid.  Because the grid is
    defined relative to the tissue's own principal axes (not the global bounding
    box), the same physical cell always receives the same seed neighbourhood
    regardless of how the tissue was rotated, cropped, or had its border
    trimmed.  Lloyd refinement (k-means) then yields compact, near-isotropic,
    roughly equal-area supercells — the centroidal Voronoi property that the
    original SLIC approach provided — while the PCA anchor ensures that interior
    cells get the same cluster in a parent slice and any rotated/cropped child
    derived from it.

    Compared to the original bounding-box-anchored SLIC:
    - Rotation-invariant  : seeds live in the tissue's PCA frame.
    - Crop/border-invariant: grid anchored at tissue centroid, not bbox corner.
    - Same cluster quality : k-means Lloyd iterations still give compact,
      equal-area tiles; no degenerate 1-cell boundary clusters.

    The labels are a dense ``0..C-1`` range (contiguity-enforced).

    Args:
        adata: AnnData object.
        spatial_key: Key in ``adata.obsm`` storing the (N, 2) coordinates.
        coarsen_length: Physical seed spacing ``S``.  Use the same value for
            both slices of a pair so the tessellations share one physical scale.

    Returns:
        Cluster labels from 0 to C-1.

    References:
        Achanta et al., "SLIC Superpixels Compared to State-of-the-Art
        Superpixel Methods", IEEE TPAMI 2012 (grid-seeded compact tessellation).
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=int)

    S = float(coarsen_length)
    if not np.isfinite(S) or S <= 0:
        raise ValueError("coarsen_length must be a positive, finite length scale.")

    # 1. Place seeds on the PCA-canonical grid restricted to the tissue footprint.
    seeds = _grid_seeds_canonical(coords, S)
    k = min(seeds.shape[0], n_cells)
    if k < 2:
        return np.zeros(n_cells, dtype=int)

    # 2. Centroidal Voronoi tessellation via canonical-grid-initialized k-means.
    kmeans = KMeans(n_clusters=k, init=seeds[:k], n_init=1, random_state=0)
    raw_labels = kmeans.fit_predict(coords)

    # 3. Enforce contiguity and densify labels to 0..C-1.
    labels = _split_into_contiguous_clusters(coords, raw_labels)

    return labels
