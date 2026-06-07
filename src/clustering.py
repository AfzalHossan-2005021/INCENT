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
            if i < j:  # Avoid duplicates
                edges.append((i, j))

    return edges


def _grid_seeds(coords: np.ndarray, S: float) -> np.ndarray:
    """
    Lay a regular square grid at spacing ``S`` over the coordinate bounding box
    and keep only grid points that fall within one spacing of an actual cell.

    Restricting seeds to occupied grid cells lets the tessellation follow the
    true tissue outline (holes and concave boundaries included) while keeping
    every supercell at the same physical scale. This is the SLIC seeding step
    (``S = sqrt(N/K)`` grid spacing) specialized to a point cloud.
    """
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    xs = np.arange(mins[0] + 0.5 * S, maxs[0] + S, S)
    ys = np.arange(mins[1] + 0.5 * S, maxs[1] + S, S)
    if xs.size == 0:
        xs = np.array([(mins[0] + maxs[0]) / 2.0])
    if ys.size == 0:
        ys = np.array([(mins[1] + maxs[1]) / 2.0])

    gx, gy = np.meshgrid(xs, ys)
    grid = np.column_stack([gx.ravel(), gy.ravel()])

    tree = cKDTree(coords)
    d, _ = tree.query(grid, k=1)
    seeds = grid[d <= S]

    if seeds.shape[0] == 0:
        # Degenerate footprint: fall back to a single central seed.
        seeds = coords.mean(axis=0, keepdims=True)
    return seeds


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


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    *,
    coarsen_length: float,
) -> np.ndarray:
    """
    Partition cells into uniform, contiguous supercells (mesoregions) using a
    deterministic grid-seeded centroidal Voronoi tessellation (SLIC-style).

    Cells are assigned to the nearest of a set of seeds placed on a regular grid
    at physical spacing ``coarsen_length``; Lloyd refinement (k-means on the 2-D
    coordinates) then yields a centroidal Voronoi tessellation, i.e. compact,
    near-isotropic tiles of roughly equal area. Because the seed spacing is a
    physical length supplied by the caller, two slices clustered with the same
    ``coarsen_length`` get supercells of matching size, shape, and (up to the
    inter-slice rigid transform) position -- the property the cluster-level FGW
    alignment depends on.

    The result is fully deterministic (fixed grid initialization, ``n_init=1``)
    and seed-independent, and the labels are a dense ``0..C-1`` range.

    Args:
        adata: AnnData object.
        spatial_key: Key in ``adata.obsm`` storing the (N, 2) spatial coordinates.
        coarsen_length: Physical seed spacing ``S`` of the supercell grid. The
            same value should be used for both slices of a pair so the
            tessellations are at one shared scale. Typically derived from the
            slices' characteristic cell spacing (see
            ``core.estimate_coarsen_length``).

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

    # 1. Seed a regular grid restricted to the tissue footprint.
    seeds = _grid_seeds(coords, S)
    k = min(seeds.shape[0], n_cells)
    if k < 2:
        # Footprint smaller than one supercell, or too few cells: one region.
        return np.zeros(n_cells, dtype=int)

    # 2. Centroidal Voronoi tessellation via grid-initialized k-means on coords.
    kmeans = KMeans(n_clusters=k, init=seeds[:k], n_init=1, random_state=0)
    raw_labels = kmeans.fit_predict(coords)

    # 3. Enforce contiguity and densify labels to 0..C-1.
    labels = _split_into_contiguous_clusters(coords, raw_labels)

    return labels
