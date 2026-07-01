import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
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


def _farthest_point_seeds(coords: np.ndarray, S: float) -> np.ndarray:
    """
    Deterministic farthest-point (greedy Poisson-disk) seeding.

    Seeds are chosen directly from cell positions: start at the cell nearest
    the section centroid, then repeatedly add the cell farthest (Euclidean
    distance) from every seed chosen so far, stopping once the farthest
    remaining candidate is closer than ``S`` to an existing seed. The result
    is a set of seeds at roughly uniform spacing ``S`` that follows the
    tissue outline automatically -- holes and concave boundaries are
    respected because seeds are only ever placed on real cell positions.

    Unlike a grid, this procedure depends only on pairwise Euclidean
    distances between cells and is therefore invariant to rotation and
    translation of the input coordinates by construction. No coordinate-
    frame canonicalization (PCA) and no sign-ambiguity resolution are
    required, and the seed grid cannot land in a rotation-dependent phase
    relative to the tissue.

    Deterministic: initialization is the fixed nearest-centroid point, not a
    random draw, and each subsequent choice is the unique running argmax of
    a min-distance field (ties broken by lowest array index).

    Complexity: O(k * n) for k retained seeds and n cells (each iteration
    is one vectorized distance update over all cells); k is set by S via
    the stopping rule and is of the same order as the previous grid-based
    seed count for the same S.
    """
    n = coords.shape[0]
    if n == 0:
        return np.empty((0, 2))

    centroid = coords.mean(axis=0)
    start = int(np.argmin(np.linalg.norm(coords - centroid, axis=1)))

    seed_idx = [start]
    d_min = np.linalg.norm(coords - coords[start], axis=1)

    while True:
        cand = int(np.argmax(d_min))
        if d_min[cand] < S or len(seed_idx) >= n:
            break
        seed_idx.append(cand)
        d_min = np.minimum(d_min, np.linalg.norm(coords - coords[cand], axis=1))

    return coords[seed_idx]


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    *,
    coarsen_length: float,
) -> np.ndarray:
    """
    Partition cells into uniform, contiguous supercells (mesoregions) using a
    farthest-point-seeded centroidal Voronoi tessellation.

    Seeds are chosen by greedy farthest-point sampling at target spacing
    ``coarsen_length`` (see ``_farthest_point_seeds``), directly on the
    original cell coordinates. Because the seeding step uses only pairwise
    Euclidean distances, the resulting partition is invariant to rotation
    and translation of the input by construction -- no coordinate-frame
    canonicalization is needed. Lloyd refinement (k-means) then yields
    compact, near-isotropic, roughly equal-area supercells, exactly as in
    the grid-seeded version.

    Labels are a dense ``0..C-1`` range (contiguity-enforced).

    Args:
        adata: AnnData object.
        spatial_key: Key in ``adata.obsm`` storing the (N, 2) coordinates.
        coarsen_length: Target seed spacing ``S``. Use the same value for
            both slices of a pair so the tessellations share one physical
            scale.

    Returns:
        Cluster labels from 0 to C-1.

    References:
        Achanta et al., IEEE TPAMI 2012 (compact tessellation via seeded
        Lloyd refinement); farthest-point / Poisson-disk sampling is
        standard practice for blue-noise seed placement in computational
        geometry (e.g. Cook 1986; Lloyd 1982 for the refinement step).
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]
    if n_cells == 0:
        return np.zeros(0, dtype=int)

    S = float(coarsen_length)
    if not np.isfinite(S) or S <= 0:
        raise ValueError("coarsen_length must be a positive, finite length scale.")

    # 1. Place seeds directly on the tissue by greedy farthest-point sampling.
    seeds = _farthest_point_seeds(coords, S)
    k = min(seeds.shape[0], n_cells)
    if k < 2:
        return np.zeros(n_cells, dtype=int)

    # 2. Centroidal Voronoi tessellation via farthest-point-initialized k-means.
    kmeans = KMeans(n_clusters=k, init=seeds[:k], n_init=1, random_state=0)
    raw_labels = kmeans.fit_predict(coords)

    # 3. Enforce contiguity and densify labels to 0..C-1.
    labels = _split_into_contiguous_clusters(coords, raw_labels)

    return labels
