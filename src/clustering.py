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
    projection is positive, which keeps the sign stable even under a
    partial crop. Superseded as the pipeline default by
    ``_farthest_point_seeds`` (rotation/reflection-invariant by
    construction, no PCA sign-ambiguity to resolve); retained here so the
    original PCA-grid seeding can be run as an ablation baseline (see
    ``method="grid"`` in :func:`cluster_cells_spatial`).

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


def _grid_seeds(coords: np.ndarray, S: float) -> np.ndarray:
    """
    Lay a regular grid of spacing ``S`` in the PCA-canonical frame of the
    tissue, anchored at the tissue centroid, and return the seeds back in
    the original coordinate space.

    This is the pipeline's original (pre-farthest-point) seeding strategy
    (Achanta et al., SLIC, IEEE TPAMI 2012 grid-seeded tessellation),
    kept as an ablation baseline: unlike ``_farthest_point_seeds``, the
    seed lattice is anchored to a PCA frame that is itself estimated from
    the data, so seed placement (and hence mesoregion identity) can shift
    under rotation or cropping.

    Only seeds that have at least one cell within distance ``S`` are kept,
    so the tessellation follows the true tissue outline.
    """
    canonical, Vt, centroid = _pca_canonical_coords(coords)

    c_min = canonical.min(axis=0)
    c_max = canonical.max(axis=0)
    k_lo = np.floor(c_min / S).astype(int) - 1
    k_hi = np.ceil(c_max / S).astype(int) + 2

    ks = np.arange(k_lo[0], k_hi[0])
    ls = np.arange(k_lo[1], k_hi[1])
    gk, gl = np.meshgrid(ks, ls)
    seeds_can = np.column_stack([gk.ravel(), gl.ravel()]) * S  # (M, 2) canonical frame

    tree = cKDTree(canonical)
    d, _ = tree.query(seeds_can, k=1)
    seeds_can = seeds_can[d <= S]

    if seeds_can.shape[0] == 0:
        seeds_can = canonical.mean(axis=0, keepdims=True)

    return seeds_can @ Vt + centroid


def _random_seeds(coords: np.ndarray, S: float, rng: np.random.Generator) -> np.ndarray:
    """
    Random sequential (Poisson-disk-style) seeding at target spacing ``S``.

    Cells are visited in a random order and accepted as a seed whenever they
    are at least ``S`` from every previously accepted seed -- the same
    minimum-spacing rule as ``_farthest_point_seeds``, but driven by random
    visitation order instead of the deterministic greedy-farthest choice.
    This isolates the effect of *which* seeds are chosen (random vs.
    farthest-point) while holding the seed density fixed, for the
    "random mesoregions" ablation.
    """
    n = coords.shape[0]
    order = rng.permutation(n)

    seed_idx = [int(order[0])]
    d_min = np.linalg.norm(coords - coords[seed_idx[0]], axis=1)

    for i in order[1:]:
        i = int(i)
        if d_min[i] >= S:
            seed_idx.append(i)
            d_min = np.minimum(d_min, np.linalg.norm(coords - coords[i], axis=1))

    return coords[seed_idx]


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
    method: str = "farthest_point",
    seed: int = 0,
) -> np.ndarray:
    """
    Partition cells into uniform, contiguous supercells (mesoregions) via a
    seeded centroidal Voronoi tessellation.

    Seeds are placed at target spacing ``coarsen_length`` by one of three
    strategies (``method``), then refined by Lloyd iterations (k-means) into
    compact, near-isotropic, roughly equal-area supercells:

    * ``"farthest_point"`` (default, pipeline choice) -- deterministic greedy
      farthest-point sampling directly on cell positions (see
      ``_farthest_point_seeds``). Depends only on pairwise Euclidean
      distances, so it is invariant to rotation/translation of the input by
      construction -- no coordinate-frame canonicalization is needed.
    * ``"grid"`` -- the pipeline's original seeding: a regular grid anchored
      in the tissue's PCA frame (see ``_grid_seeds``). Kept as an ablation
      baseline; unlike ``"farthest_point"`` it depends on an estimated PCA
      frame, so seed placement can shift under rotation or cropping.
    * ``"random"`` -- Poisson-disk-style random sequential seeding at the
      same target spacing (see ``_random_seeds``), isolating the effect of
      *which* seeds are picked from the effect of seed density.

    Labels are a dense ``0..C-1`` range (contiguity-enforced).

    Args:
        adata: AnnData object.
        spatial_key: Key in ``adata.obsm`` storing the (N, 2) coordinates.
        coarsen_length: Target seed spacing ``S``. Use the same value for
            both slices of a pair so the tessellations share one physical
            scale.
        method: One of ``"farthest_point"``, ``"grid"``, ``"random"``.
        seed: Random seed used only by ``method="random"``.

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

    # 1. Place seeds according to the requested strategy.
    if method == "farthest_point":
        seeds = _farthest_point_seeds(coords, S)
    elif method == "grid":
        seeds = _grid_seeds(coords, S)
    elif method == "random":
        seeds = _random_seeds(coords, S, np.random.default_rng(seed))
    else:
        raise ValueError(
            f"method must be 'farthest_point', 'grid', or 'random'; got {method!r}."
        )

    k = min(seeds.shape[0], n_cells)
    if k < 2:
        return np.zeros(n_cells, dtype=int)

    # 2. Centroidal Voronoi tessellation via seeded k-means.
    kmeans = KMeans(n_clusters=k, init=seeds[:k], n_init=1, random_state=0)
    raw_labels = kmeans.fit_predict(coords)

    # 3. Enforce contiguity and densify labels to 0..C-1.
    labels = _split_into_contiguous_clusters(coords, raw_labels)

    return labels
