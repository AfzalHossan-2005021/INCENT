import logging
import numpy as np
import scipy.sparse as sp
import warnings

from dataclasses import dataclass, field
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay
from sklearn.metrics.pairwise import cosine_distances
from scipy.special import rel_entr
from scipy.stats import rankdata
from ot.gromov import fused_unbalanced_gromov_wasserstein
from .utils import select_backend, to_backend


class AmbiguousAlignmentWarning(UserWarning):
    """
    Raised when two or more macro-overlap seeds are nearly indistinguishable.

    Exact symmetries are not identifiable from symmetric observations alone, so
    the correct scientific behavior is to flag the ambiguity rather than hide
    it behind arbitrary tie-breaking.
    """


@dataclass
class MacroSectionResult:
    """
    Container for the coarse, biologically grounded overlap selected at cluster level.

    Returning a structured result rather than a long positional tuple makes the
    hierarchical stage much easier to audit and reason about. Reviewers also
    tend to appreciate explicit diagnostics when a conservative method refuses
    to expand beyond a small seed.
    """
    idx_A: np.ndarray
    idx_B: np.ndarray
    dist_A: np.ndarray
    dist_B: np.ndarray
    initial_idx_A: np.ndarray
    initial_idx_B: np.ndarray
    seed_pairs: list[tuple[int, int]] = field(default_factory=list)
    selected_pairs: list[tuple[int, int]] = field(default_factory=list)
    alternative_hypotheses: list[dict[str, object]] = field(default_factory=list)
    diagnostics: dict[str, object] = field(default_factory=dict)
    reason: str = ""

    @property
    def ok(self) -> bool:
        return (
            self.idx_A.size > 0
            and self.idx_B.size > 0
            and len(self.selected_pairs) > 0
        )

    @property
    def ambiguous(self) -> bool:
        return bool(
            self.diagnostics.get("seed_ambiguity_detected", False)
            or self.diagnostics.get("macro_hypothesis_ambiguity_detected", False)
        )


@dataclass
class SliceClusterCache:
    """
    Shared per-slice cluster summaries reused across hierarchical stages.

    The hierarchical pipeline previously recomputed centroids and global
    morphology descriptors inside `extract_continuous_macro_section`, even
    though the same quantities had already been constructed for the coarse FGW
    stage. Centralizing them in a cache keeps the definitions identical across
    stages and makes the pipeline easier to audit.
    """
    labels: np.ndarray
    masses: np.ndarray
    centroids: np.ndarray
    valid: np.ndarray
    mu_expr: np.ndarray
    mu_struct: np.ndarray
    cluster_hist: np.ndarray
    all_types: np.ndarray


def safe_jensenshannon(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = (p + q) / 2.0
    # Compute relative entropy and clip negative float underflows before sqrt
    js_sq = np.sum(rel_entr(p, m) + rel_entr(q, m)) / 2.0
    return np.sqrt(max(js_sq, 0.0))


def build_slice_cluster_cache(
    adata,
    labels,
    spatial_key="spatial",
    feature_key="X_pca",
    label_key="cell_type_annot",
    all_types=None,
) -> SliceClusterCache:
    """
    Compute reusable cluster summaries for one slice.

    The cache contains the cluster centroids, mean expression, intrinsic
    structural summaries, global morphology descriptors, and cell-type
    histograms needed by both the coarse FGW stage and the later macro-overlap
    extraction stage.
    """
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    n_cells = coords.shape[0]

    if feature_key == "X":
        if sp.issparse(adata.X):
            expr = adata.X.toarray()
        else:
            expr = np.asarray(adata.X)
    else:
        expr = np.asarray(adata.obsm[feature_key])

    cell_types = adata.obs[label_key].astype(str).to_numpy()
    if all_types is None:
        all_types = np.array(sorted(np.unique(cell_types)), dtype=str)
    else:
        all_types = np.array(all_types, dtype=str)

    type_to_idx = {ct: i for i, ct in enumerate(all_types)}
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    masses = np.zeros(n_clusters, dtype=np.float64)
    mu_expr = np.zeros((n_clusters, expr.shape[1]), dtype=np.float64)
    centroids = np.zeros((n_clusters, 2), dtype=np.float64)
    valid = np.zeros(n_clusters, dtype=bool)
    mu_struct_local = np.zeros((n_clusters, len(all_types) * 3), dtype=np.float64)
    cluster_hist = np.zeros((n_clusters, len(all_types)), dtype=np.float64)

    for cluster_idx, cluster_id in enumerate(unique_labels):
        mask = labels == cluster_id
        cluster_size = int(np.sum(mask))
        masses[cluster_idx] = cluster_size / float(max(n_cells, 1))
        if cluster_size == 0:
            continue

        valid[cluster_idx] = True
        mu_expr[cluster_idx] = np.mean(expr[mask], axis=0)
        centroids[cluster_idx] = np.mean(coords[mask], axis=0)

        mapped_types = np.array(
            [type_to_idx[t] for t in cell_types[mask] if t in type_to_idx],
            dtype=int,
        )
        if mapped_types.size == 0:
            continue

        counts = np.bincount(mapped_types, minlength=len(all_types)).astype(np.float64)
        cluster_hist[cluster_idx] = counts / max(counts.sum(), 1.0)

        rel_coords = coords[mask] - centroids[cluster_idx]
        thetas = np.arctan2(rel_coords[:, 1], rel_coords[:, 0])
        local_feat = np.zeros((len(all_types), 3), dtype=np.float64)
        for harmonic_idx, m in enumerate((0, 1, 2)):
            if m == 0:
                mag = counts
            else:
                ang = m * thetas
                real = np.bincount(mapped_types, weights=np.cos(ang), minlength=len(all_types))
                imag = np.bincount(mapped_types, weights=np.sin(ang), minlength=len(all_types))
                mag = np.hypot(real, imag)
            local_feat[:, harmonic_idx] = mag

        flat_local = local_feat.reshape(-1)
        if flat_local.sum() > 0:
            flat_local /= flat_local.sum()
        mu_struct_local[cluster_idx] = flat_local

    return SliceClusterCache(
        labels=np.asarray(labels),
        masses=masses,
        centroids=centroids,
        valid=valid,
        mu_expr=mu_expr,
        mu_struct=mu_struct_local,
        cluster_hist=cluster_hist,
        all_types=all_types,
    )


def compute_cluster_feature_costs(mu_expr_A, mu_struct_A, mu_expr_B, mu_struct_B, delta=0.5):
    """
    Compute inter-cluster cost matrix M_cluster between two slices.
    
    Args:
        mu_expr_A: np.ndarray (C_A, D) mean expression for slice A
        mu_struct_A: np.ndarray (C_A, M) structural features for slice A
        mu_expr_B: np.ndarray (C_B, D) mean expression for slice B
        mu_struct_B: np.ndarray (C_B, M) structural features for slice B
        delta: weight for structural distance (expression distance is 1 - delta)
    Returns:
        M_cluster: np.ndarray (C_A, C_B) cost matrix
    """

    if(delta > 1.0 or delta < 0.0):
        raise ValueError("Delta must be between 0 and 1.")
    
    # Cosine distance for continuous expression
    M_expr = cosine_distances(mu_expr_A, mu_expr_B)
            
    # Jensen-Shannon for nonnegative invariant structural descriptors
    M_struct = np.zeros((mu_struct_A.shape[0], mu_struct_B.shape[0]))
    
    for i in range(mu_struct_A.shape[0]):
        for j in range(mu_struct_B.shape[0]):
            # The descriptors are normalized nonnegative summaries over cell-type-specific structure.
            M_struct[i, j] = safe_jensenshannon(mu_struct_A[i], mu_struct_B[j])
    
    M_cluster = (1.0 - delta) * M_expr + delta * M_struct

    return M_cluster


def compute_cluster_structural_matrix(centroids):
    """
    Compute the structural distance matrix C for clusters based on their centroids.
    
    Args:
        centroids: (C, 2) coords of clusters.
    
    Returns:
        C: (C, C) combined structural distance matrix.
    """
    
    # build adj matrix
    n = centroids.shape[0]
    adj = np.zeros((n, n))
    tri = Delaunay(centroids)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(n):
        for j in indices[indptr[i]:indptr[i+1]]:
            adj[i, j] = np.linalg.norm(centroids[i] - centroids[j])
    
    C_graph = dijkstra(sp.csr_matrix(adj), directed=False)
    # handle infinite dists
    C_graph[np.isinf(C_graph)] = np.max(C_graph[~np.isinf(C_graph)]) * 2
    
    return C_graph


def run_coarse_fugw(M_cluster, C_A, C_B, p_A, p_B, alpha=0.5, reg_m=1.0, use_gpu=False):
    """
    Solves cluster-level FUGW.
    """
    scale = max(C_A.max(), C_B.max()) + 1e-8

    C_A_norm = C_A / scale
    C_B_norm = C_B / scale
    M_norm = M_cluster / (np.max(M_cluster) + 1e-8)

    _, nx = select_backend(use_gpu, gpu_verbose=False)
    logging.info("Running Unbalanced FGW on %s...", "GPU" if use_gpu else "CPU")

    Cx = to_backend(C_A_norm, nx, data_type=np.float64)
    Cy = to_backend(C_B_norm, nx, data_type=np.float64)
    M  = to_backend(M_norm,   nx, data_type=np.float64)
    wx = to_backend(p_A,      nx, data_type=np.float64)
    wy = to_backend(p_B,      nx, data_type=np.float64)

    # reg_marginals controls how much marginal relaxation is allowed (lower = more mass can be dropped)
    pi_samp, pi_feat = fused_unbalanced_gromov_wasserstein(
        Cx=Cx, Cy=Cy, wx=wx, wy=wy, M=M, alpha=alpha, reg_marginals=reg_m, max_iter=5000
    )

    return nx.to_numpy(pi_samp)


def compute_pairwise_log_enrichment(pi):
    """
    Log-enrichment of each cluster-pair under a size-preserving independence null.

    The null model preserves the coarse transport marginals and therefore answers:
    "is this pair matched more strongly than expected from cluster sizes alone?"
    Positive values indicate enrichment above that null and serve as the
    parameter-free entry criterion for candidate anchor pairs.
    """
    total_mass = np.sum(pi)
    if total_mass <= 0:
        return np.full_like(pi, -np.inf, dtype=np.float64)

    P = pi / total_mass
    row_mass = P.sum(axis=1, keepdims=True)
    col_mass = P.sum(axis=0, keepdims=True)
    expected = row_mass @ col_mass

    log_enrichment = np.full_like(P, -np.inf, dtype=np.float64)
    positive_mass = P > 0
    log_enrichment[positive_mass] = np.log(
        (P[positive_mass] + 1e-12) / (expected[positive_mass] + 1e-12)
    )
    return log_enrichment


def compute_pairwise_mutual_information_contribution(pi):
    """
    Per-pair mutual-information contribution under a size-preserving independence null.

    Each entry quantifies how much a cluster-pair in the coarse transport plan is
    enriched above the random coupling induced by the transport marginals.
    """
    total_mass = np.sum(pi)
    if total_mass <= 0:
        return np.zeros_like(pi, dtype=np.float64)

    P = pi / total_mass
    log_enrichment = compute_pairwise_log_enrichment(pi)

    contrib = np.zeros_like(P, dtype=np.float64)
    positive_mass = np.isfinite(log_enrichment)
    contrib[positive_mass] = P[positive_mass] * log_enrichment[positive_mass]
    return contrib


def build_cluster_contact_graph(coords, labels, valid_mask):
    """
    Construct a parameter-light cluster contact graph from physical cluster geometry.

    The graph is deliberately based on verified spatial contact rather than raw
    centroid proximity. Candidate edges are proposed by the centroid Delaunay
    triangulation when possible and then retained only when the minimum
    inter-cluster gap is compatible with the intrinsic within-cluster spacing of
    the two clusters. This prevents long-range shortcuts between symmetric but
    non-overlapping tissue compartments.

    Returns
    -------
    adjacency:
        Boolean contact graph with self-loops on the diagonal.
    edge_lengths:
        Symmetric matrix containing centroid-to-centroid edge lengths for the
        retained contacts and zeros elsewhere.
    """
    n_clusters = len(valid_mask)
    adjacency = np.zeros((n_clusters, n_clusters), dtype=bool)
    edge_lengths = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    np.fill_diagonal(adjacency, True)

    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) < 2:
        return adjacency, edge_lengths

    centroids = np.zeros((len(valid_idx), 2), dtype=np.float64)
    centroid_lookup = {}
    kdtrees = {}
    intra_dists = {}

    for i, c_id in enumerate(valid_idx):
        c_coords = coords[labels == c_id]
        centroids[i] = c_coords.mean(axis=0)
        centroid_lookup[c_id] = centroids[i]

        tree = cKDTree(c_coords)
        kdtrees[c_id] = tree

        if len(c_coords) > 1:
            d, _ = tree.query(c_coords, k=2)
            intra_dists[c_id] = float(np.percentile(d[:, 1], 99))
        else:
            intra_dists[c_id] = 0.0

    if len(valid_idx) >= 3:
        try:
            tri = Delaunay(centroids)
            candidate_edges = set()
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i + 1, 3):
                        u = valid_idx[simplex[i]]
                        v = valid_idx[simplex[j]]
                        candidate_edges.add((min(u, v), max(u, v)))
        except Exception:
            candidate_edges = {
                (min(u, v), max(u, v))
                for i, u in enumerate(valid_idx)
                for v in valid_idx[i + 1:]
            }
    else:
        u, v = valid_idx[0], valid_idx[1]
        candidate_edges = {(min(u, v), max(u, v))}

    for u, v in candidate_edges:
        min_dists, _ = kdtrees[u].query(coords[labels == v], k=1)
        min_gap = float(np.min(min_dists))

        if min_gap <= (intra_dists[u] + intra_dists[v]):
            edge_len = float(np.linalg.norm(centroid_lookup[u] - centroid_lookup[v]))
            adjacency[u, v] = True
            adjacency[v, u] = True
            edge_lengths[u, v] = edge_len
            edge_lengths[v, u] = edge_len

    return adjacency, edge_lengths


def normalize_contact_graph(edge_lengths):
    """
    Normalize cluster-contact edge lengths by the intrinsic cluster spacing.

    The median retained edge length is used as the characteristic graph unit so
    that geodesic distances and rigid residuals become dimensionless and
    comparable across slices with different sampling density.
    """
    positive = edge_lengths[edge_lengths > 0]
    scale = float(np.median(positive)) if positive.size > 0 else 1.0
    return edge_lengths / max(scale, 1e-12), max(scale, 1e-12)


def fit_weighted_rigid_transform(source_points, target_points, weights=None):
    """
    Fit the best rigid transform mapping source points to target points.

    Reflection is allowed because cross-slice alignment must handle arbitrary
    flips. When only one matched pair is available, the transform reduces to a
    pure translation anchored at that pair.
    """
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape[0] == 0:
        return np.eye(2, dtype=np.float64), np.zeros(2, dtype=np.float64)

    if source_points.shape[0] == 1:
        return np.eye(2, dtype=np.float64), target_points[0] - source_points[0]

    if weights is None:
        weights = np.ones(source_points.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)

    weights = weights / max(weights.sum(), 1e-12)
    source_center = (weights[:, None] * source_points).sum(axis=0)
    target_center = (weights[:, None] * target_points).sum(axis=0)

    source_centered = source_points - source_center
    target_centered = target_points - target_center

    H = source_centered.T @ (weights[:, None] * target_centered)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = target_center - R @ source_center
    return R, t


def empirical_logit_evidence(values, larger_is_better=True):
    """
    Convert a score vector into centered, parameter-free evidence values.

    Scores are ranked empirically and mapped to log-odds. This places unrelated
    evidence channels on a common scale without introducing hand-tuned weights.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.zeros(0, dtype=np.float64)

    # Protect against any rogue NaNs that might have slipped through
    nan_mask = np.isnan(values)
    if np.any(nan_mask):
        values = values.copy()
        if np.all(nan_mask):
            return np.zeros(values.size, dtype=np.float64)
        filtered = values[~nan_mask]
        values[nan_mask] = np.min(filtered) if larger_is_better else np.max(filtered)

    working = values if larger_is_better else -values
    ranks = rankdata(working, method="average")
    p = ranks / (values.size + 1.0)
    return np.log(p) - np.log1p(-p)


def equal_area_ring_edges(max_radius, n_rings):
    """
    Equal-area radial bins for global tissue-shape summaries.
    """
    max_radius = max(float(max_radius), 1e-12)
    return max_radius * np.sqrt(np.linspace(0.0, 1.0, int(n_rings) + 1))


def collect_candidate_match_pairs(Pi_cluster, valid_A, valid_B):
    """
    Assemble transport-supported cluster pairs and their overall evidence.

    We intentionally keep all positive-mass cluster pairs rather than only
    pairs already enriched above the independence null. The enriched pairs still
    dominate the seed selection, but slightly under-enriched neighbors remain
    available for later coupled expansion when their local topology strongly
    supports them. This prevents the macro-section from freezing after one or
    two clusters simply because the coarse transport mass was split across a
    nearby symmetric alternative.

    The primary transport-derived score is the per-pair mutual-information
    contribution, which already combines pair specificity with matched mass.
    Log-enrichment is retained only as a secondary tie-break so that transport
    evidence is not double-counted additively.
    """
    log_enrichment = compute_pairwise_log_enrichment(Pi_cluster)
    mi_contrib = compute_pairwise_mutual_information_contribution(Pi_cluster)

    positive_mass_flat_idx = np.flatnonzero(Pi_cluster > 0)
    if positive_mass_flat_idx.size == 0:
        diagnostics = {
            "num_positive_mass_pairs": 0,
            "num_enriched_pairs": 0,
        }
        return [], np.zeros(0), np.zeros(0), {}, mi_contrib, log_enrichment, diagnostics

    positive_mi = mi_contrib.ravel()[positive_mass_flat_idx]
    positive_enrichment = log_enrichment.ravel()[positive_mass_flat_idx]
    sorted_idx = positive_mass_flat_idx[
        np.lexsort((-positive_enrichment, -positive_mi))
    ]

    matches = []
    mi_signal = []
    enrichment_signal = []
    for idx in sorted_idx:
        u, v = np.unravel_index(idx, Pi_cluster.shape)
        if not (valid_A[u] and valid_B[v]):
            continue

        matches.append((u, v))
        mi_signal.append(float(mi_contrib[u, v]))
        enrichment_signal.append(float(log_enrichment[u, v]))

    mi_signal = np.asarray(mi_signal, dtype=np.float64)
    enrichment_signal = np.asarray(enrichment_signal, dtype=np.float64)

    mi_evidence = empirical_logit_evidence(mi_signal, larger_is_better=True)

    global_pair_evidence = {
        pair: float(me)
        for pair, me in zip(matches, mi_evidence)
    }
    global_pair_scores = np.array([global_pair_evidence[pair] for pair in matches], dtype=np.float64)

    diagnostics = {
        "num_positive_mass_pairs": int(np.sum(Pi_cluster > 0)),
        "num_enriched_pairs": int(np.sum(log_enrichment > 0)),
        "transport_score_mode": "mi_primary_enrichment_tiebreak",
    }
    return matches, enrichment_signal, global_pair_scores, global_pair_evidence, mi_contrib, log_enrichment, diagnostics


def score_frontier_matches(
    frontier_A,
    frontier_B,
    candidate_pair_set,
    selected_pairs,
    global_pair_evidence,
    adj_A,
    adj_B,
    centroids_A,
    centroids_B,
    mi_contrib,
    transform_scale,
):
    """
    Score the current frontier of admissible cluster-pairs.

    The score combines:
    1. global pair evidence from transport enrichment and niche context
    2. support from already selected neighboring pairs
    3. rigid consistency, but only once at least two selected pairs define an
       orientation-aware transform
    """
    if not frontier_A or not frontier_B or not selected_pairs:
        return [], []

    use_rigid = len(selected_pairs) >= 2
    if use_rigid:
        seed_weights = np.array(
            [max(mi_contrib[u, v], 1e-12) for u, v in selected_pairs],
            dtype=np.float64
        )
        R_seed, t_seed = fit_weighted_rigid_transform(
            centroids_A[[u for u, _ in selected_pairs]],
            centroids_B[[v for _, v in selected_pairs]],
            weights=seed_weights
        )
    else:
        R_seed = None
        t_seed = None

    frontier_pairs = []
    support_strengths = []
    rigid_residuals = []

    for u in sorted(frontier_A):
        for v in sorted(frontier_B):
            if (u, v) not in candidate_pair_set:
                continue

            support_pairs = [
                (su, sv)
                for su, sv in selected_pairs
                if adj_A[u, su] and adj_B[v, sv]
            ]
            if not support_pairs:
                continue

            support_strength = float(np.sum([
                max(global_pair_evidence[(su, sv)], 0.0)
                for su, sv in support_pairs
            ]))

            if use_rigid:
                rigid_prediction = centroids_A[u] @ R_seed.T + t_seed
                rigid_residual = float(
                    np.linalg.norm(centroids_B[v] - rigid_prediction) / transform_scale
                )
            else:
                rigid_residual = 0.0

            frontier_pairs.append((u, v))
            support_strengths.append(support_strength)
            rigid_residuals.append(rigid_residual)

    if not frontier_pairs:
        return [], []

    support_evidence = empirical_logit_evidence(support_strengths, larger_is_better=True)
    if use_rigid:
        rigid_evidence = empirical_logit_evidence(rigid_residuals, larger_is_better=False)
    else:
        rigid_evidence = np.zeros(len(frontier_pairs), dtype=np.float64)

    frontier_scores = []
    for pair, se, re in zip(
        frontier_pairs,
        support_evidence,
        rigid_evidence,
    ):
        frontier_scores.append(global_pair_evidence[pair] + float(se + re))

    return frontier_pairs, frontier_scores


def empty_macro_section_result(n_cells_A, n_cells_B, reason, diagnostics=None):
    """
    Build a conservative empty result for cases with no trustworthy macro-overlap.

    Distances are set to infinity rather than zero so downstream code cannot
    silently interpret a failure as a high-confidence core.
    """
    diagnostics = {} if diagnostics is None else dict(diagnostics)
    diagnostics.setdefault("status", "empty")
    diagnostics.setdefault("reason", reason)
    return MacroSectionResult(
        idx_A=np.array([], dtype=int),
        idx_B=np.array([], dtype=int),
        dist_A=np.full(n_cells_A, np.inf, dtype=np.float64),
        dist_B=np.full(n_cells_B, np.inf, dtype=np.float64),
        initial_idx_A=np.array([], dtype=int),
        initial_idx_B=np.array([], dtype=int),
        diagnostics=diagnostics,
        reason=reason,
    )


def build_match_graph(matches, adj_A, adj_B):
    """
    Build the product-graph adjacency over admissible matched cluster-pairs.

    Two matched pairs are connected only when their constituent clusters are
    spatial neighbors in both slices. This keeps the initial motif and later
    expansion tied to shared tissue topology rather than centroid proximity
    alone.
    """
    num_matches = len(matches)
    match_adj = np.zeros((num_matches, num_matches), dtype=bool)

    for i in range(num_matches):
        u1, v1 = matches[i]
        for j in range(i + 1, num_matches):
            u2, v2 = matches[j]
            if adj_A[u1, u2] and adj_B[v1, v2]:
                match_adj[i, j] = True
                match_adj[j, i] = True

    return match_adj


def expand_macro_match_frontier(
    seed_pairs,
    matches,
    global_pair_evidence,
    mi_contrib,
    adj_A,
    adj_B,
    edge_scale_A,
    edge_scale_B,
    centroids_A,
    centroids_B,
):
    """
    Expand the initial seed by one-to-one frontier matching on the pair graph.

    The expansion remains intentionally conservative: a frontier pair is added
    only when it is contiguous in both tissues and beats the private unmatched
    alternative in the assignment problem. This yields a natural stopping rule
    with no target-size hyperparameter.
    """
    selected_pairs = list(seed_pairs)
    selected_A = {u for u, _ in seed_pairs}
    selected_B = {v for _, v in seed_pairs}
    candidate_pair_set = set(matches)
    transform_scale = max(edge_scale_A, edge_scale_B, 1e-12)

    diagnostics = {
        "expansion_rounds": 0,
        "accepted_pairs_per_round": [],
        "stop_reason": "seed_only",
    }

    while True:
        diagnostics["expansion_rounds"] += 1

        frontier_A = {
            node
            for node in np.where(np.any(adj_A[list(selected_A), :], axis=0))[0]
            if node not in selected_A
        }
        frontier_B = {
            node
            for node in np.where(np.any(adj_B[list(selected_B), :], axis=0))[0]
            if node not in selected_B
        }

        if not frontier_A or not frontier_B:
            diagnostics["stop_reason"] = "frontier_exhausted"
            break

        frontier_pairs, frontier_scores = score_frontier_matches(
            frontier_A=frontier_A,
            frontier_B=frontier_B,
            candidate_pair_set=candidate_pair_set,
            selected_pairs=selected_pairs,
            global_pair_evidence=global_pair_evidence,
            adj_A=adj_A,
            adj_B=adj_B,
            centroids_A=centroids_A,
            centroids_B=centroids_B,
            mi_contrib=mi_contrib,
            transform_scale=transform_scale,
        )
        if not frontier_pairs:
            diagnostics["stop_reason"] = "no_contiguous_frontier_pairs"
            break

        accepted_pairs = solve_frontier_assignment(frontier_A, frontier_B, frontier_pairs, frontier_scores)
        if not accepted_pairs:
            diagnostics["stop_reason"] = "frontier_not_better_than_unmatched"
            break

        diagnostics["accepted_pairs_per_round"].append(len(accepted_pairs))
        for u, v in accepted_pairs:
            selected_pairs.append((u, v))
            selected_A.add(u)
            selected_B.add(v)

    diagnostics["selected_cluster_count_A"] = len(selected_A)
    diagnostics["selected_cluster_count_B"] = len(selected_B)
    return selected_pairs, diagnostics


def materialize_macro_section_result(
    labels_A,
    labels_B,
    coords_A,
    coords_B,
    seed_pairs,
    selected_pairs,
    diagnostics,
    alternative_hypotheses=None,
):
    """
    Convert the selected matched cluster pairs into cell indices and core distances.
    """
    strong_A = sorted({u for u, _ in selected_pairs})
    strong_B = sorted({v for _, v in selected_pairs})
    initial_clusters_A = sorted({u for u, _ in seed_pairs})
    initial_clusters_B = sorted({v for _, v in seed_pairs})

    idx_A = np.where(np.isin(labels_A, strong_A))[0]
    idx_B = np.where(np.isin(labels_B, strong_B))[0]
    initial_idx_A = np.where(np.isin(labels_A, initial_clusters_A))[0]
    initial_idx_B = np.where(np.isin(labels_B, initial_clusters_B))[0]

    if idx_A.size == 0 or idx_B.size == 0:
        return empty_macro_section_result(
            len(coords_A),
            len(coords_B),
            reason="selected cluster pairs did not map to any cells",
            diagnostics=diagnostics,
        )

    tree_A = cKDTree(coords_A[idx_A])
    tree_B = cKDTree(coords_B[idx_B])
    dist_A, _ = tree_A.query(coords_A)
    dist_B, _ = tree_B.query(coords_B)

    diagnostics = dict(diagnostics)
    diagnostics["selected_cell_count_A"] = int(idx_A.size)
    diagnostics["selected_cell_count_B"] = int(idx_B.size)

    return MacroSectionResult(
        idx_A=idx_A,
        idx_B=idx_B,
        dist_A=dist_A,
        dist_B=dist_B,
        initial_idx_A=initial_idx_A,
        initial_idx_B=initial_idx_B,
        seed_pairs=list(seed_pairs),
        selected_pairs=list(selected_pairs),
        alternative_hypotheses=[] if alternative_hypotheses is None else list(alternative_hypotheses),
        diagnostics=diagnostics,
        reason="ok",
    )


def summarize_macro_hypothesis(
    labels_A,
    labels_B,
    seed_pairs,
    selected_pairs,
    score=None,
    diagnostics=None,
):
    """
    Summarize a macro-overlap hypothesis for downstream inspection/reporting.

    The summary is intentionally lightweight: it exposes the matched cluster
    pairs and the induced cell subsets without duplicating distance fields.
    """
    idx_A = np.where(np.isin(labels_A, sorted({u for u, _ in selected_pairs})))[0]
    idx_B = np.where(np.isin(labels_B, sorted({v for _, v in selected_pairs})))[0]
    initial_idx_A = np.where(np.isin(labels_A, sorted({u for u, _ in seed_pairs})))[0]
    initial_idx_B = np.where(np.isin(labels_B, sorted({v for _, v in seed_pairs})))[0]

    return {
        "score": None if score is None else float(score),
        "seed_pairs": list(seed_pairs),
        "selected_pairs": list(selected_pairs),
        "idx_A": idx_A,
        "idx_B": idx_B,
        "initial_idx_A": initial_idx_A,
        "initial_idx_B": initial_idx_B,
        "diagnostics": {} if diagnostics is None else dict(diagnostics),
    }


def solve_seed_assignment(matches, pair_scores):
    """
    Compute a deterministic one-to-one seed candidate set by maximum evidence matching.

    This avoids arbitrary seed selection when several cluster pairs have similar
    support. The assignment does not force every cluster to match: each source
    cluster receives a private dummy option with zero evidence.
    """
    if len(matches) == 0:
        return np.array([], dtype=int)

    pair_scores = np.asarray(pair_scores, dtype=np.float64)
    nodes_A = np.array(sorted({u for u, _ in matches}), dtype=int)
    nodes_B = np.array(sorted({v for _, v in matches}), dtype=int)
    row_map = {u: i for i, u in enumerate(nodes_A)}
    col_map = {v: i for i, v in enumerate(nodes_B)}

    BIG = 1e9
    cost = np.full((len(nodes_A), len(nodes_B) + len(nodes_A)), BIG, dtype=np.float64)
    np.fill_diagonal(cost[:, len(nodes_B):], 0.0)

    match_to_index = {}
    for idx, ((u, v), score) in enumerate(zip(matches, pair_scores)):
        cost[row_map[u], col_map[v]] = min(cost[row_map[u], col_map[v]], -float(score))
        match_to_index[(u, v)] = idx

    row_ind, col_ind = linear_sum_assignment(cost)
    selected = []
    for r, c in zip(row_ind, col_ind):
        if c < len(nodes_B) and cost[r, c] < 0:
            pair = (int(nodes_A[r]), int(nodes_B[c]))
            selected.append(match_to_index[pair])

    return np.array(sorted(selected), dtype=int)


def rank_seed_motifs(match_adj, match_scores, allowed_indices, match_tiebreak_scores=None):
    """
    Enumerate admissible seed motifs in deterministic score order.

    The motif preference remains triangle > edge > singleton, but motif
    candidates are restricted to a globally consistent one-to-one assignment.
    Motifs are ranked primarily by their total evidence score; summed
    log-enrichment is used only as a secondary tie-break so transport evidence
    is not counted twice in the primary objective.
    """
    allowed_indices = np.array(sorted(set(int(x) for x in allowed_indices)), dtype=int)
    if allowed_indices.size == 0:
        return {}

    match_scores = np.asarray(match_scores, dtype=np.float64)
    if match_tiebreak_scores is None:
        match_tiebreak_scores = np.zeros_like(match_scores, dtype=np.float64)
    else:
        match_tiebreak_scores = np.asarray(match_tiebreak_scores, dtype=np.float64)

    allowed = set(allowed_indices.tolist())
    ranked = {3: [], 2: [], 1: []}

    for i in allowed_indices:
        ranked[1].append((float(match_scores[i]), float(match_tiebreak_scores[i]), (int(i),)))

    for i_pos, i in enumerate(allowed_indices):
        for j in allowed_indices[i_pos + 1:]:
            if match_adj[i, j]:
                ranked[2].append((
                    float(match_scores[i] + match_scores[j]),
                    float(match_tiebreak_scores[i] + match_tiebreak_scores[j]),
                    tuple(sorted((int(i), int(j))))
                ))

    for i_pos, i in enumerate(allowed_indices):
        for j_pos in range(i_pos + 1, allowed_indices.size):
            j = int(allowed_indices[j_pos])
            if not match_adj[i, j]:
                continue
            for k in allowed_indices[j_pos + 1:]:
                if k in allowed and match_adj[i, k] and match_adj[j, k]:
                    ranked[3].append((
                        float(match_scores[i] + match_scores[j] + match_scores[k]),
                        float(match_tiebreak_scores[i] + match_tiebreak_scores[j] + match_tiebreak_scores[k]),
                        tuple(sorted((int(i), int(j), int(k))))
                    ))

    for motif_size in ranked:
        ranked[motif_size].sort(key=lambda item: (-item[0], -item[1], item[2]))
    return ranked


def solve_frontier_assignment(frontier_A, frontier_B, candidate_pairs, candidate_scores):
    """
    Solve one-to-one matching on the current product-graph frontier.

    Each frontier node in slice A receives a private dummy option with zero
    evidence. A real pair is accepted only when its joint evidence is stronger
    than that neutral alternative, which yields an automatic stopping rule
    without prescribing a target region size.
    """
    if len(frontier_A) == 0 or len(frontier_B) == 0 or len(candidate_pairs) == 0:
        return []

    frontier_A = np.array(sorted(frontier_A), dtype=int)
    frontier_B = np.array(sorted(frontier_B), dtype=int)
    col_map = {v: i for i, v in enumerate(frontier_B)}

    BIG = 1e9
    cost = np.full((len(frontier_A), len(frontier_B) + len(frontier_A)), BIG, dtype=np.float64)
    np.fill_diagonal(cost[:, len(frontier_B):], 0.0)

    row_map = {u: i for i, u in enumerate(frontier_A)}
    for (u, v), score in zip(candidate_pairs, candidate_scores):
        if u not in row_map or v not in col_map:
            continue
        cost[row_map[u], col_map[v]] = -float(score)

    row_ind, col_ind = linear_sum_assignment(cost)

    selected = []
    for r, c in zip(row_ind, col_ind):
        if c < len(frontier_B) and cost[r, c] < 0:
            selected.append((int(frontier_A[r]), int(frontier_B[c])))
    return selected


def select_initial_match_components(matches, match_adj, match_scores, match_tiebreak_scores=None, top_k=3):
    """
    Enumerate the top seed motifs to be expanded as competing macro hypotheses.

    Instead of committing immediately to a single seed, the routine returns up
    to ``top_k`` motifs ranked by total motif evidence, with larger motifs used
    as a topology-aware tie-break. Summed log-enrichment is used only as a
    secondary ordering signal among equal primary scores, so transport evidence
    is not counted twice in the main objective. This keeps the usual preference
    for triangle/edge/singleton support, but does not let a weak triangle
    suppress a much stronger edge once we have decided to evaluate several
    seeds anyway.
    """
    if len(matches) == 0 or match_adj.shape[0] == 0:
        return [], {
            "seed_assignment_count": 0,
            "seed_candidate_count": 0,
            "seed_trial_count": 0,
        }

    match_scores = np.asarray(match_scores, dtype=np.float64)
    if match_tiebreak_scores is None:
        match_tiebreak_scores = np.zeros_like(match_scores, dtype=np.float64)
    else:
        match_tiebreak_scores = np.asarray(match_tiebreak_scores, dtype=np.float64)
    assigned_indices = solve_seed_assignment(matches, match_scores)
    if assigned_indices.size == 0:
        assigned_indices = np.arange(len(matches), dtype=int)
        assignment_mode = "fallback_all_candidates"
    else:
        assignment_mode = "stable_one_to_one"

    ranked = rank_seed_motifs(
        match_adj,
        match_scores,
        assigned_indices,
        match_tiebreak_scores=match_tiebreak_scores,
    )
    ordered_records = []
    best_size = None
    for motif_size in (3, 2, 1):
        motif_records = ranked.get(motif_size, [])
        if motif_records and best_size is None:
            best_size = motif_size
        for score, tiebreak, indices in motif_records:
            ordered_records.append((int(motif_size), float(score), float(tiebreak), tuple(indices)))

    ordered_records.sort(key=lambda item: (-item[1], -item[2], -item[0], item[3]))

    if not ordered_records:
        return [], {
            "seed_assignment_count": int(assigned_indices.size),
            "seed_candidate_count": 0,
            "seed_trial_count": 0,
            "seed_assignment_mode": assignment_mode,
        }

    top_k = max(int(top_k), 1)
    chosen_records = ordered_records[:top_k]
    selected = [np.array(indices, dtype=int) for _, _, _, indices in chosen_records]

    best_score = chosen_records[0][1]
    best_tiebreak = chosen_records[0][2]
    best_indices = chosen_records[0][3]
    best_size = int(chosen_records[0][0])
    second_score = chosen_records[1][1] if len(chosen_records) > 1 else -np.inf
    second_tiebreak = chosen_records[1][2] if len(chosen_records) > 1 else -np.inf
    second_indices = chosen_records[1][3] if len(chosen_records) > 1 else tuple()

    motif_scores = np.array([score for _, score, _, _ in ordered_records], dtype=np.float64)
    unique_scores = np.unique(motif_scores)
    positive_gaps = np.diff(np.sort(unique_scores))
    positive_gaps = positive_gaps[positive_gaps > 0]
    score_resolution = float(np.min(positive_gaps)) if positive_gaps.size > 0 else 0.0

    if np.isfinite(second_score):
        score_gap = float(best_score - second_score)
        score_ratio = float(np.exp(score_gap))
        ambiguity_detected = bool(
            np.isclose(best_score, second_score, rtol=1e-6, atol=1e-8)
            or score_gap <= max(score_resolution, 1e-12)
        )
    else:
        score_gap = np.inf
        score_ratio = np.inf
        ambiguity_detected = False

    diagnostics = {
        "seed_assignment_count": int(assigned_indices.size),
        "seed_candidate_count": int(len(ordered_records)),
        "seed_trial_count": int(len(chosen_records)),
        "seed_assignment_mode": assignment_mode,
        "seed_motif_size": int(best_size),
        "seed_best_score": float(best_score),
        "seed_best_tiebreak": float(best_tiebreak),
        "seed_second_score": float(second_score) if np.isfinite(second_score) else None,
        "seed_second_tiebreak": float(second_tiebreak) if np.isfinite(second_tiebreak) else None,
        "seed_log_evidence_gap": float(score_gap) if np.isfinite(score_gap) else None,
        "seed_evidence_ratio": float(score_ratio) if np.isfinite(score_ratio) else None,
        "seed_score_resolution": float(score_resolution),
        "seed_ambiguity_detected": ambiguity_detected,
        "seed_best_indices": list(best_indices),
        "seed_second_indices": list(second_indices),
        "seed_assignment_pairs": [matches[i] for i in assigned_indices.tolist()],
        "seed_trial_indices": [list(record[3]) for record in chosen_records],
        "seed_trial_scores": [float(record[1]) for record in chosen_records],
        "seed_trial_tiebreaks": [float(record[2]) for record in chosen_records],
        "seed_trial_sizes": [int(record[0]) for record in chosen_records],
    }
    return selected, diagnostics


def score_macro_hypothesis(
    selected_pairs,
    matches,
    match_adj,
    global_pair_evidence,
    mi_contrib,
    Pi_cluster,
    edge_scale_A,
    edge_scale_B,
    centroids_A,
    centroids_B,
):
    """
    Score a fully expanded macro-overlap hypothesis.

    We do not rerun FGW as the primary selection criterion because raw FGW
    objectives are not directly comparable across differently sized overlap
    subsets and tend to over-favor tiny, very clean seeds. Instead, we evaluate
    the hypothesis using only:

    1. node evidence from the global pair score
    2. an absolute rigid-consistency penalty once the hypothesis defines an orientation

    This produces a size-aware score without adding extra hyperparameters: a
    larger hypothesis wins only if it keeps contributing positive biological
    evidence without breaking rigid consistency.
    """
    if len(selected_pairs) == 0:
        return {
            "total_score": -np.inf,
            "node_score": -np.inf,
            "topology_score": -np.inf,
            "attachment_score": -np.inf,
            "rigid_score": -np.inf,
            "selected_pair_count": 0,
            "selected_edge_count": 0,
            "selected_mi_sum": 0.0,
            "selected_transport_mass": 0.0,
        }

    node_score = float(np.sum([global_pair_evidence[pair] for pair in selected_pairs]))
    selected_mi_sum = float(np.sum([mi_contrib[u, v] for u, v in selected_pairs]))
    selected_transport_mass = float(np.sum([Pi_cluster[u, v] for u, v in selected_pairs]))

    match_to_index = {pair: idx for idx, pair in enumerate(matches)}
    selected_indices = np.array(
        [match_to_index[pair] for pair in selected_pairs if pair in match_to_index],
        dtype=int,
    )
    selected_indices.sort()

    selected_edge_count = 0
    for i_pos, i in enumerate(selected_indices):
        for j in selected_indices[i_pos + 1:]:
            if not match_adj[int(i), int(j)]:
                continue
            selected_edge_count += 1

    topology_score = 0.0
    attachment_score = 0.0

    if len(selected_pairs) >= 2:
        weights = np.array(
            [max(mi_contrib[u, v], 1e-12) for u, v in selected_pairs],
            dtype=np.float64,
        )
        R_sel, t_sel = fit_weighted_rigid_transform(
            centroids_A[[u for u, _ in selected_pairs]],
            centroids_B[[v for _, v in selected_pairs]],
            weights=weights,
        )
        transform_scale = max(edge_scale_A, edge_scale_B, 1e-12)
        rigid_residuals = [
            float(
                np.linalg.norm(centroids_B[v] - (centroids_A[u] @ R_sel.T + t_sel))
                / transform_scale
            )
            for u, v in selected_pairs
        ]
        rigid_score = -float(np.sum(np.log1p(np.asarray(rigid_residuals, dtype=np.float64))))
    else:
        rigid_score = 0.0

    total_score = float(node_score + rigid_score)
    return {
        "total_score": total_score,
        "node_score": node_score,
        "topology_score": topology_score,
        "attachment_score": attachment_score,
        "rigid_score": rigid_score,
        "selected_pair_count": int(len(selected_pairs)),
        "selected_edge_count": int(selected_edge_count),
        "selected_mi_sum": selected_mi_sum,
        "selected_transport_mass": selected_transport_mass,
    }


def extract_continuous_macro_section(
    sliceA,
    sliceB,
    labels_A,
    labels_B,
    Pi_cluster,
    spatial_key='spatial',
    label_key='cell_type_annot',
    cluster_cache_A=None,
    cluster_cache_B=None,
):
    """
    Identify a compact, biologically consistent overlap region from the coarse alignment.

    The procedure intentionally avoids fixed region-size targets and barycentric
    growth heuristics. It proceeds in two stages:

    1. Seed selection:
       Transport-supported cluster-pairs are scored by transport enrichment
       and local niche context.
       A deterministic one-to-one seed assignment is computed first, and the
       top few connected seed motifs are retained by total seed evidence, with
       larger motifs used as a topology-aware tie-break. This preserves the
       usual preference for triangle/edge/singleton support without allowing a
       weak high-order seed to suppress a much stronger lower-order one.

    2. Coupled frontier expansion:
       Starting from each seed motif, the method grows both slices jointly on
       the product graph of admissible cluster-pairs. The expansion keeps all
       positive-mass transport pairs available, but accepts frontier pairs only
       when they are jointly supported by transport/context evidence, support
       from already selected neighboring matches, and, once orientation is
       identifiable, rigid consistency under the current
       seed-derived transform. The final macro-overlap is the expanded
       hypothesis with the highest total node-and-edge evidence.

    This design prevents the region from drifting into nearby symmetric
    compartments because expansion is never allowed independently on the two
    slices and every newly added pair must beat the neutral unmatched
    alternative in a one-to-one frontier assignment. When two seed motifs are
    exactly indistinguishable under the observed features, the method reports
    the ambiguity rather than pretending uniqueness.

    When per-slice cluster caches are supplied, the function reuses the same
    centroids and histograms already computed for the coarse alignment stage.
    This avoids silently using two slightly different definitions of the same
    cluster summary in different parts of the pipeline.

    Returns
    -------
    MacroSectionResult
        Structured selection result containing chosen cells, seed cells,
        cluster-pair correspondences, and diagnostics explaining why expansion
        stopped.
    """
    N, M = sliceA.shape[0], sliceB.shape[0]

    total_mass = np.sum(Pi_cluster)
    if total_mass == 0:
        return empty_macro_section_result(
            N,
            M,
            reason="coarse transport plan had zero mass",
            diagnostics={"total_transport_mass": 0.0},
        )

    coords_A, coords_B = sliceA.obsm[spatial_key], sliceB.obsm[spatial_key]

    num_clusters_A, num_clusters_B = Pi_cluster.shape
    if cluster_cache_A is None:
        feature_key_A = "X_pca" if "X_pca" in sliceA.obsm else "X"
        cluster_cache_A = build_slice_cluster_cache(
            sliceA,
            labels_A,
            spatial_key=spatial_key,
            feature_key=feature_key_A,
            label_key=label_key,
        )
    if cluster_cache_B is None:
        feature_key_B = "X_pca" if "X_pca" in sliceB.obsm else "X"
        cluster_cache_B = build_slice_cluster_cache(
            sliceB,
            labels_B,
            spatial_key=spatial_key,
            feature_key=feature_key_B,
            label_key=label_key,
            all_types=cluster_cache_A.all_types,
        )

    if cluster_cache_A.centroids.shape[0] != num_clusters_A or cluster_cache_B.centroids.shape[0] != num_clusters_B:
        raise ValueError(
            "Cluster cache dimensions did not match the coarse transport plan. "
            "Make sure the cache was built from the same labels used for Pi_cluster."
        )
    if cluster_cache_A.cluster_hist.shape[1] != cluster_cache_B.cluster_hist.shape[1]:
        raise ValueError(
            "Slice cluster caches were built with incompatible cell-type vocabularies."
        )

    # 1. Reuse cluster centroids and validity masks from the shared cache.
    centroids_A = np.asarray(cluster_cache_A.centroids, dtype=np.float64)
    centroids_B = np.asarray(cluster_cache_B.centroids, dtype=np.float64)
    valid_A = np.asarray(cluster_cache_A.valid, dtype=bool)
    valid_B = np.asarray(cluster_cache_B.valid, dtype=bool)

    # 2. Build cluster contact graphs and intrinsic geodesic coordinates.
    adj_A, edge_A = build_cluster_contact_graph(coords_A, labels_A, valid_A)
    adj_B, edge_B = build_cluster_contact_graph(coords_B, labels_B, valid_B)

    edge_A_norm, edge_scale_A = normalize_contact_graph(edge_A)
    edge_B_norm, edge_scale_B = normalize_contact_graph(edge_B)

    # 4. Assemble candidate cluster-pairs and their overall evidence.
    matches, match_tiebreak_scores, global_pair_scores, global_pair_evidence, mi_contrib, log_enrichment, diagnostics = collect_candidate_match_pairs(
        Pi_cluster,
        valid_A,
        valid_B
    )
    num_matches = len(matches)
    if num_matches == 0:
        return empty_macro_section_result(
            N,
            M,
            reason="no transport-supported cluster pairs survived candidate assembly",
            diagnostics=diagnostics,
        )

    print(
        "[HOT] Macro-section candidates: "
        f"{diagnostics['num_positive_mass_pairs']} transport-supported pairs, "
        f"{diagnostics['num_enriched_pairs']} enriched-above-null pairs."
    )

    # 5. Build the pair graph.
    match_adj = build_match_graph(matches, adj_A, adj_B)
    diagnostics["match_graph_edges"] = int(np.sum(match_adj) // 2)

    seed_index_trials, seed_diagnostics = select_initial_match_components(
        matches,
        match_adj,
        global_pair_scores,
        match_tiebreak_scores=match_tiebreak_scores,
        top_k=3,
    )
    diagnostics.update(seed_diagnostics)
    if len(seed_index_trials) == 0:
        return empty_macro_section_result(
            N,
            M,
            reason="candidate pairs existed but no seed motif could be selected",
            diagnostics=diagnostics,
        )

    print(
        "[HOT] Initial macro seed trials: "
        f"{len(seed_index_trials)} motif(s) will be expanded independently; "
        f"pair-graph edges={diagnostics['match_graph_edges']}."
    )
    if diagnostics.get("seed_evidence_ratio") is not None:
        print(
            "[HOT] Seed competition: "
            f"log-gap={diagnostics['seed_log_evidence_gap']:.4f}, "
            f"evidence-ratio={diagnostics['seed_evidence_ratio']:.4f}."
        )
    if diagnostics.get("seed_ambiguity_detected", False):
        competing_pairs = [matches[i] for i in diagnostics.get("seed_second_indices", [])]
        diagnostics["competing_seed_pairs"] = competing_pairs
        warnings.warn(
            "The macro-overlap seed is ambiguous: at least two connected seed motifs "
            "have indistinguishable evidence under the observed features.",
            AmbiguousAlignmentWarning,
        )

    print("[HOT] Coupled frontier expansion: accepting frontier pairs only when they improve on the unmatched null.")
    hypothesis_records = []
    seen_hypotheses = {}
    for trial_rank, seed_indices in enumerate(seed_index_trials, start=1):
        seed_pairs = [matches[i] for i in seed_indices]
        seed_name = {1: "singleton", 2: "edge", 3: "triangle"}.get(len(seed_pairs), f"{len(seed_pairs)}-pair")
        print(f"[HOT] Expanding seed trial {trial_rank}: {seed_name} with {len(seed_pairs)} matched pair(s).")

        selected_pairs, expansion_diagnostics = expand_macro_match_frontier(
            seed_pairs=seed_pairs,
            matches=matches,
            global_pair_evidence=global_pair_evidence,
            mi_contrib=mi_contrib,
            adj_A=adj_A,
            adj_B=adj_B,
            edge_scale_A=edge_scale_A,
            edge_scale_B=edge_scale_B,
            centroids_A=centroids_A,
            centroids_B=centroids_B,
        )
        hypothesis_score = score_macro_hypothesis(
            selected_pairs=selected_pairs,
            matches=matches,
            match_adj=match_adj,
            global_pair_evidence=global_pair_evidence,
            mi_contrib=mi_contrib,
            Pi_cluster=Pi_cluster,
            edge_scale_A=edge_scale_A,
            edge_scale_B=edge_scale_B,
            centroids_A=centroids_A,
            centroids_B=centroids_B,
        )
        hypothesis_diagnostics = dict(expansion_diagnostics)
        hypothesis_diagnostics.update({
            "seed_trial_rank": int(trial_rank),
            "seed_trial_indices": list(map(int, seed_indices.tolist())),
            "seed_trial_size": int(len(seed_pairs)),
            "hypothesis_total_score": float(hypothesis_score["total_score"]),
            "hypothesis_node_score": float(hypothesis_score["node_score"]),
            "hypothesis_topology_score": float(hypothesis_score["topology_score"]),
            "hypothesis_attachment_score": float(hypothesis_score["attachment_score"]),
            "hypothesis_rigid_score": float(hypothesis_score["rigid_score"]),
            "hypothesis_pair_count": int(hypothesis_score["selected_pair_count"]),
            "hypothesis_edge_count": int(hypothesis_score["selected_edge_count"]),
            "hypothesis_mi_sum": float(hypothesis_score["selected_mi_sum"]),
            "hypothesis_transport_mass": float(hypothesis_score["selected_transport_mass"]),
        })

        summary = summarize_macro_hypothesis(
            labels_A=labels_A,
            labels_B=labels_B,
            seed_pairs=seed_pairs,
            selected_pairs=selected_pairs,
            score=hypothesis_score["total_score"],
            diagnostics=hypothesis_diagnostics,
        )
        key = tuple(sorted(summary["selected_pairs"]))
        existing = seen_hypotheses.get(key)
        if existing is None:
            seen_hypotheses[key] = summary
        else:
            existing_score = float(existing.get("score", -np.inf))
            new_score = float(summary.get("score", -np.inf))
            existing_mi = float(existing.get("diagnostics", {}).get("hypothesis_mi_sum", -np.inf))
            new_mi = float(summary.get("diagnostics", {}).get("hypothesis_mi_sum", -np.inf))
            if (new_score, new_mi) > (existing_score, existing_mi):
                seen_hypotheses[key] = summary

    hypothesis_records = list(seen_hypotheses.values())
    if not hypothesis_records:
        return empty_macro_section_result(
            N,
            M,
            reason="all seed trials failed to produce a macro-overlap hypothesis",
            diagnostics=diagnostics,
        )

    hypothesis_records.sort(
        key=lambda record: (
            float(record.get("score", -np.inf)),
            float(record.get("diagnostics", {}).get("hypothesis_mi_sum", -np.inf)),
            float(record.get("diagnostics", {}).get("hypothesis_transport_mass", -np.inf)),
            len(record.get("selected_pairs", [])),
        ),
        reverse=True,
    )

    best_hypothesis = hypothesis_records[0]
    alternative_hypotheses = hypothesis_records[1:]
    diagnostics.update(best_hypothesis.get("diagnostics", {}))
    diagnostics["competing_hypothesis_count"] = int(len(alternative_hypotheses))
    diagnostics["macro_hypothesis_trial_count"] = int(len(hypothesis_records))
    diagnostics["macro_hypothesis_scores"] = [
        float(record.get("score", -np.inf)) for record in hypothesis_records
    ]
    diagnostics["selected_seed_trial_rank"] = int(
        best_hypothesis.get("diagnostics", {}).get("seed_trial_rank", 1)
    )

    best_macro_score = float(hypothesis_records[0].get("score", -np.inf))
    second_macro_score = float(hypothesis_records[1].get("score", -np.inf)) if len(hypothesis_records) > 1 else -np.inf
    macro_scores = np.array([float(record.get("score", -np.inf)) for record in hypothesis_records], dtype=np.float64)
    unique_macro_scores = np.unique(macro_scores[np.isfinite(macro_scores)])
    positive_gaps = np.diff(np.sort(unique_macro_scores))
    positive_gaps = positive_gaps[positive_gaps > 0]
    macro_score_resolution = float(np.min(positive_gaps)) if positive_gaps.size > 0 else 0.0
    if np.isfinite(second_macro_score):
        macro_score_gap = float(best_macro_score - second_macro_score)
        macro_score_ratio = float(np.exp(macro_score_gap))
        macro_ambiguity_detected = bool(
            np.isclose(best_macro_score, second_macro_score, rtol=1e-6, atol=1e-8)
            or macro_score_gap <= max(macro_score_resolution, 1e-12)
        )
    else:
        macro_score_gap = np.inf
        macro_score_ratio = np.inf
        macro_ambiguity_detected = False
    diagnostics["macro_hypothesis_best_score"] = float(best_macro_score)
    diagnostics["macro_hypothesis_second_score"] = float(second_macro_score) if np.isfinite(second_macro_score) else None
    diagnostics["macro_hypothesis_log_gap"] = float(macro_score_gap) if np.isfinite(macro_score_gap) else None
    diagnostics["macro_hypothesis_evidence_ratio"] = float(macro_score_ratio) if np.isfinite(macro_score_ratio) else None
    diagnostics["macro_hypothesis_score_resolution"] = float(macro_score_resolution)
    diagnostics["macro_hypothesis_ambiguity_detected"] = bool(macro_ambiguity_detected)

    if diagnostics["accepted_pairs_per_round"]:
        rounds = len(diagnostics["accepted_pairs_per_round"])
        total_new = int(sum(diagnostics["accepted_pairs_per_round"]))
        print(f"[HOT] Winning expansion accepted {total_new} new pair(s) over {rounds} round(s).")
    else:
        print("[HOT] Winning expansion remained at the seed; no frontier pair beat the unmatched alternative.")
    print(
        "[HOT] Winning macro hypothesis: "
        f"trial {diagnostics['selected_seed_trial_rank']} with score {diagnostics['macro_hypothesis_best_score']:.4f}; "
        f"stop reason={diagnostics['stop_reason']}."
    )
    if macro_ambiguity_detected:
        warnings.warn(
            "The final macro-overlap hypothesis remains ambiguous after expanding the top seed trials.",
            AmbiguousAlignmentWarning,
        )

    return materialize_macro_section_result(
        labels_A=labels_A,
        labels_B=labels_B,
        coords_A=coords_A,
        coords_B=coords_B,
        seed_pairs=best_hypothesis["seed_pairs"],
        selected_pairs=best_hypothesis["selected_pairs"],
        diagnostics=diagnostics,
        alternative_hypotheses=alternative_hypotheses,
    )

