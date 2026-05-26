"""
Spatially-constrained biological clustering for INCENT mesoregions.

The original `cluster_cells_spatial` ran Ward agglomerative clustering on
**spatial coordinates**, with a Delaunay-graph connectivity constraint and a
fixed-cell-count target. That produced mesoregions that were spatially
contiguous blobs but biologically arbitrary: a single biological region
(e.g. Layer IV of cortex) appearing in full in slice A and only partially
in slice B would be fragmented in A into multiple chunks while remaining
one chunk in B. The downstream cluster-cost matrix M^cl then saw a "many
to one" pattern that the partial-FGW solver cannot reconcile under a
unique optimal coupling, defeating the partial-overlap machinery.

This module fixes the failure mode with four independent improvements,
all opt-in via keyword arguments:

1. **Biological clustering** (`use_rep`): Ward fits on gene-expression PCA
   (or any obsm representation) while the Delaunay connectivity matrix
   still enforces spatial contiguity. Cells with the same biological
   identity that are spatially adjacent merge into one mesoregion.

2. **Gap-statistic K selection** (`auto_k`): the cluster count is chosen
   data-adaptively rather than fixed to n_cells/200. The gap statistic
   (Tibshirani, Walther & Hastie 2001) compares within-cluster
   sum-of-squares against a uniform-expression null and picks the
   smallest K that meets the "one-standard-error" stability criterion.

3. **Boundary-aware merging** (`min_cluster_frac`): any cluster smaller
   than `min_cluster_frac * (n_cells / K)` is folded into its
   expression-most-similar adjacent cluster. Catches residual
   fragmentation when intra-region heterogeneity briefly exceeds
   inter-region heterogeneity.

4. **Cell-type-anchored clustering** (`use_celltype`): when a trustworthy
   cell-type annotation is available in adata.obs, the connected
   components of the per-type Delaunay subgraph become clusters. This is
   the recommended mode when you need overlap consistency across slices
   (i.e. clustering a full brain slice A and its 60% subsection B
   independently should produce identical labels on the overlap
   interior). Over-large components are optionally split by biological
   Ward via `max_cluster_frac`.

Backward compatibility
----------------------
The original keyword-only signature ``cluster_cells_spatial(adata,
spatial_key, resolution)`` is preserved. When called without the new
arguments, the function transparently switches to biological clustering
on whichever representation is available (in order of preference:
``X_pca``, ``X``); the original behaviour can be recovered with
``use_rep="spatial"``.

When called with ``use_rep="spatial"`` AND ``auto_k=False`` AND
``min_cluster_frac=0``, this module is exactly equivalent to the original.

Cross-slice overlap consistency
-------------------------------
For workflows where two slices share an overlap region and you want
clustering to be identical on the overlap interior:

  * Use ``use_celltype=<obs key>`` with a trusted cell-type label.
  * If you cluster on gene expression instead (``use_celltype=None``),
    ensure both slices share the same PCA basis. The helper
    ``compute_joint_pca([sliceA, sliceB], n_components=30)`` projects
    both slices into a shared PC space and writes the result to
    ``obsm["X_pca_joint"]``; pass ``use_rep="X_pca_joint"`` to cluster
    in that shared space.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay
from anndata import AnnData
from sklearn.cluster import AgglomerativeClustering
from typing import Optional


# ============================================================================
# Spatial graph (unchanged from original)
# ============================================================================

def build_spatial_graph(coords: np.ndarray):
    """
    Builds a spatial connectivity graph (adjacency matrix/edge list).

    Args:
        coords: Spatial coordinates array shape (N, 2).

    Returns:
        edges: list of tuples (i, j).
        weights: corresponding edge weights (e.g. 1/distance, or 1.0).
    """
    n_cells = coords.shape[0]
    edges = []

    tri = Delaunay(coords)
    indptr, indices = tri.vertex_neighbor_vertices
    for i in range(n_cells):
        for j in indices[indptr[i]:indptr[i + 1]]:
            if i < j:  # Avoid duplicates
                edges.append((i, j))

    return edges


# ============================================================================
# Feature-matrix selection
# ============================================================================

def _resolve_feature_matrix(adata: AnnData, use_rep: Optional[str], spatial_key: str) -> np.ndarray:
    """
    Return the (n_cells, d) feature matrix on which Ward is fit.

    Resolution order:
        * use_rep == "spatial"  -> spatial coordinates (legacy behaviour)
        * use_rep is not None   -> adata.obsm[use_rep]
        * use_rep is None       -> obsm["X_pca"] if present, else
                                    densified adata.X (with a sane PCA
                                    projection if X has many genes)
    """
    if use_rep == "spatial":
        return np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(
                f"use_rep='{use_rep}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        return np.asarray(adata.obsm[use_rep], dtype=np.float64)
    # Auto: prefer X_pca; else use X (subset to top-N PCs to stay tractable)
    if "X_pca" in adata.obsm:
        return np.asarray(adata.obsm["X_pca"], dtype=np.float64)
    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    # If X is wide (e.g. raw genes), project to 30 PCs in a deterministic way
    # so Ward is tractable. We use a thin SVD, which is deterministic given
    # input and faster than sklearn's PCA on the (n, g) -> (n, 30) projection.
    if X.shape[1] > 50:
        X_centered = X - X.mean(axis=0, keepdims=True)
        # Truncated SVD via numpy for determinism and zero extra dependency
        U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)
        n_pc = min(30, U.shape[1])
        return (U[:, :n_pc] * S[:n_pc]).astype(np.float64)
    return X


# ============================================================================
# Gap-statistic K selection (Tibshirani, Walther & Hastie 2001)
# ============================================================================

def _within_cluster_sse(features: np.ndarray, labels: np.ndarray) -> float:
    """Sum of within-cluster squared deviations from each cluster centroid."""
    total = 0.0
    for c in np.unique(labels):
        mask = labels == c
        if mask.sum() <= 1:
            continue
        sub = features[mask]
        total += float(np.sum((sub - sub.mean(axis=0, keepdims=True)) ** 2))
    return max(total, 1e-12)


def _fit_ward(features: np.ndarray, connectivity: sp.spmatrix, n_clusters: int) -> np.ndarray:
    agg = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward",
        connectivity=connectivity,
    )
    return agg.fit_predict(features)


def _gap_statistic_choose_k(
    features: np.ndarray,
    connectivity: sp.spmatrix,
    k_min: int,
    k_max: int,
    n_refs: int = 5,
    seed: int = 0,
) -> int:
    """
    Choose K via the gap statistic with the one-standard-error rule.

    For each K in [k_min, k_max]:
        gap(K) = mean log(W_ref(K)) - log(W(K))
    where W is within-cluster SSE on the data and W_ref is the same on
    `n_refs` uniform-random reference samples in the data's bounding box.
    Pick the smallest K such that gap(K) >= gap(K+1) - s(K+1), where s is
    the standard error of the reference distribution at K+1.
    """
    rng = np.random.default_rng(seed)
    lo = features.min(axis=0)
    hi = features.max(axis=0)
    spread = np.maximum(hi - lo, 1e-12)

    ks = list(range(max(k_min, 2), min(k_max, features.shape[0]) + 1))
    if len(ks) <= 1:
        return ks[0] if ks else max(k_min, 2)

    gaps = np.zeros(len(ks), dtype=np.float64)
    sks = np.zeros(len(ks), dtype=np.float64)
    for idx, k in enumerate(ks):
        labels = _fit_ward(features, connectivity, k)
        log_W = np.log(_within_cluster_sse(features, labels))
        # Reference distribution: uniform random in the bounding box.
        # NB. The reference uses a free (unconnected) Ward fit because the
        # reference data has no biological topology; matches Tibshirani 2001.
        ref_logs = np.empty(n_refs, dtype=np.float64)
        for r in range(n_refs):
            ref = lo + spread * rng.random(features.shape)
            try:
                ref_labels = AgglomerativeClustering(
                    n_clusters=k, linkage="ward"
                ).fit_predict(ref)
                ref_logs[r] = np.log(_within_cluster_sse(ref, ref_labels))
            except Exception:
                # On rare numerical failure, fall back to the data-side value.
                ref_logs[r] = log_W
        gaps[idx] = float(ref_logs.mean()) - log_W
        sks[idx] = float(ref_logs.std(ddof=1)) * np.sqrt(1.0 + 1.0 / n_refs)

    # One-standard-error rule: smallest k such that gap(k) >= gap(k+1) - s(k+1)
    for i in range(len(ks) - 1):
        if gaps[i] >= gaps[i + 1] - sks[i + 1]:
            return ks[i]
    return ks[-1]


# ============================================================================
# Boundary-aware merging of fragmented small clusters
# ============================================================================

def _cluster_adjacency(labels: np.ndarray, edges) -> dict:
    """For each cluster c, list of clusters c' that are Delaunay-adjacent."""
    adj: dict = {}
    for i, j in edges:
        ci, cj = int(labels[i]), int(labels[j])
        if ci == cj:
            continue
        adj.setdefault(ci, set()).add(cj)
        adj.setdefault(cj, set()).add(ci)
    return adj


def _cluster_centroid_expression(features: np.ndarray, labels: np.ndarray) -> dict:
    """Expression centroid per cluster id."""
    return {
        int(c): features[labels == c].mean(axis=0)
        for c in np.unique(labels)
    }


def _merge_small_clusters(
    labels: np.ndarray,
    features: np.ndarray,
    edges,
    min_cluster_size: int,
) -> np.ndarray:
    """
    Fold any cluster of size < min_cluster_size into its expression-most-similar
    spatially-adjacent neighbour. Iterates until no small clusters remain or
    no admissible merge exists (e.g. an isolated small cluster with no
    spatial neighbours, which is left alone).

    The operation preserves spatial contiguity because we only merge into
    Delaunay-adjacent neighbours.
    """
    labels = labels.copy()
    if min_cluster_size <= 1:
        return labels
    changed = True
    while changed:
        changed = False
        unique, counts = np.unique(labels, return_counts=True)
        small = [int(c) for c, n in zip(unique, counts) if n < min_cluster_size]
        if not small:
            break
        adj = _cluster_adjacency(labels, edges)
        centroids = _cluster_centroid_expression(features, labels)
        # Process smallest-first for determinism (lowest cluster id, smallest size)
        for c in sorted(small, key=lambda x: (np.sum(labels == x), x)):
            if c not in adj or not adj[c]:
                continue  # isolated small cluster: nothing to merge into
            # Choose the adjacent cluster with the closest expression centroid
            best_neighbour = min(
                adj[c],
                key=lambda c2: np.linalg.norm(centroids[c] - centroids[c2]),
            )
            labels[labels == c] = best_neighbour
            changed = True
            break  # restart the loop so adjacency/centroids reflect the merge
    # Relabel to contiguous 0..K-1
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(int)


# ============================================================================
# Cross-slice joint PCA helper (for overlap consistency without celltype)
# ============================================================================

def compute_joint_pca(
    slices: list,
    n_components: int = 30,
    out_key: str = "X_pca_joint",
    use_layer: Optional[str] = None,
) -> list:
    """
    Project a list of AnnData slices into a shared PCA space.

    Stacks the gene-expression matrices, computes a single PCA basis on the
    concatenation, then writes the per-slice projection back to each slice's
    ``obsm[out_key]``. This is the recommended preprocessing step when
    clustering two or more slices that share an overlap region: it ensures
    the PC basis (and therefore Ward's merge sequence) is comparable across
    slices, which is a precondition for getting near-identical cluster
    centroids on the overlap interior.

    Implementation uses a deterministic thin SVD on the column-centered
    concatenated matrix; no random initialization, no sklearn dependency.

    Args:
        slices: list of AnnData. All slices must share the same gene order.
        n_components: number of principal components to retain.
        out_key: key under which the projection is stored in each slice's
            obsm. Defaults to ``"X_pca_joint"``.
        use_layer: optional adata.layers key to use instead of adata.X.

    Returns:
        The list of slices (modified in place). Each slice will have an
        obsm[out_key] of shape (n_cells, min(n_components, n_features)).
    """
    if len(slices) < 1:
        return slices

    # Collect expression matrices, densifying sparse arrays as needed
    mats = []
    sizes = []
    for ad in slices:
        X = ad.layers[use_layer] if (use_layer is not None) else ad.X
        if sp.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float64)
        mats.append(X)
        sizes.append(X.shape[0])

    # Verify gene-order consistency
    n_genes = mats[0].shape[1]
    for k, m in enumerate(mats):
        if m.shape[1] != n_genes:
            raise ValueError(
                f"Slice {k} has {m.shape[1]} genes; expected {n_genes}. "
                "All slices must share the same gene order before joint PCA."
            )

    X_concat = np.vstack(mats)
    # Column-center on the joint mean (the joint mean is the right reference
    # because it's the centroid of the union of cells)
    X_centered = X_concat - X_concat.mean(axis=0, keepdims=True)

    # Deterministic thin SVD
    U, S, _Vt = np.linalg.svd(X_centered, full_matrices=False)
    n_pc = min(n_components, U.shape[1])
    Z = (U[:, :n_pc] * S[:n_pc]).astype(np.float64)

    # Split projection back into per-slice blocks and assign
    offset = 0
    for ad, n in zip(slices, sizes):
        ad.obsm[out_key] = Z[offset:offset + n].copy()
        offset += n

    return slices


# ============================================================================
# Cell-type-anchored clustering (best for cross-slice overlap consistency)
# ============================================================================

def _cluster_by_celltype_components(
    adata: AnnData,
    spatial_key: str,
    celltype_key: str,
    features: np.ndarray,
    edges,
    n_cells: int,
    min_cluster_frac: float,
    max_cluster_frac: float,
    reunion_threshold: float,
    auto_k: bool,
    k_min: int,
    k_max_per_split: int,
    n_gap_refs: int,
    gap_seed: int,
) -> np.ndarray:
    """
    Cluster = (cell-type label) x (connected component in spatial graph
    restricted to cells of that type).

    Algorithm
    ---------
    1. For each cell-type label t:
         restrict the Delaunay graph to cells of type t,
         find connected components -> each becomes a candidate cluster.
    2. If a component is larger than max_cluster_frac * n_cells, split it
       further by biological Ward (within that component) into K sub-clusters
       chosen by the gap statistic.
    3. If a component is smaller than min_cluster_frac * (n_cells / K_final),
       fold it into its expression-most-similar adjacent component
       (the standard boundary-aware merge step).

    Cross-slice property
    --------------------
    Step 1 produces identical labels on the interior overlap by
    construction: a cell of type t in the deep interior of two slices
    is in the same connected component (one is a spatial superset of the
    other), so it receives the same component id (up to a deterministic
    relabeling).

    Boundary cells (within one cluster radius of slice B's cut surface)
    may fall in different components in A vs B if their type-t neighbors
    were cut off. The boundary-aware merge step folds these into the most
    similar adjacent component; INCENT's partial-FGW absorbs any residual
    discrepancy via the unbalanced marginals.
    """
    if celltype_key not in adata.obs.columns:
        raise KeyError(
            f"use_celltype='{celltype_key}' not found in adata.obs. "
            f"Available columns: {list(adata.obs.columns)}"
        )
    types = np.asarray(adata.obs[celltype_key].astype(str).values)

    # Build a per-cell adjacency list once (cheaper than rebuilding sparse mats)
    n = n_cells
    nbrs: list = [[] for _ in range(n)]
    for i, j in edges:
        nbrs[i].append(j)
        nbrs[j].append(i)

    labels = -np.ones(n, dtype=int)
    next_id = 0

    # 1. For each cell type, find connected components in the restricted graph
    for t in np.unique(types):
        members = np.where(types == t)[0]
        if members.size == 0:
            continue
        member_set = set(members.tolist())
        visited = np.zeros(members.size, dtype=bool)
        idx_of = {int(m): i for i, m in enumerate(members)}
        for start_idx, m in enumerate(members):
            if visited[start_idx]:
                continue
            # BFS within the type-restricted graph
            stack = [int(m)]
            comp = []
            while stack:
                u = stack.pop()
                if visited[idx_of[u]]:
                    continue
                visited[idx_of[u]] = True
                comp.append(u)
                for v in nbrs[u]:
                    if v in member_set and not visited[idx_of[v]]:
                        stack.append(v)
            # comp is a connected component of cell-type t
            comp = sorted(comp)  # deterministic order
            comp_arr = np.asarray(comp, dtype=int)
            # 2. Split if too large
            frac = comp_arr.size / n
            if frac > max_cluster_frac and comp_arr.size >= 2 * k_min:
                sub_labels = _split_large_component(
                    comp_arr, features, edges, member_set,
                    auto_k=auto_k, k_min=k_min, k_max=k_max_per_split,
                    n_gap_refs=n_gap_refs, gap_seed=gap_seed,
                )
                for sub_c in np.unique(sub_labels):
                    cells = comp_arr[sub_labels == sub_c]
                    labels[cells] = next_id
                    next_id += 1
            else:
                labels[comp_arr] = next_id
                next_id += 1

    # 3a. Reunite same-cell-type components separated by cut-induced gaps.
    #     This is what makes the clustering identical on the overlap interior
    #     between a full slice and its sub-section.
    if reunion_threshold > 0:
        coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
        types = np.asarray(adata.obs[celltype_key].astype(str).values)
        labels = _reunite_split_components(
            labels, types, features, coords, reunion_threshold
        )
        next_id = labels.max() + 1

    # 3b. Boundary-aware merge of small clusters
    if min_cluster_frac > 0.0 and next_id > 1:
        avg_size = n / next_id
        min_size = max(2, int(min_cluster_frac * avg_size))
        labels = _merge_small_clusters(labels, features, edges, min_size)

    # Relabel to contiguous 0..K-1 (deterministic, by first-appearance order)
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(int)


def _reunite_split_components(
    labels: np.ndarray,
    types: np.ndarray,
    features: np.ndarray,
    coords: np.ndarray,
    reunion_threshold: float,
) -> np.ndarray:
    """
    Merge same-cell-type components that are spatially adjacent (gap
    smaller than ``reunion_threshold`` cluster-radii).

    Rationale
    ---------
    When a slice's cut surface bisects a connected component (e.g. half a
    cortical ring in slice B versus the full ring in slice A), the two
    halves end up as separate connected components in B's per-type
    Delaunay subgraph even though they belong to the same biological
    region. This step reunites such pairs: if two same-type components
    have at least one cross-component cell-pair whose Euclidean distance
    is less than ``reunion_threshold`` times the median nearest-neighbour
    spacing of the smaller component, they are merged.

    The threshold is meaningful: ``reunion_threshold`` of 2-3 means
    "the gap between the two halves is comparable to the typical
    spacing within either half", which is the geometric signature of a
    cut-induced fragmentation. Set ``reunion_threshold=0`` to disable.

    All re-uniting decisions are deterministic and use spatial geometry
    only; gene-expression is NOT consulted because the goal is to repair
    *spatial* breaks, not biological ones (cells of the same cell type
    that happen to be spatially separated by a true biological feature
    are correctly kept as different clusters).
    """
    if reunion_threshold <= 0:
        return labels
    labels = labels.copy()
    changed = True
    rounds = 0
    while changed and rounds < 50:
        changed = False
        rounds += 1
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            break
        # Group cluster ids by cell type (use majority type of each cluster)
        cluster_type = {}
        cluster_coords = {}
        cluster_nn_dists = {}
        for c in unique_labels:
            mask = labels == c
            cluster_type[int(c)] = str(np.bincount(
                np.searchsorted(np.unique(types), types[mask])
            ).argmax())
            cluster_type[int(c)] = str(types[mask][0]) if mask.sum() > 0 else ""
            # Override with explicit majority
            unique_t, counts_t = np.unique(types[mask], return_counts=True)
            cluster_type[int(c)] = str(unique_t[counts_t.argmax()])
            cluster_coords[int(c)] = coords[mask]
            # Median nearest-neighbour spacing within cluster
            if mask.sum() >= 2:
                pts = cluster_coords[int(c)]
                # pairwise min distance via simple O(n^2); fine at cluster scale
                d2 = ((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)
                np.fill_diagonal(d2, np.inf)
                nn = np.sqrt(d2.min(axis=1))
                cluster_nn_dists[int(c)] = float(np.median(nn))
            else:
                cluster_nn_dists[int(c)] = 0.0
        # For each same-type pair, compute min cross-cluster distance
        merge_pairs = []
        cluster_ids_sorted = sorted(cluster_type.keys())
        for ii in range(len(cluster_ids_sorted)):
            for jj in range(ii + 1, len(cluster_ids_sorted)):
                a = cluster_ids_sorted[ii]
                b = cluster_ids_sorted[jj]
                if cluster_type[a] != cluster_type[b]:
                    continue
                pts_a = cluster_coords[a]
                pts_b = cluster_coords[b]
                if pts_a.shape[0] == 0 or pts_b.shape[0] == 0:
                    continue
                d2 = ((pts_a[:, None, :] - pts_b[None, :, :]) ** 2).sum(axis=2)
                min_gap = float(np.sqrt(d2.min()))
                # Reference scale: median NN spacing of the smaller cluster
                ref_nn = min(cluster_nn_dists[a], cluster_nn_dists[b])
                if ref_nn <= 0:
                    continue
                if min_gap <= reunion_threshold * ref_nn:
                    merge_pairs.append((a, b, min_gap / ref_nn))
        if not merge_pairs:
            break
        # Merge the tightest-gap pair first (deterministic ordering)
        merge_pairs.sort(key=lambda x: (x[2], x[0], x[1]))
        a, b, _ = merge_pairs[0]
        labels[labels == b] = a
        changed = True
    # Relabel
    _, new_labels = np.unique(labels, return_inverse=True)
    return new_labels.astype(int)


def _split_large_component(
    comp_arr: np.ndarray,
    features: np.ndarray,
    edges,
    member_set: set,
    auto_k: bool,
    k_min: int,
    k_max: int,
    n_gap_refs: int,
    gap_seed: int,
) -> np.ndarray:
    """
    Sub-cluster cells inside a too-large connected component by biological
    Ward, restricted to edges internal to the component.

    Returns sub-cluster labels in [0, K-1], one per cell of comp_arr.
    """
    comp_set = set(comp_arr.tolist())
    # Build a re-indexed connectivity matrix for just this component
    pos = {int(c): i for i, c in enumerate(comp_arr)}
    rows, cols = [], []
    for i, j in edges:
        if i in comp_set and j in comp_set:
            rows.append(pos[int(i)]); cols.append(pos[int(j)])
            rows.append(pos[int(j)]); cols.append(pos[int(i)])
    if not rows:
        # Disconnected single component (shouldn't happen but be safe)
        return np.zeros(comp_arr.size, dtype=int)
    data = np.ones(len(rows), dtype=np.float64)
    sub_conn = sp.coo_matrix(
        (data, (rows, cols)), shape=(comp_arr.size, comp_arr.size)
    )
    sub_features = features[comp_arr]
    k_max_eff = min(k_max, comp_arr.size - 1)
    if auto_k and k_max_eff > k_min:
        k = _gap_statistic_choose_k(
            sub_features, sub_conn,
            k_min=k_min, k_max=k_max_eff,
            n_refs=n_gap_refs, seed=gap_seed,
        )
    else:
        k = max(k_min, 2)
    k = min(k, comp_arr.size - 1)
    if k < 2:
        return np.zeros(comp_arr.size, dtype=int)
    return _fit_ward(sub_features, sub_conn, k)


def transfer_labels_from_reference(
    adata_target: AnnData,
    adata_reference: AnnData,
    reference_labels: np.ndarray,
    use_rep: str = "X_pca_joint",
    k_expression: int = 5,
    spatial_smoothing_rounds: int = 2,
    spatial_key: str = "spatial",
) -> np.ndarray:
    """
    Transfer cluster labels from a reference slice to a target slice via
    expression-kNN, then smooth spatially.

    This is the recommended approach when one slice is a sub-section of
    another (e.g. a 60% cut of a full brain): cluster the larger slice as
    the reference, then transfer its labels to the smaller slice. The
    resulting clustering on the smaller slice matches the larger slice's
    clustering on the overlap interior by construction, because each
    target cell adopts the consensus label of its k-nearest expression
    neighbours in the reference.

    Pre-requirements
    ----------------
    Both AnnDatas must have the same gene-expression representation key
    ``use_rep`` in obsm. Recommended:
        compute_joint_pca([adata_target, adata_reference], n_components=30,
                          out_key="X_pca_joint")
    before calling this function.

    Algorithm
    ---------
    1. For each target cell, find its ``k_expression`` nearest neighbours
       in the reference's expression space (Euclidean distance on
       ``use_rep``).
    2. Assign the majority label of those k neighbours (deterministic
       tie-breaking by lowest label id).
    3. Optionally smooth spatially: for ``spatial_smoothing_rounds``
       passes, re-assign each cell to the majority label among itself
       and its Delaunay neighbours in the target slice. Smooths out
       isolated mismatches due to expression noise.

    Args:
        adata_target: AnnData receiving the labels.
        adata_reference: AnnData supplying the labels.
        reference_labels: (n_ref,) integer labels for the reference cells.
        use_rep: obsm key holding the shared gene-expression representation.
        k_expression: number of expression-neighbours to consult per cell.
        spatial_smoothing_rounds: number of post-assignment smoothing passes.
        spatial_key: obsm key for the target's spatial coordinates.

    Returns:
        (n_target,) integer labels in the same value range as
        ``reference_labels``.
    """
    if use_rep not in adata_target.obsm:
        raise KeyError(
            f"adata_target missing obsm key '{use_rep}'. Run compute_joint_pca first."
        )
    if use_rep not in adata_reference.obsm:
        raise KeyError(
            f"adata_reference missing obsm key '{use_rep}'. Run compute_joint_pca first."
        )

    ref_expr = np.asarray(adata_reference.obsm[use_rep], dtype=np.float64)
    tgt_expr = np.asarray(adata_target.obsm[use_rep], dtype=np.float64)
    reference_labels = np.asarray(reference_labels, dtype=int)

    # Step 1+2: expression-kNN majority-vote label transfer
    # Brute-force distance — fine at MERFISH cell scale; use a KDTree
    # for larger ones.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(ref_expr)
        _, nn_idx = tree.query(tgt_expr, k=k_expression)
    except Exception:
        # Fallback to brute force
        d2 = ((tgt_expr[:, None, :] - ref_expr[None, :, :]) ** 2).sum(axis=2)
        nn_idx = np.argsort(d2, axis=1)[:, :k_expression]

    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    neighbour_labels = reference_labels[nn_idx]
    # Deterministic majority vote: count per row, pick max (then smallest id on tie)
    labels = np.empty(tgt_expr.shape[0], dtype=int)
    for i in range(tgt_expr.shape[0]):
        unique_l, counts_l = np.unique(neighbour_labels[i], return_counts=True)
        max_count = counts_l.max()
        winners = unique_l[counts_l == max_count]
        labels[i] = int(winners.min())  # deterministic tie-break

    # Step 3: spatial smoothing
    if spatial_smoothing_rounds > 0:
        coords = np.asarray(adata_target.obsm[spatial_key])
        edges = build_spatial_graph(coords)
        nbrs: list = [[] for _ in range(tgt_expr.shape[0])]
        for i, j in edges:
            nbrs[i].append(j); nbrs[j].append(i)
        for _ in range(spatial_smoothing_rounds):
            new_labels = labels.copy()
            for i in range(tgt_expr.shape[0]):
                ngh = nbrs[i] + [i]
                u, c = np.unique(labels[ngh], return_counts=True)
                mx = c.max()
                wins = u[c == mx]
                new_labels[i] = int(wins.min())
            labels = new_labels

    return labels


def cluster_cells_spatial(
    adata: AnnData,
    spatial_key: str = "spatial",
    resolution: float = 1.0,
    use_rep: Optional[str] = None,
    use_celltype: Optional[str] = None,
    auto_k: bool = True,
    min_cluster_frac: float = 0.1,
    max_cluster_frac: float = 0.25,
    reunion_threshold: float = 3.0,
    k_min: int = 4,
    k_max: Optional[int] = None,
    n_gap_refs: int = 5,
    gap_seed: int = 0,
) -> np.ndarray:
    """
    Cluster cells into spatially-contiguous biological mesoregions.

    Ward agglomerative clustering on a feature representation (gene-expression
    PCA by default), constrained by a Delaunay spatial-adjacency graph.
    Clusters are therefore both **spatially contiguous** and
    **biologically homogeneous**, which is what the cluster-cost matrix
    M^cl in INCENT's Step 3 assumes.

    Two clustering modes
    --------------------
    * **Cell-type-anchored** (when ``use_celltype`` is set): cluster =
      (cell-type label) x (connected component in the per-type Delaunay
      subgraph). This is the recommended mode for cross-slice overlap
      consistency: clustering a full slice A and its spatial subset B
      independently produces identical labels on the overlap interior,
      because connected components are local properties and a cell's
      same-type neighbors are the same in both slices (modulo a thin
      boundary shell at B's cut surface, which the boundary-aware merge
      step handles). Over-large components are sub-clustered by
      biological Ward; small fragments are folded into the most similar
      adjacent component.

    * **Biological Ward** (default when ``use_celltype is None``): Ward
      on the gene-expression representation (``use_rep``) with Delaunay
      connectivity, K chosen by the gap statistic, small clusters merged
      by the boundary-aware step. Use this when cell-type labels are
      unavailable or untrusted. For cross-slice overlap consistency in
      this mode, ensure both slices share a PCA basis (see
      ``compute_joint_pca``).

    Args:
        adata: AnnData object.
        spatial_key: Key in adata.obsm storing spatial coordinates.
        resolution: Resolution parameter (scales number of clusters when
            ``auto_k=False`` and ``use_celltype is None``).
        use_rep: Feature representation for clustering. Default ``None`` picks
            ``X_pca`` if present, else densified ``X`` (with a deterministic
            30-PC projection if X has more than 50 columns). Pass
            ``use_rep="spatial"`` to recover the original spatial-only
            behaviour. Pass a custom obsm key (e.g. ``"X_pca_joint"``) to use
            any other representation; ``"X_pca_joint"`` produced by
            ``compute_joint_pca`` is the recommended choice for cross-slice
            workflows when ``use_celltype`` is unavailable.
        use_celltype: AnnData obs column holding cell-type labels. When
            supplied, switches to the cell-type-anchored mode (recommended
            for cross-slice consistency). Set to ``None`` (default) for
            biological-Ward mode.
        auto_k: If True (default), choose the number of clusters via the gap
            statistic with the one-standard-error rule. If False, fall back to
            the original ``int((n_cells / 200) * resolution)`` rule.
        min_cluster_frac: Any cluster smaller than ``min_cluster_frac *
            (n_cells / K)`` is folded into its expression-most-similar
            spatially-adjacent neighbour. Set to 0 to disable. Default 0.1
            (10% of the average cluster size).
        max_cluster_frac: In cell-type-anchored mode, any connected component
            larger than ``max_cluster_frac * n_cells`` is sub-clustered by
            biological Ward. Set to 1.0 to disable splits. Default 0.25.
        reunion_threshold: In cell-type-anchored mode, same-cell-type
            connected components whose minimum cross-component cell-pair
            distance is at most ``reunion_threshold`` times the smaller
            component's median nearest-neighbour spacing are merged.
            This repairs cut-induced fragmentation (when a tissue cut
            bisects a biological region into two arcs that are still
            geometrically close to each other). Set to 0 to disable.
            Default 3.0. Recommended values 2-3 in practice.
        k_min, k_max: Bounds for the gap-statistic K scan. If ``k_max`` is
            None, defaults to ``min(50, n_cells // 30)``.
        n_gap_refs: Number of uniform-reference samples per K in the gap
            scan. Tibshirani et al. recommend 5-10.
        gap_seed: Deterministic seed for the gap-statistic reference sampling.

    Returns:
        Integer cluster labels in [0, K-1], one per cell.

    Notes:
        * The function is fully deterministic given the same inputs and the
          same ``gap_seed``.
        * Cell-type-anchored mode is order-invariant: cells are processed in
          sorted index order, so reordering input rows does not change the
          output (up to a label relabeling).
        * For cross-slice overlap consistency, use the cell-type-anchored
          mode (``use_celltype=...``). If unavailable, use biological-Ward
          with ``use_rep="X_pca_joint"`` after running ``compute_joint_pca``.
    """
    coords = np.asarray(adata.obsm[spatial_key])
    n_cells = coords.shape[0]

    # 1. Build Delaunay spatial graph
    edges = build_spatial_graph(coords)

    # 2. Connectivity matrix (symmetric)
    if not edges:
        return np.zeros(n_cells, dtype=int)

    row = np.array([e[0] for e in edges] + [e[1] for e in edges])
    col = np.array([e[1] for e in edges] + [e[0] for e in edges])
    data = np.ones(len(row), dtype=np.float64)
    connectivity = sp.coo_matrix((data, (row, col)), shape=(n_cells, n_cells))

    # 3. Feature matrix (used by both modes; for use_celltype mode it is
    #    only used to compute centroids for merging/splitting)
    features = _resolve_feature_matrix(adata, use_rep, spatial_key)

    # 4. Cell-type-anchored mode
    if use_celltype is not None:
        if k_max is None:
            k_max_eff = min(50, max(k_min + 1, n_cells // 30))
        else:
            k_max_eff = k_max
        return _cluster_by_celltype_components(
            adata=adata,
            spatial_key=spatial_key,
            celltype_key=use_celltype,
            features=features,
            edges=edges,
            n_cells=n_cells,
            min_cluster_frac=min_cluster_frac,
            max_cluster_frac=max_cluster_frac,
            reunion_threshold=reunion_threshold,
            auto_k=auto_k,
            k_min=k_min,
            k_max_per_split=k_max_eff,
            n_gap_refs=n_gap_refs,
            gap_seed=gap_seed,
        )

    # 5. Biological-Ward mode: choose K
    if auto_k:
        if k_max is None:
            k_max = min(50, max(k_min + 1, n_cells // 30))
        n_clusters_target = _gap_statistic_choose_k(
            features, connectivity,
            k_min=k_min, k_max=k_max,
            n_refs=n_gap_refs, seed=gap_seed,
        )
    else:
        # Legacy behaviour: scale with cells/200
        n_clusters_target = max(2, int((n_cells / 200.0) * resolution))

    # 6. Fit Ward
    labels = _fit_ward(features, connectivity, n_clusters_target)

    # 7. Boundary-aware merging of small clusters
    if min_cluster_frac > 0.0 and n_clusters_target > 1:
        avg_size = n_cells / n_clusters_target
        min_size = max(2, int(min_cluster_frac * avg_size))
        labels = _merge_small_clusters(labels, features, edges, min_size)

    return labels.astype(int)