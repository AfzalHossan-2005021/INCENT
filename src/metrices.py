"""
Alignment quality metrics for INCENT.

This module computes every alignment-quality metric required for a
publication-grade benchmark of a partial/unbalanced ST transport plan:

Per-pair quality metrics
------------------------
* Neighborhood JSD and gene-expression cosine objectives (mass-normalised
  so unbalanced plans remain comparable to balanced ones).
* Probabilistic cell-type matching.
* Pairwise Alignment Accuracy (PAA) -- the canonical PASTE/PASTE2 metric.
* Label-Transfer Adjusted Rand Index (LTARI).
* Fraction Of Samples Closer Than the True Match (FOSCTTM).
* Landmark Euclidean error (mean, median, RMSE) when ground-truth landmark
  pairs are supplied.
* Spatial Coherence Score (SCS) of transferred labels.

Compactness / variance diagnostics
----------------------------------
* Forward/reverse spatial compactness and effective support (preserved from
  the original module).

All functions accept either a balanced (sum=1) or unbalanced (sum<=1) pi and
behave gracefully when pi is sparse.

The headline entry point is ``calculate_performance_metrics``, which now
computes every metric the available data supports and prints them in a
single, formatted table. The original return-dictionary keys are preserved,
and ground-truth-dependent metrics are added when their inputs are passed.
"""

from __future__ import annotations

import numpy as np
import torch

from typing import Optional, Sequence

from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import euclidean_distances

from .utils import select_backend
from .core import (
    calculate_neighborhood_dissimilarity,
    calculate_gene_expression_cosine_distance,
    calculate_cell_type_mismatch,
    estimate_characteristic_spacing
)


# ============================================================================
# Cost-based objectives (mass-normalised, unbalanced-safe)
# ============================================================================

def calculate_neighborhood_similarity(js_dist_neighborhood, pi, normalize_mass: bool = True):
    """
    Sum of element-wise JSD * pi, optionally normalised by total transported mass.

    For unbalanced pi this raw sum is not comparable across methods because it
    scales with total mass. Setting ``normalize_mass=True`` (recommended default
    in the benchmark) divides by sum(pi) so the value is comparable.
    """
    total = float(np.sum(pi))
    raw = float(np.sum(js_dist_neighborhood * pi))
    if normalize_mass and total > 1e-12:
        return raw / total
    return raw


def calculate_gene_expression_similarity(cosine_dist_gene_expr, pi, normalize_mass: bool = True):
    """
    Sum of element-wise cosine-distance * pi, optionally mass-normalised.
    """
    total = float(np.sum(pi))
    raw = float(np.sum(cosine_dist_gene_expr * pi))
    if normalize_mass and total > 1e-12:
        return raw / total
    return raw


def cell_type_matching(cell_type_mismatch, pi_mat):
    """
    Fraction of transported mass that lands on the correct cell-type.

    Returns a value in [0, 1]: 1.0 means every unit of transported mass
    couples cells of the same type; 0.0 means every unit couples different
    types. Computed as <(1 - mismatch), pi> / <1, pi>.
    """
    M_match = 1.0 - cell_type_mismatch
    expected_matches = float(np.sum(M_match * pi_mat))
    total_mass = float(np.sum(pi_mat))
    if total_mass > 0:
        return expected_matches / total_mass
    return 0.0


# ============================================================================
# Pairwise Alignment Accuracy (PAA)
# ============================================================================

def pairwise_alignment_accuracy(
    labels_A: Sequence,
    labels_B: Sequence,
    pi: np.ndarray,
    weighted: bool = True,
) -> float:
    """
    Canonical PAA metric (PASTE/PASTE2).

    For each source cell, the predicted target label is the one receiving the
    largest transported mass; the source is counted correct if its true label
    equals the predicted target label. Sources with no transported mass are
    skipped. Weighted mode (default) uses each source's row mass as its
    contribution, matching PASTE2's canonical formulation under unbalanced
    transport.

    Returns
    -------
    accuracy in [0, 1].
    """
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)
    pi = np.asarray(pi, dtype=np.float64)
    if pi.shape != (labels_A.size, labels_B.size):
        raise ValueError(
            f"pi shape {pi.shape} does not match labels "
            f"({labels_A.size}, {labels_B.size})"
        )
    row_mass = pi.sum(axis=1)
    active = row_mass > 1e-12
    if not active.any():
        return 0.0
    best_tgt = np.argmax(pi[active], axis=1)
    pred_labels = labels_B[best_tgt]
    true_labels = labels_A[active]
    correct = (pred_labels == true_labels).astype(np.float64)
    if weighted:
        w = row_mass[active]
        return float(np.sum(correct * w) / np.sum(w))
    return float(np.mean(correct))


# ============================================================================
# Label-Transfer Adjusted Rand Index (LTARI)
# ============================================================================

def label_transfer_ari(
    labels_A: Sequence,
    labels_B: Sequence,
    pi: np.ndarray,
) -> float:
    """
    Transfer target labels to source via pi, then compute ARI versus true
    source labels. Sources with no transported mass are excluded.
    """
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)
    pi = np.asarray(pi, dtype=np.float64)
    row_mass = pi.sum(axis=1)
    active = row_mass > 1e-12
    if not active.any():
        return 0.0
    pred = labels_B[np.argmax(pi[active], axis=1)]
    return float(adjusted_rand_score(labels_A[active], pred))


# ============================================================================
# FOSCTTM
# ============================================================================

def foscttm(
    coords_A_aligned: np.ndarray,
    coords_B_aligned: np.ndarray,
    true_matches: Optional[np.ndarray] = None,
) -> float:
    """
    Fraction Of Samples Closer Than the True Match.

    Given aligned coordinates (post-Procrustes) of two slices that share a
    known one-to-one correspondence, return the average fraction of off-target
    points closer to a query than its true partner. Lower is better; 0.5 is
    random, 0 is perfect.

    Parameters
    ----------
    coords_A_aligned, coords_B_aligned : (n, d) arrays in a shared frame.
    true_matches : optional (n,) array mapping source index -> target index.
        Defaults to identity (the two arrays are already paired by index).
    """
    A = np.asarray(coords_A_aligned, dtype=np.float64)
    B = np.asarray(coords_B_aligned, dtype=np.float64)
    n = A.shape[0]
    if B.shape[0] != n:
        raise ValueError("FOSCTTM requires equal-sized paired sets.")
    if true_matches is None:
        true_matches = np.arange(n)
    D = euclidean_distances(A, B)
    true_d = D[np.arange(n), true_matches]
    frac_AB = np.mean(D < true_d[:, None], axis=1)
    inv = np.argsort(true_matches)
    Dt = D.T
    true_d_BA = Dt[np.arange(n), inv]
    frac_BA = np.mean(Dt < true_d_BA[:, None], axis=1)
    return float(0.5 * (np.mean(frac_AB) + np.mean(frac_BA)))


# ============================================================================
# Landmark error
# ============================================================================

def landmark_error(
    coords_A_aligned: np.ndarray,
    coords_B_aligned: np.ndarray,
    landmark_pairs: np.ndarray,
    reduction: str = "mean",
) -> float:
    """
    Mean / median / RMSE Euclidean distance between paired landmarks after
    alignment.

    Parameters
    ----------
    landmark_pairs : (K, 2) int array. Each row is (i_in_A, j_in_B).
    reduction : 'mean' | 'median' | 'rmse'.
    """
    pairs = np.asarray(landmark_pairs)
    if pairs.size == 0:
        return float("nan")
    A = np.asarray(coords_A_aligned)[pairs[:, 0]]
    B = np.asarray(coords_B_aligned)[pairs[:, 1]]
    d = np.linalg.norm(A - B, axis=1)
    if reduction == "mean":
        return float(np.mean(d))
    if reduction == "median":
        return float(np.median(d))
    if reduction == "rmse":
        return float(np.sqrt(np.mean(d ** 2)))
    raise ValueError(reduction)


# ============================================================================
# Spatial Coherence Score
# ============================================================================

def spatial_coherence_score(
    coords: np.ndarray,
    labels: Sequence,
    k: int = 6,
) -> float:
    """
    Fraction of k-nearest neighbours sharing the same label.

    A high SCS for transferred labels at the source positions indicates that
    the alignment preserves spatial-domain structure.
    """
    coords = np.asarray(coords, dtype=np.float64)
    labels = np.asarray(labels)
    if coords.shape[0] < k + 1:
        return float("nan")
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)
    neighbours = idx[:, 1:]
    same = (labels[neighbours] == labels[:, None]).astype(np.float64)
    return float(np.mean(same))


# ============================================================================
# Helper for the headline aggregator: optional coordinate alignment
# ============================================================================

def _post_procrustes_coords(sliceA, sliceB, pi):
    """
    Project A and B into a shared frame via the pi-induced Procrustes,
    falling back to original obsm['spatial'] if visualize.stack_slices_pairwise
    is unavailable or fails.
    """
    try:
        from .visualize import stack_slices_pairwise
        aligned = stack_slices_pairwise([sliceA, sliceB], [pi], output_params=False)
        return (
            np.asarray(aligned[0].obsm["spatial"]),
            np.asarray(aligned[1].obsm["spatial"]),
        )
    except Exception:
        return (
            np.asarray(sliceA.obsm["spatial"]),
            np.asarray(sliceB.obsm["spatial"]),
        )


# ============================================================================
# Compactness / variance diagnostics
# ============================================================================

def calculate_forward_reverse_compactness(
    pi_mat,
    sliceA,
    sliceB,
    spatial_key: str = "spatial",
    k: int = 6,
    verbose: bool = True,
):
    """
    Diagnostic metric for spatial compactness of a transport plan.

    Measures how spatially coherent each cell's mass assignment is in both
    the forward (source → target) and reverse (target → source) directions.
    All variance outputs are normalised by σ², where σ is the characteristic
    cell spacing estimated from the two slices, making values dataset-agnostic
    and directly comparable across experiments and competing methods.

    Args:
        pi_mat      : Alignment mapping matrix (ns × nt). Rows = source cells,
                      columns = target cells.
        sliceA      : Source AnnData object. Must contain spatial coordinates
                      in .obsm[spatial_key].
        sliceB      : Target AnnData object. Must contain spatial coordinates
                      in .obsm[spatial_key].
        spatial_key : Key in .obsm that holds 2-D spatial coordinates.
                      Default: 'spatial'.
        k           : Number of nearest neighbours used by
                      estimate_characteristic_spacing() to compute σ.
                      Default: 6 (consistent with calculate_spatial_distance).
        verbose     : If True, prints a formatted interpretation table.
                      Default: True.

    Returns:
        dict with keys:
            'sigma'               – Characteristic cell spacing σ (raw units).
                                    Divide any raw variance by σ² to reproduce
                                    the normalised values below.
            'forward_compactness' – Mean spatial variance of target coordinates
                                    per source cell, normalised by σ².
                                    Low (< 1.0) = spatially coherent, one-to-one-
                                    like mapping. High (> 9.0) ≈ random/PASTE.
            'reverse_compactness' – Same, transposed: variance of source
                                    coordinates per target cell, normalised by σ².
            'asymmetry_ratio'     – forward_compactness / reverse_compactness.
                                    ≈ 1.0  → balanced, bijective alignment.
                                    >> 1.0 → many sources fan out to few targets
                                              (possible tissue loss / cell death).
                                    << 1.0 → one source maps to many scattered
                                              targets (possible proliferation).
            'locality_score_fwd'  – 1 / (1 + forward_compactness), mapped to
                                    [0, 1]. Higher is better. Convenient scalar
                                    for tables and figures.
            'locality_score_rev'  – 1 / (1 + reverse_compactness), same scale.
            'effective_support_fwd' – Mean participation ratio (row-wise):
                                    (Σ_j π_ij)² / Σ_j π_ij².  ≈ 1 means
                                    near-hard assignment; ≈ n_target means
                                    uniform mass spreading.
            'effective_support_rev' – Same, column-wise (target perspective).

    Interpretation guide (normalised compactness values):
        < 1.0   Excellent  — mapping lands within ~1 cell spacing
        1 – 4   Good       — soft but spatially focused assignment
        4 – 9   Poor       — diffuse, spanning several cell diameters
        > 9     Random     — comparable to a uniform / PASTE-style plan
    """
    pi_mat = np.asarray(pi_mat)
    eps = 1e-12

    # ── Characteristic spacing (same estimator used in calculate_spatial_distance) ──
    sigma_s = estimate_characteristic_spacing(sliceA, k=k, spatial_key=spatial_key)
    sigma_t = estimate_characteristic_spacing(sliceB, k=k, spatial_key=spatial_key)
    sigma   = max(sigma_s, sigma_t, eps)

    # ── Normalise coordinates so variances are in units of σ² ──
    Xs = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64) / sigma
    Xt = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64) / sigma

    # ── Forward compactness: variance of target coords per source cell ──
    pi_row_sums      = pi_mat.sum(axis=1)
    pi_row_sums_safe = np.maximum(pi_row_sums, eps)
    pi_row_norm      = pi_mat / pi_row_sums_safe[:, None]

    bary_t        = pi_row_norm @ Xt                                  # (ns, 2)
    E_x2_fwd      = pi_row_norm @ np.sum(Xt ** 2, axis=1)            # E[||x||²]
    Ex_2_fwd      = np.sum(bary_t ** 2, axis=1)                      # ||E[x]||²

    active_src    = pi_row_sums > eps
    var_fwd       = np.clip(E_x2_fwd[active_src] - Ex_2_fwd[active_src], 0.0, None)
    forward_compactness = float(np.mean(var_fwd)) if var_fwd.size > 0 else 0.0

    # ── Reverse compactness: variance of source coords per target cell ──
    pi_col_sums      = pi_mat.sum(axis=0)
    pi_col_sums_safe = np.maximum(pi_col_sums, eps)
    pi_col_norm      = pi_mat / pi_col_sums_safe[None, :]

    bary_s        = pi_col_norm.T @ Xs                                # (nt, 2)
    E_x2_rev      = pi_col_norm.T @ np.sum(Xs ** 2, axis=1)
    Ex_2_rev      = np.sum(bary_s ** 2, axis=1)

    active_tgt    = pi_col_sums > eps
    var_rev       = np.clip(E_x2_rev[active_tgt] - Ex_2_rev[active_tgt], 0.0, None)
    reverse_compactness = float(np.mean(var_rev)) if var_rev.size > 0 else 0.0

    # ── Asymmetry ratio ──
    if reverse_compactness > eps:
        asymmetry_ratio = forward_compactness / reverse_compactness
    else:
        asymmetry_ratio = float("nan")

    # ── Locality scores (higher = better, bounded [0, 1]) ──
    locality_score_fwd = 1.0 / (1.0 + forward_compactness)
    locality_score_rev = 1.0 / (1.0 + reverse_compactness)

    # ── Effective support (participation ratio) ──
    eff_sup_fwd = (pi_row_sums ** 2) / np.maximum(
        np.sum(pi_mat ** 2, axis=1), eps
    )
    eff_sup_rev = (pi_col_sums ** 2) / np.maximum(
        np.sum(pi_mat ** 2, axis=0), eps
    )
    mean_eff_sup_fwd = (
        float(np.mean(eff_sup_fwd[active_src])) if eff_sup_fwd[active_src].size > 0 else 0.0
    )
    mean_eff_sup_rev = (
        float(np.mean(eff_sup_rev[active_tgt])) if eff_sup_rev[active_tgt].size > 0 else 0.0
    )

    # ── Qualitative label ──
    def _label(fc):
        if   fc < 1.0: return "Excellent"
        elif fc < 4.0: return "Good"
        elif fc < 9.0: return "Poor"
        else:          return "Random-equivalent"

    # ── Optional verbose table ──
    if verbose:
        asym_str = f"{asymmetry_ratio:.4f}" if not (asymmetry_ratio != asymmetry_ratio) else "n/a"
        print("\n" + "=" * 80)
        title = "SPATIAL COMPACTNESS METRICS"
        print(" " * ((80 - len(title)) // 2) + title)
        print("=" * 80)
        print(f"  Characteristic cell spacing σ : {sigma:.4f} (raw coordinate units)")
        print(f"  All variance values are normalised by σ² = {sigma**2:.4f}")
        print("-" * 80)
        print(f"  {'Metric':<38} {'Value':>12}  {'Interpretation'}")
        print("-" * 80)
        print(f"  {'Forward compactness (σ²)':<38} {forward_compactness:>12.4f}  {_label(forward_compactness)}")
        print(f"  {'Reverse compactness (σ²)':<38} {reverse_compactness:>12.4f}  {_label(reverse_compactness)}")
        print(f"  {'Asymmetry ratio (fwd / rev)':<38} {asym_str:>12}  {'≈1.0 balanced | >1 death | <1 prolif.'}")
        print(f"  {'Locality score fwd  [0→1]':<38} {locality_score_fwd:>12.4f}  {'higher is better'}")
        print(f"  {'Locality score rev  [0→1]':<38} {locality_score_rev:>12.4f}  {'higher is better'}")
        print(f"  {'Effective support fwd (cells)':<38} {mean_eff_sup_fwd:>12.2f}  {'1=hard assign | →n_t=uniform'}")
        print(f"  {'Effective support rev (cells)':<38} {mean_eff_sup_rev:>12.2f}  {'1=hard assign | →n_s=uniform'}")
        print("=" * 80)
        print(f"  Guide: <1.0 Excellent | 1–4 Good | 4–9 Poor | >9 Random/PASTE-equivalent")
        print("=" * 80 + "\n")

    return {
        "sigma":                sigma,
        "forward_compactness":  forward_compactness,
        "reverse_compactness":  reverse_compactness,
        "asymmetry_ratio":      asymmetry_ratio,
        "locality_score_fwd":   locality_score_fwd,
        "locality_score_rev":   locality_score_rev,
        "effective_support_fwd": mean_eff_sup_fwd,
        "effective_support_rev": mean_eff_sup_rev,
    }


# ============================================================================
# Pretty printing helper
# ============================================================================

def _print_table(rows, title: str = "ALIGNMENT QUALITY METRICS", width: int = 92):
    """
    rows: iterable of (metric_label, initial_value, final_value, hint).
    initial_value may be None to indicate "no initial value applicable".
    hint is a short string appended at the end of the row (e.g. ' (lower=better)')
    and may be ''.
    """
    bar = "=" * width
    sub = "-" * width

    def _fmt(v):
        if v is None:
            return f"{'-':>14}"
        if isinstance(v, float) and np.isnan(v):
            return f"{'nan':>14}"
        try:
            return f"{float(v):>14.6f}"
        except Exception:
            return f"{str(v):>14}"

    print()
    print(bar)
    print(" " * ((width - len(title)) // 2) + title)
    print(bar)
    print(f"{'Metric':<48}{'Initial':>14}{'Final':>14}  {'Note'}")
    print(sub)
    for label, init_v, fin_v, hint in rows:
        print(f"{label:<48}{_fmt(init_v)}{_fmt(fin_v)}  {hint}")
    print(bar)
    print()


# ============================================================================
# Headline aggregator
# ============================================================================

def calculate_performance_metrics(
    final_pi,
    init_pi=None,
    js_dist_neighborhood=None,
    cosine_dist_gene_expr=None,
    cell_type_mismatch=None,
    sliceA=None,
    sliceB=None,
    use_rep=None,
    radius=100.0,
    use_gpu=True,
    normalize_mass: bool = True,
    label_key: str = "cell_type_annot",
    ground_truth_pairs: Optional[np.ndarray] = None,
    landmark_pairs: Optional[np.ndarray] = None,
    verbose: bool = True,
):
    """
    Compute every alignment-quality metric the available inputs allow, print
    a single formatted table, and return them in a dictionary.

    Backward-compatible signature: every original keyword argument is
    preserved, the original returned keys (``initial_obj_neighbor``,
    ``final_obj_neighbor``, ``initial_obj_gene``, ``final_obj_gene``,
    ``initial_cell_type_match``, ``final_cell_type_match``) are still
    populated.

    New optional inputs
    -------------------
    label_key : str
        AnnData ``obs`` column to use as cell-type labels (defaults to
        ``"cell_type_annot"``). Required for PAA / LTARI / SCS.
    ground_truth_pairs : optional (K, 2) int array of (i_in_A, j_in_B)
        one-to-one correspondences. Enables FOSCTTM (which uses every pair)
        and landmark error (uses up to 50 random pairs).
    landmark_pairs : optional (K, 2) int array of landmark pairs. If supplied,
        used directly for landmark error; otherwise a 50-pair subsample of
        ``ground_truth_pairs`` is used.
    normalize_mass : whether to divide JSD/cosine costs by total transported
        mass (default True). Set to False for back-compatible raw sums.

    Output dictionary keys
    ----------------------
    Original keys:
        ``initial_obj_neighbor``, ``final_obj_neighbor``,
        ``initial_obj_gene``,     ``final_obj_gene``,
        ``initial_cell_type_match``, ``final_cell_type_match``
    Label-aware (added if labels available):
        ``paa_celltype``, ``ltari_celltype``, ``scs_transferred``
    Ground-truth-aware (added if ground_truth_pairs supplied):
        ``foscttm``, ``landmark_mean_err``, ``landmark_median_err``,
        ``landmark_rmse``
    """
    final_pi = np.asarray(final_pi)
    if init_pi is None:
        init_pi = np.ones(final_pi.shape) / (final_pi.shape[0] * final_pi.shape[1])

    # --- Compute supporting distance matrices on demand ------------------
    if js_dist_neighborhood is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to compute js_dist_neighborhood")
        use_gpu, nx = select_backend(use_gpu=use_gpu, gpu_verbose=False)
        js_dist_neighborhood = calculate_neighborhood_dissimilarity(
            sliceA, sliceB, radius, nx=nx, data_type=np.float32, eps=1e-6
        )
        if isinstance(js_dist_neighborhood, torch.Tensor):
            js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()
    if cosine_dist_gene_expr is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to compute cosine_dist_gene_expr")
        cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(sliceA, sliceB, use_rep)
    if cell_type_mismatch is None:
        if sliceA is None or sliceB is None:
            raise ValueError("sliceA and sliceB must be provided to compute cell_type_mismatch")
        cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)

    results: dict = {}

    # --- Cost-based objectives ------------------------------------------
    results["initial_obj_neighbor"] = calculate_neighborhood_similarity(
        js_dist_neighborhood, init_pi, normalize_mass=normalize_mass
    )
    results["final_obj_neighbor"] = calculate_neighborhood_similarity(
        js_dist_neighborhood, final_pi, normalize_mass=normalize_mass
    )
    results["initial_obj_gene"] = calculate_gene_expression_similarity(
        cosine_dist_gene_expr, init_pi, normalize_mass=normalize_mass
    )
    results["final_obj_gene"] = calculate_gene_expression_similarity(
        cosine_dist_gene_expr, final_pi, normalize_mass=normalize_mass
    )
    results["initial_cell_type_match"] = cell_type_matching(cell_type_mismatch, init_pi)
    results["final_cell_type_match"] = cell_type_matching(cell_type_mismatch, final_pi)

    # --- Label-aware metrics --------------------------------------------
    labels_available = (
        sliceA is not None
        and sliceB is not None
        and label_key in sliceA.obs.columns
        and label_key in sliceB.obs.columns
    )
    if labels_available:
        labels_A = sliceA.obs[label_key].astype(str).values
        labels_B = sliceB.obs[label_key].astype(str).values
        results["paa_celltype"] = pairwise_alignment_accuracy(
            labels_A, labels_B, final_pi, weighted=True
        )
        results["ltari_celltype"] = label_transfer_ari(labels_A, labels_B, final_pi)
        # SCS of transferred labels at source positions
        row_mass = final_pi.sum(axis=1)
        active = row_mass > 1e-12
        if active.any():
            try:
                pred = labels_B[np.argmax(final_pi[active], axis=1)]
                results["scs_transferred"] = spatial_coherence_score(
                    np.asarray(sliceA.obsm["spatial"])[active], pred, k=6
                )
            except Exception:
                results["scs_transferred"] = float("nan")
        else:
            results["scs_transferred"] = float("nan")

    # --- Ground-truth-aware metrics (FOSCTTM + landmark) -----------------
    gt_available = (
        sliceA is not None and sliceB is not None
        and ground_truth_pairs is not None
        and np.asarray(ground_truth_pairs).size >= 2
    )
    if gt_available:
        try:
            cA, cB = _post_procrustes_coords(sliceA, sliceB, final_pi)
            gt = np.asarray(ground_truth_pairs)
            idx_A = gt[:, 0].astype(int)
            idx_B = gt[:, 1].astype(int)
            results["foscttm"] = foscttm(cA[idx_A], cB[idx_B])
        except Exception:
            results["foscttm"] = float("nan")
        # landmarks
        try:
            lp = landmark_pairs
            if lp is None:
                rng = np.random.default_rng(0)
                k = min(50, np.asarray(ground_truth_pairs).shape[0])
                sel = rng.choice(np.asarray(ground_truth_pairs).shape[0], size=k, replace=False)
                lp = np.asarray(ground_truth_pairs)[sel]
            results["landmark_mean_err"] = landmark_error(cA, cB, lp, reduction="mean")
            results["landmark_median_err"] = landmark_error(cA, cB, lp, reduction="median")
            results["landmark_rmse"] = landmark_error(cA, cB, lp, reduction="rmse")
        except Exception:
            results["landmark_mean_err"] = float("nan")
            results["landmark_median_err"] = float("nan")
            results["landmark_rmse"] = float("nan")

    # --- Pretty print ----------------------------------------------------
    if verbose:
        rows = []

        # Initial/Final cost objectives
        rows.append((
            " Neighborhood JSD",
            results["initial_obj_neighbor"], results["final_obj_neighbor"],
            "(lower=better)",
        ))
        rows.append((
            " Gene expression cosine",
            results["initial_obj_gene"], results["final_obj_gene"],
            "(lower=better)",
        ))
        rows.append((
            " Cell-type match (mass fraction)",
            results["initial_cell_type_match"], results["final_cell_type_match"],
            "(higher=better)",
        ))

        # Label-aware metrics
        if "paa_celltype" in results:
            rows.append((" PAA (cell-type, weighted)", None, results["paa_celltype"], "(higher=better)"))
        if "ltari_celltype" in results:
            rows.append((" LTARI (cell-type)", None, results["ltari_celltype"], "(higher=better)"))
        if "scs_transferred" in results:
            rows.append((" SCS of transferred labels", None, results["scs_transferred"], "(higher=better)"))

        # FOSCTTM + landmarks
        if "foscttm" in results:
            rows.append((" FOSCTTM (post-Procrustes)", None, results["foscttm"], "(lower=better)"))
        if "landmark_mean_err" in results:
            rows.append((" Landmark mean error", None, results["landmark_mean_err"], "(lower=better)"))
            rows.append((" Landmark median error", None, results["landmark_median_err"], "(lower=better)"))
            rows.append((" Landmark RMSE", None, results["landmark_rmse"], "(lower=better)"))

        _print_table(rows, title="ALIGNMENT QUALITY METRICS")

        # Improvements for the original cost objectives
        def _pct(x_init, x_fin, higher_better):
            if abs(x_init) < 1e-12:
                return float("nan")
            delta = (x_init - x_fin) if not higher_better else (x_fin - x_init)
            return delta / abs(x_init) * 100.0

        print(" Improvements (vs initial)")
        print(f"   Neighborhood JSD          : {_pct(results['initial_obj_neighbor'], results['final_obj_neighbor'], False):>+8.2f}%")
        print(f"   Gene expression cosine    : {_pct(results['initial_obj_gene'], results['final_obj_gene'], False):>+8.2f}%")
        print(f"   Cell-type match           : {_pct(results['initial_cell_type_match'], results['final_cell_type_match'], True):>+8.2f}%")
        print()

    return results
