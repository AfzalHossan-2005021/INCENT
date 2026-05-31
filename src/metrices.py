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
