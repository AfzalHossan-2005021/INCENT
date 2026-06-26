"""
metrices.py — Alignment quality metrics for INCENT benchmarking.

Metric taxonomy
---------------
PRIMARY  (objective-independent; use in main benchmark table):
    lta            Label Transfer Accuracy — hard argmax prediction accuracy.
    foscttm        Fraction Of Samples Closer Than True Match (coupling-space).
                   Auto-detected from sliceB when produced by adjacent_slice.py.
    gpr            Geometric Preservation Rate (barycentric projection + kNN).
    gpr_per_k      GPR broken down per neighbourhood size k.

SCALABILITY (report alongside primary metrics):
    benchmark_method   Wall-clock time + peak RSS memory for any callable.

SUPPLEMENTARY (soft-plan weighted; direct components of INCENT's objective):
    These carry systematic bias when comparing methods with different objectives.
    Report in supplementary material only — never as the primary evaluation.
    final_obj_neighbor      Weighted JSD neighbourhood dissimilarity.
    final_obj_gene          Weighted cosine gene-expression dissimilarity.
    final_cell_type_match   Soft cell-type correspondence fraction.

Ground-truth correspondences (FOSCTTM)
---------------------------------------
adjacent_slice.simulate_adjacent_slice stores:
    sliceB.uns["self_alignment_test"]["adjacent_simulation"]
               ["dropout_kept_positions"]    shape (n_B,), int64
For each cell j in sliceB, dropout_kept_positions[j] is the integer row index
of its true match in sliceA.  calculate_performance_metrics auto-extracts this;
pass gt_src_indices explicitly to override.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

from .core import (
    calculate_cell_type_mismatch,
    calculate_gene_expression_cosine_distance,
    calculate_neighborhood_dissimilarity,
)
from .utils import select_backend


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY METRIC 1 — Label Transfer Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def label_transfer_accuracy(
    pi: np.ndarray,
    labels_A: np.ndarray,
    labels_B: np.ndarray,
    mass_threshold: float = 0.0,
) -> Tuple[float, Dict]:
    """
    Label Transfer Accuracy (LTA).

    For each source cell i, the predicted target is j* = argmax_j Pi[i, j].
    LTA = fraction of source cells for which label_A[i] == label_B[j*].

    This metric is independent of INCENT's optimization objective.  INCENT
    minimizes a *soft* weighted mismatch (beta term in M1); LTA evaluates hard
    argmax accuracy, which no compared method optimizes directly.

    For unbalanced transport plans, rows with negligible total mass correspond
    to cells the solver placed outside the overlap region.  Set mass_threshold
    to exclude them from the accuracy calculation; 0.0 (default) evaluates all.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
        Transport plan.  Need not sum to 1; unbalanced plans are supported.
    labels_A : array-like, shape (n_A,)
        Cell-type annotation for source slice (sliceA.obs['cell_type_annot']).
    labels_B : array-like, shape (n_B,)
        Cell-type annotation for target slice (sliceB.obs['cell_type_annot']).
    mass_threshold : float in [0, 1)
        Skip source cells whose row mass is below (mass_threshold * mean_row_mass).
        Zero-mass rows are *always* excluded regardless of this value (a cell with
        no transported mass has no meaningful argmax prediction).  Default 0.0
        evaluates all cells that received any mass.

    Returns
    -------
    lta : float
        Accuracy in [0, 1].  Higher is better.
    detail : dict
        n_evaluated, n_correct, n_skipped_low_mass, per_cell_correct (bool array).
    """
    pi = np.asarray(pi, dtype=np.float64)
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)

    row_masses = pi.sum(axis=1)
    active = row_masses > 0.0                              # always drop zero-mass rows
    if mass_threshold > 0.0:
        mean_mass = row_masses.mean()
        active &= row_masses >= mass_threshold * mean_mass

    n_active = int(active.sum())
    if n_active == 0:
        return 0.0, {
            'n_evaluated': 0, 'n_correct': 0,
            'n_skipped_low_mass': pi.shape[0],
            'per_cell_correct': np.array([], dtype=bool),
        }

    predicted_targets = np.argmax(pi[active], axis=1)           # (n_active,)
    correct = labels_A[active] == labels_B[predicted_targets]   # (n_active,) bool
    lta = float(correct.mean())

    return lta, {
        'n_evaluated': n_active,
        'n_correct': int(correct.sum()),
        'n_skipped_low_mass': int((~active).sum()),
        'per_cell_correct': correct,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY METRIC 2 — FOSCTTM
# ─────────────────────────────────────────────────────────────────────────────

def foscttm(
    pi: np.ndarray,
    gt_src_indices: np.ndarray,
    chunk_size: int = 512,
) -> Dict[str, float]:
    """
    FOSCTTM — Fraction Of Samples Closer Than True Match (coupling-space).

    Standard definition (Demetci et al., 2022) adapted for OT coupling matrices.
    For each ground-truth pair (i*, j), the score is the fraction of cells in
    the opposing set that receive *more* coupling weight than the true match.
    Lower is better; 0.0 = true match is always ranked first.

    Two directions are computed and averaged for a symmetric estimate:
        B→A : for each j in B, rank of its true A-match in column j of Pi.
        A→B : for each i* in A, rank of its true B-match in row i* of Pi.

    Requires ground-truth correspondences produced by adjacent_slice.py:
        gt_src_indices[j] = i*  ←→  sliceB cell j matches sliceA cell i*.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
    gt_src_indices : array-like, shape (n_matched,)
        Integer row indices into sliceA.  For all-surviving-cell simulations
        n_matched == n_B.  Auto-extracted from sliceB.uns when available.
    chunk_size : int
        Column/row chunk size for memory-bounded computation.  At chunk_size=512
        peak working memory is O(n_A * 512 * 8) bytes ≈ 40 MB for n_A = 10,000.

    Returns
    -------
    dict with keys:
        'foscttm'       float — symmetric mean (primary reported value).
        'foscttm_B_to_A' float — B→A direction.
        'foscttm_A_to_B' float — A→B direction.
    """
    pi = np.asarray(pi, dtype=np.float64)
    gt_src_indices = np.asarray(gt_src_indices, dtype=np.int64)
    n_A, n_B = pi.shape
    n_matched = len(gt_src_indices)

    # True-match coupling weights: Pi[gt_src_indices[j], j] for j = 0..n_matched-1
    j_range = np.arange(n_matched, dtype=np.int64)
    true_weights = pi[gt_src_indices, j_range]              # (n_matched,)

    # ── Direction B→A ──────────────────────────────────────────────────────
    # For each j, score = [strictly_better + 0.5 * tied_excl_self] / (n_A - 1)
    # Midpoint tie-breaking with self-exclusion ensures:
    #   perfect plan  → 0.0  (only the true match has nonzero weight)
    #   uniform plan  → 0.5  (all n_A - 1 other cells are tied)
    scores_B = np.empty(n_matched, dtype=np.float64)
    for start in range(0, n_matched, chunk_size):
        end = min(start + chunk_size, n_matched)
        col_block = pi[:, start:end]                         # (n_A, chunk)
        tw = true_weights[start:end][np.newaxis, :]          # (1, chunk)
        strictly = (col_block > tw).sum(axis=0).astype(np.float64)
        tied = (col_block == tw).sum(axis=0).astype(np.float64) - 1.0  # subtract self
        scores_B[start:end] = (strictly + 0.5 * tied) / (n_A - 1)
    foscttm_B_to_A = float(scores_B.mean())

    # ── Direction A→B ──────────────────────────────────────────────────────
    # Same midpoint rule; denominator is n_B - 1 (all B cells except true match).
    scores_A = np.empty(n_matched, dtype=np.float64)
    for start in range(0, n_matched, chunk_size):
        end = min(start + chunk_size, n_matched)
        rows = gt_src_indices[start:end]
        row_block = pi[rows, :]                              # (chunk, n_B)
        tw = true_weights[start:end][:, np.newaxis]          # (chunk, 1)
        strictly = (row_block > tw).sum(axis=1).astype(np.float64)
        tied = (row_block == tw).sum(axis=1).astype(np.float64) - 1.0  # subtract self
        scores_A[start:end] = (strictly + 0.5 * tied) / (n_B - 1)
    foscttm_A_to_B = float(scores_A.mean())

    return {
        'foscttm': 0.5 * (foscttm_B_to_A + foscttm_A_to_B),
        'foscttm_B_to_A': foscttm_B_to_A,
        'foscttm_A_to_B': foscttm_A_to_B,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY METRIC 3 — Geometric Preservation Rate
# ─────────────────────────────────────────────────────────────────────────────

def geometric_preservation_rate(
    pi: np.ndarray,
    coords_A: np.ndarray,
    coords_B: np.ndarray,
    k_values: Sequence[int] = (5, 10, 15),
) -> Dict:
    """
    Geometric Preservation Rate (GPR).

    Measures whether local spatial neighbourhoods in sliceA are preserved after
    alignment, evaluated in the barycentric projection of Pi into sliceB space.

    For each source cell i:
        1. Project: p_i = Σ_j Pi[i,j] * coords_B[j] / Σ_j Pi[i,j]
        2. Find k nearest spatial neighbours of i in sliceA coords: N_k^src(i)
        3. Find k nearest projected neighbours of p_i in {p_j}: N_k^proj(i)
        4. GPR@k(i) = |N_k^src(i) ∩ N_k^proj(i)| / k

    GPR = mean over all (i, k) pairs.  Higher is better; 1.0 = perfect.

    Barycentric projection is used (rather than argmax) because it is smooth,
    handles unbalanced plans gracefully, and penalises diffuse mappings that
    scatter mass over distant target cells.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
    coords_A : ndarray, shape (n_A, 2)
    coords_B : ndarray, shape (n_B, 2)
    k_values : sequence of int
        Neighbourhood sizes.  Default (5, 10, 15) spans local to meso-scale.

    Returns
    -------
    dict with keys:
        'gpr'       float — mean across all k (primary reported value).
        'gpr_per_k' dict  — {k: float} individual scores.
    """
    pi = np.asarray(pi, dtype=np.float64)
    coords_A = np.asarray(coords_A, dtype=np.float64)[:, :2]
    coords_B = np.asarray(coords_B, dtype=np.float64)[:, :2]
    n_A = pi.shape[0]

    # Barycentric projection (row-normalised)
    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)     # guard zero-mass rows
    proj = (pi @ coords_B) / row_sums                       # (n_A, 2)

    k_max = max(k_values) + 1                                # +1 to exclude self

    nn_src = NearestNeighbors(n_neighbors=k_max, algorithm='ball_tree', n_jobs=-1)
    nn_src.fit(coords_A)
    src_neighbors = nn_src.kneighbors(coords_A, return_distance=False)[:, 1:]   # (n_A, k_max-1)

    nn_proj = NearestNeighbors(n_neighbors=k_max, algorithm='ball_tree', n_jobs=-1)
    nn_proj.fit(proj)
    proj_neighbors = nn_proj.kneighbors(proj, return_distance=False)[:, 1:]     # (n_A, k_max-1)

    gpr_per_k: Dict[int, float] = {}
    for k in k_values:
        src_k = src_neighbors[:, :k]    # (n_A, k)
        proj_k = proj_neighbors[:, :k]  # (n_A, k)
        scores = np.array([
            len(np.intersect1d(src_k[i], proj_k[i])) / k
            for i in range(n_A)
        ], dtype=np.float64)
        gpr_per_k[k] = float(scores.mean())

    gpr = float(np.mean(list(gpr_per_k.values())))
    return {'gpr': gpr, 'gpr_per_k': gpr_per_k}


# ─────────────────────────────────────────────────────────────────────────────
# SCALABILITY UTILITY
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_method(fn: Callable, *args, **kwargs) -> Tuple:
    """
    Measure wall-clock time and peak heap memory for any callable.

    Uses time.perf_counter() for sub-millisecond timing and tracemalloc for
    Python heap peak.  Does not capture GPU memory; instrument CUDA separately
    with torch.cuda.max_memory_allocated() when benchmarking GPU-accelerated methods.

    Usage
    -----
    pi, perf = benchmark_method(incent.pairwise_align, sliceA, sliceB, alpha=0.5)

    Parameters
    ----------
    fn : callable
    *args, **kwargs : forwarded to fn.

    Returns
    -------
    result : return value of fn(*args, **kwargs).
    perf   : dict — 'wall_time_s' (float), 'peak_memory_mb' (float).
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {
        'wall_time_s': wall,
        'peak_memory_mb': peak_bytes / (1024.0 ** 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — soft-plan metrics (components of INCENT's objective)
# ─────────────────────────────────────────────────────────────────────────────
# All three functions below operate on the full Pi matrix via element-wise
# multiplication with a pre-computed distance/mismatch matrix.  Because INCENT
# directly minimizes these quantities, using them as evaluation metrics
# introduces systematic bias favouring INCENT over baselines that optimize
# different objectives.  Retain for internal diagnostics and supplementary
# material; do NOT use as primary evaluation metrics.

def calculate_neighborhood_dissimilarity_cost(
    js_dist_neighborhood: np.ndarray, pi: np.ndarray
) -> float:
    """Weighted JSD neighbourhood dissimilarity (SUPPLEMENTARY ONLY)."""
    return float(np.sum(js_dist_neighborhood * pi))


def calculate_gene_expression_dissimilarity(
    cosine_dist_gene_expr: np.ndarray, pi: np.ndarray
) -> float:
    """Weighted cosine gene-expression dissimilarity (SUPPLEMENTARY ONLY)."""
    return float(np.sum(cosine_dist_gene_expr * pi))


def cell_type_matching(cell_type_mismatch: np.ndarray, pi_mat: np.ndarray) -> float:
    """
    Soft cell-type correspondence fraction (SUPPLEMENTARY ONLY).

    Returns the fraction of transported mass landing on same-type target cells.
    """
    total_mass = float(np.sum(pi_mat))
    if total_mass <= 0.0:
        return 0.0
    return float(np.sum((1.0 - cell_type_mismatch) * pi_mat)) / total_mass


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _relative_improvement(initial: float, final: float) -> float:
    """Signed relative improvement (positive = better for dissimilarity metrics)."""
    if initial == 0.0:
        return 0.0 if final == 0.0 else float('nan')
    return (initial - final) / initial * 100.0


def _extract_gt_indices(sliceB) -> Optional[np.ndarray]:
    """
    Extract ground-truth source indices from an adjacent_slice simulation.
    Returns None if sliceB was not produced by simulate_adjacent_slice.
    """
    try:
        return np.asarray(
            sliceB.uns["self_alignment_test"]["adjacent_simulation"][
                "dropout_kept_positions"],
            dtype=np.int64,
        )
    except (KeyError, TypeError):
        return None


def _extract_coords(slc, spatial_key: str = 'spatial') -> np.ndarray:
    return np.asarray(slc.obsm[spatial_key], dtype=np.float64)[:, :2]


def _print_metrics(results: Dict, W: int = 82) -> None:
    """Formatted console report separating primary from supplementary metrics."""

    def _row(label, value, note=''):
        if value is None:
            return f"  {label:<44} {'N/A':<10} {note}"
        return f"  {label:<44} {value:<10.4f} {note}"

    sep = "─" * W
    print()
    print("=" * W)
    print(f"{'ALIGNMENT QUALITY METRICS':^{W}}")
    print("=" * W)

    # ── Primary ──────────────────────────────────────────────────────────────
    print(f"\n  {'PRIMARY METRICS  (objective-independent)':}")
    print(f"  {sep}")
    print(_row("Label Transfer Accuracy (LTA)", results.get('lta'), "↑"))
    if results.get('foscttm') is not None:
        print(_row("FOSCTTM (symmetric mean)", results.get('foscttm'), "↓"))
        print(_row("  FOSCTTM  A→B", results.get('foscttm_A_to_B'), "↓"))
        print(_row("  FOSCTTM  B→A", results.get('foscttm_B_to_A'), "↓"))
    print(_row("Geometric Preservation Rate (GPR)", results.get('gpr'), "↑"))
    if results.get('gpr_per_k') is not None:
        for k, v in sorted(results['gpr_per_k'].items()):
            print(_row(f"  GPR@{k}", v, "↑"))

    # ── LTA detail ───────────────────────────────────────────────────────────
    d = results.get('lta_detail')
    if d:
        print(f"\n  {'LTA detail':}")
        print(f"  {sep}")
        print(f"  {'cells evaluated':<44} {d['n_evaluated']}")
        print(f"  {'cells correct':<44} {d['n_correct']}")
        print(f"  {'cells skipped (low mass)':<44} {d['n_skipped_low_mass']}")

    # ── Supplementary ────────────────────────────────────────────────────────
    print(f"\n  {'SUPPLEMENTARY METRICS  (in INCENT objective — biased; supplementary only)':}")
    print(f"  {sep}")
    for label, k_init, k_fin, direction, lower_is_better in [
        ("Neighbourhood Dissimilarity (JSD)",
         'initial_obj_neighbor', 'final_obj_neighbor', "↓", True),
        ("Gene Expression Dissimilarity (Cos)",
         'initial_obj_gene', 'final_obj_gene', "↓", True),
        ("Cell-type Correspondence, soft (%)",
         'initial_cell_type_match', 'final_cell_type_match', "↑", False),
    ]:
        v0 = results.get(k_init)
        v1 = results.get(k_fin)
        if v0 is None or v1 is None:
            print(f"  {label:<44} N/A")
            continue
        scale = 100.0 if 'cell_type' in k_init else 1.0
        # For lower-is-better metrics: positive % = improvement (value decreased).
        # For higher-is-better metrics: positive % = improvement (value increased).
        # Both branches normalise against the initial value v0 for consistency.
        imp = _relative_improvement(v0, v1) if lower_is_better else -_relative_improvement(v0, v1)
        print(f"  {label:<44} {v0*scale:.4f} → {v1*scale:.4f}  ({imp:+.1f}%)  {direction}")

    print("=" * W)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def calculate_performance_metrics(
    final_pi: np.ndarray,
    init_pi: Optional[np.ndarray] = None,
    js_dist_neighborhood: Optional[np.ndarray] = None,
    cosine_dist_gene_expr: Optional[np.ndarray] = None,
    cell_type_mismatch: Optional[np.ndarray] = None,
    sliceA=None,
    sliceB=None,
    use_rep: Optional[str] = None,
    radius: float = 100.0,
    use_gpu: bool = True,
    # Primary metric parameters
    gpr_k_values: Sequence[int] = (5, 10, 15),
    mass_threshold: float = 0.0,
    gt_src_indices: Optional[np.ndarray] = None,
    spatial_key: str = 'spatial',
    foscttm_chunk_size: int = 512,
) -> Dict:
    """
    Compute all alignment quality metrics for INCENT benchmarking.

    PRIMARY metrics (objective-independent, for the main benchmark table):
        'lta'            Label Transfer Accuracy (higher = better).
        'foscttm'        Mean FOSCTTM (lower = better).  Requires ground-truth
                         correspondences; auto-detected from sliceB.uns when
                         sliceB was produced by adjacent_slice.simulate_adjacent_slice.
        'foscttm_A_to_B' A→B directional FOSCTTM.
        'foscttm_B_to_A' B→A directional FOSCTTM.
        'gpr'            Geometric Preservation Rate, mean over gpr_k_values (higher = better).
        'gpr_per_k'      dict {k: GPR@k}.
        'lta_detail'     dict with per-cell breakdown.

    SUPPLEMENTARY metrics (soft-plan weighted; in INCENT's objective):
        'initial_obj_neighbor'     Weighted JSD before alignment.
        'final_obj_neighbor'       Weighted JSD after alignment.
        'initial_obj_gene'         Weighted cosine before alignment.
        'final_obj_gene'           Weighted cosine after alignment.
        'initial_cell_type_match'  Soft cell-type fraction before alignment.
        'final_cell_type_match'    Soft cell-type fraction after alignment.

    Parameters
    ----------
    final_pi : ndarray, shape (n_A, n_B)
        Final transport plan.
    init_pi : ndarray or None
        Baseline plan for supplementary comparison.  Defaults to uniform.
    js_dist_neighborhood : ndarray or None
        Pre-computed JSD matrix.  Computed from sliceA/B if None.
    cosine_dist_gene_expr : ndarray or None
        Pre-computed cosine distance matrix.  Computed from sliceA/B if None.
    cell_type_mismatch : ndarray or None
        Pre-computed binary mismatch matrix.  Computed from sliceA/B if None.
    sliceA, sliceB : AnnData or None
        Required if any distance matrix must be computed.
    use_rep : str or None
        obsm key for gene-expression representation.
    radius : float
        Neighbourhood radius for JSD calculation.
    use_gpu : bool
    gpr_k_values : sequence of int
        Neighbourhood sizes for GPR.  Default (5, 10, 15).
    mass_threshold : float
        LTA: skip source cells with row mass below (threshold * mean_row_mass).
    gt_src_indices : array-like or None
        Ground-truth source indices for FOSCTTM.  Auto-detected from sliceB.uns
        when None and sliceB was produced by simulate_adjacent_slice.
    spatial_key : str
        obsm key for spatial coordinates.
    foscttm_chunk_size : int
        Memory chunk size for FOSCTTM.  Default 512.

    Returns
    -------
    dict : All metric values as described above.
    """
    final_pi = np.asarray(final_pi, dtype=np.float64)

    if init_pi is None:
        init_pi = np.ones(final_pi.shape, dtype=np.float64) / float(final_pi.size)
    else:
        init_pi = np.asarray(init_pi, dtype=np.float64)

    # ── Supplementary distance matrices ───────────────────────────────────────
    if js_dist_neighborhood is None:
        if sliceA is None or sliceB is None:
            raise ValueError(
                "sliceA and sliceB must be provided to compute js_dist_neighborhood.")
        _, nx = select_backend(use_gpu=use_gpu, gpu_verbose=False)
        js_dist_neighborhood = calculate_neighborhood_dissimilarity(
            sliceA, sliceB, radius, nx=nx, data_type=np.float32, eps=1e-6)
        if isinstance(js_dist_neighborhood, torch.Tensor):
            js_dist_neighborhood = js_dist_neighborhood.detach().cpu().numpy()
    js_dist_neighborhood = np.asarray(js_dist_neighborhood, dtype=np.float64)

    if cosine_dist_gene_expr is None:
        if sliceA is None or sliceB is None:
            raise ValueError(
                "sliceA and sliceB must be provided to compute cosine_dist_gene_expr.")
        cosine_dist_gene_expr = calculate_gene_expression_cosine_distance(
            sliceA, sliceB, use_rep)
    cosine_dist_gene_expr = np.asarray(cosine_dist_gene_expr, dtype=np.float64)

    if cell_type_mismatch is None:
        if sliceA is None or sliceB is None:
            raise ValueError(
                "sliceA and sliceB must be provided to compute cell_type_mismatch.")
        cell_type_mismatch = calculate_cell_type_mismatch(sliceA, sliceB)
    cell_type_mismatch = np.asarray(cell_type_mismatch, dtype=np.float64)

    results: Dict = {}

    # ── PRIMARY: LTA ──────────────────────────────────────────────────────────
    if sliceA is not None and sliceB is not None:
        lta_val, lta_detail = label_transfer_accuracy(
            final_pi,
            sliceA.obs['cell_type_annot'].values,
            sliceB.obs['cell_type_annot'].values,
            mass_threshold=mass_threshold,
        )
        results['lta'] = lta_val
        results['lta_detail'] = lta_detail
    else:
        results['lta'] = None
        results['lta_detail'] = None

    # ── PRIMARY: FOSCTTM ──────────────────────────────────────────────────────
    if gt_src_indices is None and sliceB is not None:
        gt_src_indices = _extract_gt_indices(sliceB)

    if gt_src_indices is not None:
        gt_src_indices = np.asarray(gt_src_indices, dtype=np.int64)
        results.update(
            foscttm(final_pi, gt_src_indices, chunk_size=foscttm_chunk_size))
    else:
        results['foscttm'] = None
        results['foscttm_A_to_B'] = None
        results['foscttm_B_to_A'] = None

    # ── PRIMARY: GPR ──────────────────────────────────────────────────────────
    if sliceA is not None and sliceB is not None:
        results.update(geometric_preservation_rate(
            final_pi,
            _extract_coords(sliceA, spatial_key),
            _extract_coords(sliceB, spatial_key),
            k_values=gpr_k_values,
        ))
    else:
        results['gpr'] = None
        results['gpr_per_k'] = None

    # ── SUPPLEMENTARY: soft-plan metrics ──────────────────────────────────────
    results['initial_obj_neighbor'] = calculate_neighborhood_dissimilarity_cost(
        js_dist_neighborhood, init_pi)
    results['final_obj_neighbor'] = calculate_neighborhood_dissimilarity_cost(
        js_dist_neighborhood, final_pi)
    results['initial_obj_gene'] = calculate_gene_expression_dissimilarity(
        cosine_dist_gene_expr, init_pi)
    results['final_obj_gene'] = calculate_gene_expression_dissimilarity(
        cosine_dist_gene_expr, final_pi)
    results['initial_cell_type_match'] = cell_type_matching(cell_type_mismatch, init_pi)
    results['final_cell_type_match'] = cell_type_matching(cell_type_mismatch, final_pi)

    _print_metrics(results)
    return results