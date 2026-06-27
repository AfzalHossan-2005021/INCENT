"""
evaluation.py
=============
Alignment quality metric battery for INCENT.

Metric taxonomy
---------------
PRIMARY  (objective-independent; use in the main benchmark table):
    lta            Label Transfer Accuracy — hard argmax prediction accuracy.
    foscttm        Fraction Of Samples Closer Than True Match (Demetci et al., 2022).
                   Auto-extracted from the simulated slice's .uns when available.
    gpr            Geometric Preservation Rate (barycentric projection + kNN).
    gpr_per_k      GPR broken down per neighbourhood size k.

EXPRESSION PROXY  (label-free; soft expression-consistency check):
    expr_corr      Mean per-gene Pearson correlation, predicted vs measured.

SUPPLEMENTARY  (direct components of INCENT's objective; biased when comparing methods):
    final_obj_neighbor      Weighted JSD neighbourhood dissimilarity.
    final_obj_gene          Weighted cosine gene-expression dissimilarity.
    final_cell_type_match   Soft cell-type correspondence fraction.

SCALABILITY:
    benchmark_method        Wall-clock time + peak RSS for any callable.

Orientation convention
----------------------
pi has shape (n_sliceA, n_sliceB) as returned by hierarchical_pairwise_align(A, B).
In evaluate_alignment, sim_axis controls which slice is the simulated one
(0 = sliceA is sim, 1 = sliceB is sim); pi is internally oriented to (n_sim, n_parent)
before computing LTA, GPR, and expression correlation.  FOSCTTM receives the transposed
plan (n_parent, n_sim) as the metric requires.

In calculate_performance_metrics the convention is sliceA = reference/parent (rows),
sliceB = simulated (cols); FOSCTTM ground-truth indices are auto-extracted from sliceB.uns.

Ground-truth correspondences (FOSCTTM)
---------------------------------------
simulate_adjacent_slice stores:
    sim.uns["self_alignment_test"]["adjacent_simulation"]["dropout_kept_positions"]
               shape (n_sim,), int64
For each sim cell j, dropout_kept_positions[j] is the row index of its true match in
the reference slice.  evaluate_alignment / calculate_performance_metrics extract this
automatically; pass gt_src_indices explicitly to override.
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

from .core import (
    calculate_cell_type_mismatch,
    calculate_gene_expression_cosine_distance,
    calculate_neighborhood_dissimilarity,
)
from .utils import select_backend


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)


def _extract_gt_indices(slc) -> Optional[np.ndarray]:
    """Extract ground-truth source indices from a simulate_adjacent_slice output."""
    try:
        return np.asarray(
            slc.uns["self_alignment_test"]["adjacent_simulation"]["dropout_kept_positions"],
            dtype=np.int64,
        )
    except (KeyError, TypeError):
        return None


def _extract_coords(slc, spatial_key: str = "spatial") -> np.ndarray:
    return np.asarray(slc.obsm[spatial_key], dtype=np.float64)[:, :2]


def _relative_improvement(initial: float, final: float) -> float:
    if initial == 0.0:
        return 0.0 if final == 0.0 else float("nan")
    return (initial - final) / initial * 100.0


def _print_metrics(results: Dict, W: int = 82) -> None:
    """Formatted console report separating primary from supplementary metrics."""

    def _row(label, value, note=""):
        if value is None:
            return f"  {label:<44} {'N/A':<10} {note}"
        return f"  {label:<44} {value:<10.4f} {note}"

    sep = "─" * W
    print()
    print("=" * W)
    print(f"{'ALIGNMENT QUALITY METRICS':^{W}}")
    print("=" * W)

    print(f"\n  {'PRIMARY METRICS  (objective-independent)':}")
    print(f"  {sep}")
    print(_row("Label Transfer Accuracy (LTA)", results.get("lta"), "↑"))
    if results.get("foscttm") is not None:
        print(_row("FOSCTTM (symmetric mean)", results.get("foscttm"), "↓"))
        print(_row("  FOSCTTM  A→B", results.get("foscttm_A_to_B"), "↓"))
        print(_row("  FOSCTTM  B→A", results.get("foscttm_B_to_A"), "↓"))
    print(_row("Geometric Preservation Rate (GPR)", results.get("gpr"), "↑"))
    if results.get("gpr_per_k") is not None:
        for k, v in sorted(results["gpr_per_k"].items()):
            print(_row(f"  GPR@{k}", v, "↑"))
    if results.get("expr_corr") is not None:
        print(_row("Expression Transfer Correlation", results.get("expr_corr"), "↑"))

    d = results.get("lta_detail")
    if d:
        print(f"\n  {'LTA detail':}")
        print(f"  {sep}")
        print(f"  {'cells evaluated':<44} {d['n_evaluated']}")
        print(f"  {'cells correct':<44} {d['n_correct']}")
        print(f"  {'cells skipped (low mass)':<44} {d['n_skipped_low_mass']}")

    print(f"\n  {'SUPPLEMENTARY METRICS  (in INCENT objective — biased; supplementary only)':}")
    print(f"  {sep}")
    for label, k_init, k_fin, direction, lower_is_better in [
        ("Neighbourhood Dissimilarity (JSD)",
         "initial_obj_neighbor", "final_obj_neighbor", "↓", True),
        ("Gene Expression Dissimilarity (Cos)",
         "initial_obj_gene", "final_obj_gene", "↓", True),
        ("Cell-type Correspondence, soft (%)",
         "initial_cell_type_match", "final_cell_type_match", "↑", False),
    ]:
        v0 = results.get(k_init)
        v1 = results.get(k_fin)
        if v0 is None or v1 is None:
            print(f"  {label:<44} N/A")
            continue
        scale = 100.0 if "cell_type" in k_init else 1.0
        imp = _relative_improvement(v0, v1) if lower_is_better else -_relative_improvement(v0, v1)
        print(f"  {label:<44} {v0*scale:.4f} → {v1*scale:.4f}  ({imp:+.1f}%)  {direction}")

    print("=" * W)
    print()


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

    For each source cell i the predicted target is j* = argmax_j Pi[i, j].
    LTA = fraction of source cells for which label_A[i] == label_B[j*].

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
    labels_A, labels_B : array-like
    mass_threshold : float in [0, 1)
        Skip source cells whose row mass is below (mass_threshold * mean_row_mass).
        Zero-mass rows are always excluded regardless of this value.

    Returns
    -------
    lta : float — accuracy in [0, 1], higher is better.
    detail : dict — n_evaluated, n_correct, n_skipped_low_mass, per_cell_correct.
    """
    pi = np.asarray(pi, dtype=np.float64)
    labels_A = np.asarray(labels_A)
    labels_B = np.asarray(labels_B)

    row_masses = pi.sum(axis=1)
    active = row_masses > 0.0
    if mass_threshold > 0.0:
        mean_mass = row_masses.mean()
        active &= row_masses >= mass_threshold * mean_mass

    n_active = int(active.sum())
    if n_active == 0:
        return 0.0, {
            "n_evaluated": 0,
            "n_correct": 0,
            "n_skipped_low_mass": pi.shape[0],
            "per_cell_correct": np.array([], dtype=bool),
        }

    predicted_targets = np.argmax(pi[active], axis=1)
    correct = labels_A[active] == labels_B[predicted_targets]
    return float(correct.mean()), {
        "n_evaluated": n_active,
        "n_correct": int(correct.sum()),
        "n_skipped_low_mass": int((~active).sum()),
        "per_cell_correct": correct,
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
    FOSCTTM — Fraction Of Samples Closer Than True Match (Demetci et al., 2022).

    Coupling-space ranking metric: for each ground-truth pair (i*, j) the score is
    the fraction of cells in the opposing set that receive strictly more coupling weight
    than the true match.  Lower is better; 0.0 = perfect; ~0.5 = random.

    Two symmetric directions are computed and averaged:
        B→A : for each sim cell j, rank of its true ref-match in column j of Pi.
        A→B : for each ref cell i*, rank of its true sim-match in row i* of Pi.

    Tie-breaking is midpoint so that a perfect plan → 0.0 and a uniform plan → 0.5.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
        Transport plan; rows = reference (parent), columns = simulated.
    gt_src_indices : array-like, shape (n_matched,)
        gt_src_indices[j] = i* — the reference-slice row index of the true match for
        simulated cell j.  Auto-extracted from sim.uns by evaluate_alignment when
        produced by simulate_adjacent_slice.
    chunk_size : int
        Memory chunk for the inner loops.  Default 512 ≈ 40 MB peak at 10k cells.

    Returns
    -------
    dict : 'foscttm' (symmetric mean, lower is better),
           'foscttm_B_to_A', 'foscttm_A_to_B'.
    """
    pi = np.asarray(pi, dtype=np.float64)
    gt_src_indices = np.asarray(gt_src_indices, dtype=np.int64)
    n_A, n_B = pi.shape
    n_matched = len(gt_src_indices)

    j_range = np.arange(n_matched, dtype=np.int64)
    true_weights = pi[gt_src_indices, j_range]          # (n_matched,)

    # ── Direction B→A ────────────────────────────────────────────────────────
    scores_B = np.empty(n_matched, dtype=np.float64)
    for start in range(0, n_matched, chunk_size):
        end = min(start + chunk_size, n_matched)
        col_block = pi[:, start:end]                    # (n_A, chunk)
        tw = true_weights[start:end][np.newaxis, :]    # (1, chunk)
        strictly = (col_block > tw).sum(axis=0).astype(np.float64)
        tied = (col_block == tw).sum(axis=0).astype(np.float64) - 1.0
        scores_B[start:end] = (strictly + 0.5 * tied) / (n_A - 1)
    foscttm_B_to_A = float(scores_B.mean())

    # ── Direction A→B ────────────────────────────────────────────────────────
    scores_A = np.empty(n_matched, dtype=np.float64)
    for start in range(0, n_matched, chunk_size):
        end = min(start + chunk_size, n_matched)
        rows = gt_src_indices[start:end]
        row_block = pi[rows, :]                         # (chunk, n_B)
        tw = true_weights[start:end][:, np.newaxis]    # (chunk, 1)
        strictly = (row_block > tw).sum(axis=1).astype(np.float64)
        tied = (row_block == tw).sum(axis=1).astype(np.float64) - 1.0
        scores_A[start:end] = (strictly + 0.5 * tied) / (n_B - 1)
    foscttm_A_to_B = float(scores_A.mean())

    return {
        "foscttm": 0.5 * (foscttm_B_to_A + foscttm_A_to_B),
        "foscttm_B_to_A": foscttm_B_to_A,
        "foscttm_A_to_B": foscttm_A_to_B,
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

    For each source cell i:
      1. Barycentric projection: p_i = Σ_j Pi[i,j] · coords_B[j] / Σ_j Pi[i,j]
      2. kNN of i in source space: N_k^src(i).
      3. kNN of p_i among {p_j}: N_k^proj(i).
      4. GPR@k(i) = |N_k^src(i) ∩ N_k^proj(i)| / k.

    GPR = mean over all cells and k values.  Higher is better; 1.0 = perfect geometry
    preservation.  Penalises diffuse mappings that scatter local neighbourhoods.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
        Transport plan; rows = source (sim), columns = target (reference).
    coords_A : ndarray, shape (n_A, 2)  — source spatial coordinates.
    coords_B : ndarray, shape (n_B, 2)  — target spatial coordinates.
    k_values : sequence of int
        Neighbourhood sizes.  Default (5, 10, 15) spans local to meso-scale.

    Returns
    -------
    dict : 'gpr' (mean over k_values, higher is better), 'gpr_per_k' {k: float}.
    """
    pi = np.asarray(pi, dtype=np.float64)
    coords_A = np.asarray(coords_A, dtype=np.float64)[:, :2]
    coords_B = np.asarray(coords_B, dtype=np.float64)[:, :2]
    n_A = pi.shape[0]

    row_sums = pi.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    proj = (pi @ coords_B) / row_sums                  # (n_A, 2)

    k_max = max(k_values) + 1
    nn_src = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", n_jobs=-1)
    nn_src.fit(coords_A)
    src_neighbors = nn_src.kneighbors(coords_A, return_distance=False)[:, 1:]  # exclude self

    nn_proj = NearestNeighbors(n_neighbors=k_max, algorithm="ball_tree", n_jobs=-1)
    nn_proj.fit(proj)
    proj_neighbors = nn_proj.kneighbors(proj, return_distance=False)[:, 1:]

    gpr_per_k: Dict[int, float] = {}
    for k in k_values:
        src_k = src_neighbors[:, :k]
        proj_k = proj_neighbors[:, :k]
        scores = np.array([
            len(np.intersect1d(src_k[i], proj_k[i])) / k
            for i in range(n_A)
        ], dtype=np.float64)
        gpr_per_k[k] = float(scores.mean())

    return {"gpr": float(np.mean(list(gpr_per_k.values()))), "gpr_per_k": gpr_per_k}


# ─────────────────────────────────────────────────────────────────────────────
# EXPRESSION PROXY
# ─────────────────────────────────────────────────────────────────────────────

def expression_transfer_corr(
    pi: np.ndarray,
    expr_source: np.ndarray,
    expr_target: np.ndarray,
    *,
    max_genes: int = 2000,
) -> dict:
    """
    Mean per-gene Pearson correlation between barycentric-predicted and measured
    target expression (label-free expression-consistency proxy).

    For each mapped target cell j:
        pred[j] = Σ_i Pi[i,j] · expr_source[i] / Σ_i Pi[i,j]
    Then correlate pred vs expr_target across mapped cells, per gene.

    Genes with near-zero variance in either side are skipped; up to max_genes
    most-variable genes are used for speed.

    Returns dict: 'expr_corr', 'n_genes', 'n_scored'.
    """
    pi = np.asarray(pi, dtype=np.float64)
    A = _dense(expr_source)
    B = _dense(expr_target)
    col_mass = pi.sum(axis=0)
    mapped = col_mass > 0
    if mapped.sum() < 3:
        return {"expr_corr": float("nan"), "n_genes": 0, "n_scored": int(mapped.sum())}

    pred = np.zeros_like(B)
    pred[mapped] = (pi[:, mapped].T @ A) / col_mass[mapped, None]

    Bm, Pm = B[mapped], pred[mapped]
    if B.shape[1] > max_genes:
        keep = np.argsort(Bm.var(axis=0))[::-1][:max_genes]
        Bm, Pm = Bm[:, keep], Pm[:, keep]

    corrs = []
    for g in range(Bm.shape[1]):
        b, p = Bm[:, g], Pm[:, g]
        if b.std() > 1e-12 and p.std() > 1e-12:
            corrs.append(float(np.corrcoef(b, p)[0, 1]))
    return {
        "expr_corr": float(np.nanmean(corrs)) if corrs else float("nan"),
        "n_genes": int(len(corrs)),
        "n_scored": int(mapped.sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SUPPLEMENTARY — components of INCENT's objective
# ─────────────────────────────────────────────────────────────────────────────
# These three functions operate on the full Pi matrix via element-wise
# multiplication with a pre-computed distance/mismatch matrix.  Because INCENT
# directly minimises these quantities, using them as evaluation metrics
# introduces systematic bias favouring INCENT over baselines.  Retain for
# internal diagnostics and supplementary material only.

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
# SCALABILITY
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_method(fn: Callable, *args, **kwargs) -> Tuple:
    """
    Measure wall-clock time and peak Python heap memory for any callable.

    Usage: result, perf = benchmark_method(incent.pairwise_align, sliceA, sliceB)
    Returns (result, {'wall_time_s': float, 'peak_memory_mb': float}).
    Does not capture GPU memory; use torch.cuda.max_memory_allocated() separately.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    wall = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, {"wall_time_s": wall, "peak_memory_mb": peak_bytes / (1024.0 ** 2)}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT 1 — convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_alignment(
    pi: np.ndarray,
    sliceA,
    sliceB,
    *,
    spatial_key: str = "spatial",
    label_key: str = "cell_type_annot",
    sim_axis: int = 0,
    gpr_k_values: Sequence[int] = (5, 10, 15),
    mass_threshold: float = 0.0,
    gt_src_indices: Optional[np.ndarray] = None,
    include_expression: bool = True,
    foscttm_chunk_size: int = 512,
) -> dict:
    """
    Run the full primary metric battery on one aligned (sliceA, sliceB) pair.

    pi has shape (n_A, n_B) as returned by hierarchical_pairwise_align(sliceA, sliceB).
    sim_axis indicates which slice is the simulated one (0 = sliceA, 1 = sliceB); pi
    is internally oriented to rows=sim, cols=reference before LTA, GPR, and expression.
    FOSCTTM receives the transposed plan (n_ref, n_sim).

    FOSCTTM ground-truth indices are auto-extracted from the simulated slice's .uns
    when produced by simulate_adjacent_slice; pass gt_src_indices to override.

    Parameters
    ----------
    pi : ndarray, shape (n_A, n_B)
    sliceA, sliceB : AnnData
    spatial_key : str   — obsm key for spatial coordinates.
    label_key : str     — obs column for cell-type labels (also tries label_key + '_clean').
    sim_axis : int      — 0 if sliceA is the simulated slice, 1 if sliceB is.
    gpr_k_values : sequence of int — neighbourhood sizes for GPR.
    mass_threshold : float — LTA: skip source cells below this fraction of mean mass.
    gt_src_indices : array or None — override auto-extracted FOSCTTM ground truth.
    include_expression : bool — whether to compute expression_transfer_corr.
    foscttm_chunk_size : int — memory chunk for FOSCTTM inner loops.

    Returns
    -------
    dict with keys:
        lta, lta_detail                             Label Transfer Accuracy.
        foscttm, foscttm_A_to_B, foscttm_B_to_A   FOSCTTM (None if no gt available).
        neg_foscttm                                 1 − foscttm; higher is better; used as
                                                    the weight-selection objective.
        gpr, gpr_per_k                              Geometric Preservation Rate.
        expr_corr                                   Expression correlation (if requested).
    """
    pi = np.asarray(pi, dtype=np.float64)

    # Orient so that rows = sim, cols = reference
    pi_oriented = pi.T if sim_axis == 1 else pi
    sim_slice = sliceA if sim_axis == 0 else sliceB
    par_slice = sliceB if sim_axis == 0 else sliceA

    out: dict = {}

    # ── LTA ──────────────────────────────────────────────────────────────────
    clean_key = label_key + "_clean"
    col_sim = (clean_key if clean_key in sim_slice.obs.columns
               else label_key if label_key in sim_slice.obs.columns else None)
    col_par = (clean_key if clean_key in par_slice.obs.columns
               else label_key if label_key in par_slice.obs.columns else None)
    if col_sim is not None and col_par is not None:
        lta_val, lta_detail = label_transfer_accuracy(
            pi_oriented,
            sim_slice.obs[col_sim].astype(str).to_numpy(),
            par_slice.obs[col_par].astype(str).to_numpy(),
            mass_threshold=mass_threshold,
        )
        out["lta"] = lta_val
        out["lta_detail"] = lta_detail
    else:
        out["lta"] = None
        out["lta_detail"] = None

    # ── FOSCTTM ──────────────────────────────────────────────────────────────
    # Expects pi of shape (n_ref, n_sim); gt_src_indices[j] = ref-row for sim cell j.
    if gt_src_indices is None:
        gt_arr = _extract_gt_indices(sim_slice)
    else:
        gt_arr = np.asarray(gt_src_indices, dtype=np.int64)

    if gt_arr is not None:
        f = foscttm(pi_oriented.T, gt_arr, chunk_size=foscttm_chunk_size)
        out.update(f)
        # neg_foscttm = 1 - foscttm: converts FOSCTTM to a higher-is-better score
        # suitable for use as the weight-selection objective (select_alignment_weights).
        out["neg_foscttm"] = float(1.0 - f["foscttm"])
    else:
        out["foscttm"] = None
        out["foscttm_A_to_B"] = None
        out["foscttm_B_to_A"] = None
        out["neg_foscttm"] = None

    # ── GPR ──────────────────────────────────────────────────────────────────
    coords_sim = _extract_coords(sim_slice, spatial_key)
    coords_par = _extract_coords(par_slice, spatial_key)
    gpr_result = geometric_preservation_rate(
        pi_oriented, coords_sim, coords_par, k_values=gpr_k_values)
    out.update(gpr_result)

    # ── Expression transfer correlation ───────────────────────────────────────
    if include_expression:
        ec = expression_transfer_corr(pi_oriented, sim_slice.X, par_slice.X)
        out["expr_corr"] = ec["expr_corr"]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT 2 — full battery with supplementary metrics
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
    gpr_k_values: Sequence[int] = (5, 10, 15),
    mass_threshold: float = 0.0,
    gt_src_indices: Optional[np.ndarray] = None,
    spatial_key: str = "spatial",
    foscttm_chunk_size: int = 512,
) -> Dict:
    """
    Compute all alignment quality metrics for INCENT benchmarking.

    Convention: pi has shape (n_A, n_B) with sliceA = reference/parent (rows)
    and sliceB = simulated (cols).  FOSCTTM ground-truth indices are auto-extracted
    from sliceB.uns when sliceB was produced by simulate_adjacent_slice.

    PRIMARY metrics (objective-independent, main benchmark table):
        'lta', 'lta_detail'
        'foscttm', 'foscttm_A_to_B', 'foscttm_B_to_A'  (None if no gt available)
        'gpr', 'gpr_per_k'

    SUPPLEMENTARY metrics (biased; components of INCENT's objective):
        'initial_obj_neighbor', 'final_obj_neighbor'
        'initial_obj_gene', 'final_obj_gene'
        'initial_cell_type_match', 'final_cell_type_match'

    Parameters
    ----------
    final_pi : ndarray, shape (n_A, n_B)
    init_pi : ndarray or None — baseline plan for supplementary before/after comparison.
    js_dist_neighborhood : ndarray or None — computed from sliceA/B if None.
    cosine_dist_gene_expr : ndarray or None — computed from sliceA/B if None.
    cell_type_mismatch : ndarray or None — computed from sliceA/B if None.
    sliceA, sliceB : AnnData or None — required when any distance matrix must be computed.
    use_rep : str or None — obsm key for gene-expression features.
    radius : float — neighbourhood radius for JSD computation.
    use_gpu : bool
    gpr_k_values : sequence of int
    mass_threshold : float — LTA mass threshold.
    gt_src_indices : array or None — override auto-extracted FOSCTTM ground truth.
    spatial_key : str
    foscttm_chunk_size : int
    """
    final_pi = np.asarray(final_pi, dtype=np.float64)
    init_pi = (np.ones(final_pi.shape, dtype=np.float64) / float(final_pi.size)
               if init_pi is None else np.asarray(init_pi, dtype=np.float64))

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
    if (sliceA is not None and sliceB is not None
            and "cell_type_annot" in sliceA.obs.columns
            and "cell_type_annot" in sliceB.obs.columns):
        lta_val, lta_detail = label_transfer_accuracy(
            final_pi,
            sliceA.obs["cell_type_annot"].values,
            sliceB.obs["cell_type_annot"].values,
            mass_threshold=mass_threshold,
        )
        results["lta"] = lta_val
        results["lta_detail"] = lta_detail
    else:
        results["lta"] = None
        results["lta_detail"] = None

    # ── PRIMARY: FOSCTTM ──────────────────────────────────────────────────────
    if gt_src_indices is None and sliceB is not None:
        gt_src_indices = _extract_gt_indices(sliceB)

    if gt_src_indices is not None:
        results.update(foscttm(
            final_pi,
            np.asarray(gt_src_indices, dtype=np.int64),
            chunk_size=foscttm_chunk_size,
        ))
    else:
        results["foscttm"] = None
        results["foscttm_A_to_B"] = None
        results["foscttm_B_to_A"] = None

    # ── PRIMARY: GPR ──────────────────────────────────────────────────────────
    if sliceA is not None and sliceB is not None:
        results.update(geometric_preservation_rate(
            final_pi,
            _extract_coords(sliceA, spatial_key),
            _extract_coords(sliceB, spatial_key),
            k_values=gpr_k_values,
        ))
    else:
        results["gpr"] = None
        results["gpr_per_k"] = None

    # ── SUPPLEMENTARY ─────────────────────────────────────────────────────────
    results["initial_obj_neighbor"] = calculate_neighborhood_dissimilarity_cost(
        js_dist_neighborhood, init_pi)
    results["final_obj_neighbor"] = calculate_neighborhood_dissimilarity_cost(
        js_dist_neighborhood, final_pi)
    results["initial_obj_gene"] = calculate_gene_expression_dissimilarity(
        cosine_dist_gene_expr, init_pi)
    results["final_obj_gene"] = calculate_gene_expression_dissimilarity(
        cosine_dist_gene_expr, final_pi)
    results["initial_cell_type_match"] = cell_type_matching(cell_type_mismatch, init_pi)
    results["final_cell_type_match"] = cell_type_matching(cell_type_mismatch, final_pi)

    _print_metrics(results)
    return results
