"""
benchmark/ablation.py
======================
Component ablation study for INCENT.

Answers one question: which parts of INCENT are actually responsible for the
gain, and are those gains stable across datasets? Each named variant removes
or replaces exactly one design component of the pipeline, holding every other
weight and hyperparameter fixed, and is compared to the full pipeline
("Full INCENT") on the *same* paired synthetic self-alignment instances.

Variant -> code mapping
------------------------
Every variant below reaches an existing, already-tested toggle in
src/core.py / src/hierarchical.py / src/clustering.py; nothing is simulated
or approximated at the benchmark-script level.

    Full INCENT                     hierarchical_pairwise_align, package defaults
    w/o cell type                   beta=0 (fine) and delta=0 (coarse), gamma renormalized
    w/o neighborhood                gamma=0 (fine), beta renormalized
    w/o gene expression             beta+gamma renormalized to 1 (fine), delta=1 (coarse)
    w/o multi-scale neighborhood    radius=<mid-multiplier single scale> (see _midscale_radius)
    w/o shared-region detection     skip_shared_region_detection=True
    balanced instead of unbalanced  balanced=True (exact balanced FGW; reg_m ignored)
    w/o hierarchy                   pairwise_align (flat cell-level FGW) instead of
                                     hierarchical_pairwise_align
    w/o geometric admissibility     use_geometric_admissibility=False
    random mesoregions              mesoregion_method="random"
    grid mesoregions                mesoregion_method="grid"

The first seven rows (``MAIN_VARIANTS``) are the recommended main-paper
ablation set; the remaining four (``SUPPLEMENTARY_VARIANTS``) are supplementary.
``ALL_VARIANTS = MAIN_VARIANTS + SUPPLEMENTARY_VARIANTS`` is the script default.

Statistical design
------------------
For every (variant, metric) pair, per-instance deltas are computed against
Full INCENT on the *same* instance index (paired design):

    delta_i = ablation_i - full_i

and summarized as mean +/- s.e.m. across instances, a paired Wilcoxon
signed-rank test against the null delta=0, and a percent-degradation score
relative to Full INCENT's mean (sign convention depends on whether the metric
is lower-is-better, e.g. FOSCTTM, or higher-is-better, e.g. LTA / expr_corr --
see ``LOWER_IS_BETTER`` / ``HIGHER_IS_BETTER``). Benjamini-Hochberg q-values
are reported across the ablation variants for each metric.

Run as a script::

    python -m benchmark.ablation --reference_h5ad slice.h5ad \\
        --section_h5ad section.h5ad --outdir results/ablation

or call :func:`run_ablation` directly with in-memory AnnData objects.
"""

from __future__ import annotations

import json
import os
import concurrent.futures
from dataclasses import dataclass, field
from queue import Queue
from typing import Callable, Optional

import numpy as np
from scipy import stats as scipy_stats

try:
    # Package context (e.g. `import INCENT` from a parent directory, as on Kaggle):
    # `src` only exists as a submodule of the enclosing package, not top-level.
    from ..src.core import (
        hierarchical_pairwise_align,
        pairwise_align,
        estimate_characteristic_spacing,
        DEFAULT_REG_M,
    )
    from ..src.tuning import make_self_alignment_instances, gpu_available, _quiet, DEFAULT_INIT
    from ..src.evaluation import evaluate_alignment
except ImportError:
    # Script context (`python -m benchmark.ablation` from the repo root):
    # `benchmark` and `src` are sibling top-level packages on sys.path.
    from src.core import (
        hierarchical_pairwise_align,
        pairwise_align,
        estimate_characteristic_spacing,
        DEFAULT_REG_M,
    )
    from src.tuning import make_self_alignment_instances, gpu_available, _quiet, DEFAULT_INIT
    from src.evaluation import evaluate_alignment


# Metrics returned by evaluate_alignment(), grouped by which direction is
# "better", so percent-degradation can be signed consistently across metrics.
LOWER_IS_BETTER = {"foscttm", "foscttm_A_to_B", "foscttm_B_to_A"}
HIGHER_IS_BETTER = {"lta", "expr_corr", "neg_foscttm"}
PRIMARY_METRICS = ("foscttm", "lta", "expr_corr")


# ---------------------------------------------------------------------------
# Weight-transform helpers (fine level: alpha/beta/gamma; coarse level: delta)
# ---------------------------------------------------------------------------

def _identity(w: dict) -> dict:
    return dict(w)


def _without_cell_type(w: dict) -> dict:
    """beta -> 0 (fine cell-type mismatch term); delta -> 0 (coarse structural
    term, which is itself a cell-type-composition descriptor -- see
    build_slice_cluster_cache's mu_struct). gamma is renormalized so
    gamma + expression still sum to 1 at the fine level."""
    beta, gamma = float(w["beta"]), float(w["gamma"])
    remaining = 1.0 - beta
    if remaining < 1e-6:
        raise ValueError("w/o cell type: base_weights['beta'] is ~1; nothing left to renormalize.")
    out = dict(w)
    out["beta"] = 0.0
    out["gamma"] = gamma / remaining
    out["delta"] = 0.0
    return out


def _without_neighborhood(w: dict) -> dict:
    """gamma -> 0 (fine neighborhood JSD term); beta renormalized. delta
    (coarse) is left unchanged: the coarse structural term is a cell-type
    descriptor, not the fine-level neighborhood JSD, so it is not implicated
    by this ablation."""
    beta, gamma = float(w["beta"]), float(w["gamma"])
    remaining = 1.0 - gamma
    if remaining < 1e-6:
        raise ValueError("w/o neighborhood: base_weights['gamma'] is ~1; nothing left to renormalize.")
    out = dict(w)
    out["gamma"] = 0.0
    out["beta"] = beta / remaining
    return out


def _without_gene_expression(w: dict) -> dict:
    """expression -> 0 at both levels: beta+gamma renormalized to 1 (fine),
    delta -> 1 so the coarse feature cost drops mu_expr entirely."""
    beta, gamma = float(w["beta"]), float(w["gamma"])
    total = beta + gamma
    if total < 1e-6:
        raise ValueError("w/o gene expression: base_weights['beta']+['gamma'] is ~0; nothing to renormalize onto.")
    out = dict(w)
    out["beta"] = beta / total
    out["gamma"] = gamma / total
    out["delta"] = 1.0
    return out


def _midscale_radius(sliceA, sliceB, multiplier: float = 4.0, spatial_key: str = "spatial") -> float:
    """Single-scale radius standing in for the pipeline's default multiscale
    neighborhood radii (2.5x, 4.0x, 5.0x characteristic spacing): the middle
    multiplier, matching default_radii_from_spacing's own default ladder."""
    s_A = estimate_characteristic_spacing(sliceA, spatial_key=spatial_key)
    s_B = estimate_characteristic_spacing(sliceB, spatial_key=spatial_key)
    return float(multiplier * max(s_A, s_B))


def _single_scale_kwargs_fn(sliceA, sliceB):
    return {"radius": _midscale_radius(sliceA, sliceB)}


# ---------------------------------------------------------------------------
# Variant specification
# ---------------------------------------------------------------------------

@dataclass
class AblationVariant:
    name: str
    description: str
    aligner_kind: str = "hierarchical"  # "hierarchical" | "flat" (pairwise_align, no hierarchy)
    weight_transform: Callable[[dict], dict] = field(default=_identity)
    align_kwargs: dict = field(default_factory=dict)
    # Optional per-instance kwargs (e.g. a radius that depends on the pair's
    # own characteristic spacing); merged in with highest priority.
    per_instance_kwargs_fn: Optional[Callable] = None

    @property
    def weight_keys(self):
        if self.aligner_kind == "flat":
            return ("alpha", "beta", "gamma")
        return ("alpha", "beta", "gamma", "alpha_cluster", "delta")

    @property
    def aligner(self):
        return pairwise_align if self.aligner_kind == "flat" else hierarchical_pairwise_align


MAIN_VARIANTS = [
    AblationVariant("Full INCENT", "Reference: all components enabled."),
    AblationVariant("w/o cell type", "Cell-type term set to 0 (fine beta=0, coarse delta=0).",
                     weight_transform=_without_cell_type),
    AblationVariant("w/o neighborhood", "Neighborhood term set to 0 (fine gamma=0).",
                     weight_transform=_without_neighborhood),
    AblationVariant("w/o multi-scale neighborhood",
                     "Single neighborhood radius instead of the default multiscale ladder.",
                     per_instance_kwargs_fn=_single_scale_kwargs_fn),
    AblationVariant("w/o shared-region detection",
                     "Skip macro-overlap extraction; solve cell-level FGW over the full slices.",
                     align_kwargs={"skip_shared_region_detection": True}),
    AblationVariant("balanced instead of unbalanced",
                     "Exact balanced FGW for the coarse mesoregion coupling (reg_m ignored).",
                     align_kwargs={"balanced": True}),
    AblationVariant("w/o hierarchy", "Direct cell-level FGW solve; no mesoregion stage at all.",
                     aligner_kind="flat"),
]

SUPPLEMENTARY_VARIANTS = [
    AblationVariant("w/o gene expression",
                     "Gene-expression term set to 0 at both levels (fine and coarse).",
                     weight_transform=_without_gene_expression),
    AblationVariant("w/o geometric admissibility",
                     "Disable the rigid-consistency admissibility gate in macro-overlap expansion.",
                     align_kwargs={"use_geometric_admissibility": False}),
    AblationVariant("random mesoregions",
                     "Poisson-disk random seeding instead of farthest-point-seeded CVT.",
                     align_kwargs={"mesoregion_method": "random", "mesoregion_seed": 0}),
    AblationVariant("grid mesoregions",
                     "Original PCA-canonical grid seeding instead of farthest-point-seeded CVT.",
                     align_kwargs={"mesoregion_method": "grid"}),
]

ALL_VARIANTS = MAIN_VARIANTS + SUPPLEMENTARY_VARIANTS
VARIANTS_BY_NAME = {v.name: v for v in ALL_VARIANTS}


# ---------------------------------------------------------------------------
# Alignment execution
# ---------------------------------------------------------------------------

def _is_stable_pi(pi: np.ndarray, min_total_mass: float = 0.05) -> bool:
    """True if the transport plan is numerically usable (finite, non-collapsed)."""
    if pi is None:
        return False
    if not np.isfinite(pi).all():
        return False
    if float(pi.sum()) < min_total_mass:
        return False
    return True


def _align_one(
    variant: AblationVariant, sim, ref, base_weights: dict, base_align_kwargs: dict,
    aligner_override: Optional[Callable] = None,
):
    """
    Run one variant on one (sim, ref) instance; return pi or None on failure.

    ``aligner_override``, when given, replaces ``variant.aligner`` for every
    variant regardless of ``aligner_kind`` -- the injection point unit tests
    use to substitute a cheap stub for the real (expensive) OT solvers, mirroring
    the ``aligner=`` parameter of :func:`benchmark.reg_m_sensitivity._eval_reg_m`.
    """
    weights_full = variant.weight_transform(base_weights)
    call_weights = {k: weights_full[k] for k in variant.weight_keys}

    call_kwargs = dict(base_align_kwargs)
    call_kwargs.update(variant.align_kwargs)
    if variant.per_instance_kwargs_fn is not None:
        call_kwargs.update(variant.per_instance_kwargs_fn(sim, ref))

    if variant.aligner_kind == "flat" and aligner_override is None:
        # pairwise_align has no notion of mesoregions / reg_m / hierarchy-only kwargs.
        allowed = {"use_gpu", "verbose", "gpu_verbose", "numItermax", "radius"}
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in allowed}

    aligner = aligner_override if aligner_override is not None else variant.aligner
    try:
        with _quiet(True):
            pi = aligner(sim, ref, **call_weights, **call_kwargs)
        return np.asarray(pi, dtype=np.float64)
    except Exception as e:
        print(f"[ablation] '{variant.name}' failed on one instance: {type(e).__name__}: {e}")
        return None


def _eval_variant(
    variant: AblationVariant, instances: list, base_weights: dict, base_align_kwargs: dict,
    label_key: str, min_total_mass: float,
    aligner_override: Optional[Callable] = None,
) -> dict:
    """Align all instances for one variant; return per-instance metrics + n_ok/n_unstable."""
    per_instance = []
    n_unstable = 0
    for sim, ref in instances:
        pi = _align_one(variant, sim, ref, base_weights, base_align_kwargs, aligner_override=aligner_override)
        if not _is_stable_pi(pi, min_total_mass=min_total_mass):
            n_unstable += 1
            per_instance.append(None)
            continue
        m = evaluate_alignment(pi, sim, ref, sim_axis=0, label_key=label_key, include_expression=True)
        per_instance.append(m)

    n_ok = sum(1 for m in per_instance if m is not None)
    return {
        "variant": variant.name,
        "description": variant.description,
        "n_ok": n_ok,
        "n_unstable": n_unstable,
        "per_instance": per_instance,
    }


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

def _paired_metric_arrays(full_per_instance: list, variant_per_instance: list, metric_key: str):
    """Extract matched (full, variant) values for one metric, dropping any
    instance where either side is missing/None/non-finite."""
    full_vals, var_vals = [], []
    for m_full, m_var in zip(full_per_instance, variant_per_instance):
        if m_full is None or m_var is None:
            continue
        vf, vv = m_full.get(metric_key), m_var.get(metric_key)
        if vf is None or vv is None:
            continue
        vf, vv = float(vf), float(vv)
        if not (np.isfinite(vf) and np.isfinite(vv)):
            continue
        full_vals.append(vf)
        var_vals.append(vv)
    return np.array(full_vals, dtype=np.float64), np.array(var_vals, dtype=np.float64)


def _paired_stats(full_vals: np.ndarray, var_vals: np.ndarray, metric_key: str) -> dict:
    """Paired mean/s.e.m./Wilcoxon/percent-degradation for one (metric, variant) pair."""
    n = int(full_vals.size)
    out = {
        "n": n,
        "mean_full": float(np.mean(full_vals)) if n else None,
        "mean_variant": float(np.mean(var_vals)) if n else None,
        "mean_delta": None,
        "sem_delta": None,
        "wilcoxon_stat": None,
        "wilcoxon_p": None,
        "pct_degradation": None,
    }
    if n == 0:
        return out

    deltas = var_vals - full_vals
    out["mean_delta"] = float(np.mean(deltas))
    out["sem_delta"] = float(np.std(deltas, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    if n >= 2 and np.any(deltas != 0):
        try:
            stat, p = scipy_stats.wilcoxon(deltas)
            out["wilcoxon_stat"], out["wilcoxon_p"] = float(stat), float(p)
        except ValueError:
            pass  # e.g. all-zero deltas after floating-point cancellation

    mean_full = out["mean_full"]
    if mean_full is not None and abs(mean_full) > 1e-12:
        if metric_key in LOWER_IS_BETTER:
            out["pct_degradation"] = float((out["mean_variant"] - mean_full) / abs(mean_full) * 100.0)
        elif metric_key in HIGHER_IS_BETTER:
            out["pct_degradation"] = float((mean_full - out["mean_variant"]) / abs(mean_full) * 100.0)
    return out


def _bh_fdr(p_values) -> np.ndarray:
    """Benjamini-Hochberg q-values; NaN/None p-values pass through as NaN."""
    p = np.array([np.nan if v is None else float(v) for v in p_values], dtype=np.float64)
    q = np.full_like(p, np.nan)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    idx = np.where(finite)[0]
    order = idx[np.argsort(p[idx])]
    m = order.size
    ranked_p = p[order]
    q_sorted = ranked_p * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)
    q[order] = q_sorted
    return q


def _compute_all_paired_stats(variant_rows: list, metrics=PRIMARY_METRICS) -> dict:
    """Paired stats for every (non-Full) variant x metric, with BH-FDR q-values
    computed across variants within each metric."""
    full_row = next(r for r in variant_rows if r["variant"] == "Full INCENT")
    stats_by_metric = {m: {} for m in metrics}

    for metric in metrics:
        names, p_vals = [], []
        for row in variant_rows:
            if row["variant"] == "Full INCENT":
                continue
            full_vals, var_vals = _paired_metric_arrays(full_row["per_instance"], row["per_instance"], metric)
            s = _paired_stats(full_vals, var_vals, metric)
            stats_by_metric[metric][row["variant"]] = s
            names.append(row["variant"])
            p_vals.append(s["wilcoxon_p"])

        q_vals = _bh_fdr(p_vals)
        for name, q in zip(names, q_vals):
            stats_by_metric[metric][name]["wilcoxon_q"] = None if np.isnan(q) else float(q)

    return stats_by_metric


# ---------------------------------------------------------------------------
# Parallel execution (mirrors kappa_sensitivity.py / reg_m_sensitivity.py)
# ---------------------------------------------------------------------------

def _make_device_pool_call(fn: Callable, device_ids: list) -> Callable:
    try:
        import torch
    except ImportError:
        return fn
    if not torch.cuda.is_available():
        return fn
    pool: Queue = Queue()
    for did in device_ids:
        pool.put(did)

    def _wrapped(*args, **kwargs):
        did = pool.get()
        try:
            with torch.cuda.device(did):
                return fn(*args, **kwargs)
        finally:
            pool.put(did)

    return _wrapped


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_ablation(
    section=None,
    reference=None,
    *,
    crops=None,
    real_pairs=None,
    base_weights: Optional[dict] = None,
    variants: Optional[list] = None,
    n_instances: int = 5,
    perturb_kwargs: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    label_key: str = "cell_type_annot",
    spatial_key: str = "spatial",
    min_total_mass: float = 0.05,
    n_jobs: int = 1,
    device_ids=None,
    seed: int = 0,
    outdir: Optional[str] = "results/ablation",
    aligner_override: Optional[Callable] = None,
) -> dict:
    """
    Run the full component ablation and report paired statistics for every
    variant relative to "Full INCENT".

    Instance source (priority ``real_pairs`` > ``crops`` > ``section``+``reference``),
    matching the convention of :func:`benchmark.reg_m_sensitivity.run_reg_m_sensitivity`.
    ``real_pairs`` mode aligns genuine slice pairs as-is (no synthetic perturbation,
    no FOSCTTM ground truth -- only LTA/expr_corr are meaningful there); the other
    two modes go through :func:`simulate_adjacent_slice` for exact FOSCTTM ground truth.

    Parameters
    ----------
    section, reference, crops, real_pairs:
        See :func:`src.tuning.make_self_alignment_instances`; ``real_pairs`` is an
        explicit list of ``(sliceA, sliceB)`` pairs aligned without perturbation.
    base_weights:
        The five INCENT weights (``alpha``, ``beta``, ``gamma``, ``alpha_cluster``,
        ``delta``) that "Full INCENT" uses; ablations transform this dict (e.g.
        zeroing and renormalizing). Defaults to :data:`src.tuning.DEFAULT_INIT`.
    variants:
        List of :class:`AblationVariant` to run. Defaults to :data:`ALL_VARIANTS`
        (``MAIN_VARIANTS + SUPPLEMENTARY_VARIANTS``); must include a variant named
        "Full INCENT" (the paired-statistics reference).
    n_instances:
        Synthetic pairs generated once and reused identically across every variant
        (paired design), when using ``section``+``reference`` mode.
    align_kwargs:
        Extra kwargs forwarded to every hierarchical variant's aligner (e.g.
        ``use_gpu``). Must not set any per-variant ablation kwarg
        (``balanced``, ``skip_shared_region_detection``,
        ``use_geometric_admissibility``, ``mesoregion_method``) -- those are
        controlled by each :class:`AblationVariant`.
    n_jobs, device_ids:
        Parallelize across variants (not instances), mirroring
        :func:`benchmark.reg_m_sensitivity.run_reg_m_sensitivity`.
    outdir:
        If given, writes ``ablation.json`` and ``ablation.png``.
    aligner_override:
        Testing hook: replaces every variant's aligner (regardless of
        ``aligner_kind``) with this callable. Not used in normal operation.

    Returns
    -------
    dict with keys ``rows`` (per-variant per-instance metrics), ``paired_stats``
    (per-metric per-variant paired comparison against Full INCENT), ``config``.
    """
    if real_pairs is None and crops is None and (section is None or reference is None):
        raise ValueError("Provide `real_pairs`, `crops`, or both `section` and `reference`.")

    base_weights = dict(base_weights or DEFAULT_INIT)
    variants = list(variants or ALL_VARIANTS)
    if not any(v.name == "Full INCENT" for v in variants):
        raise ValueError("`variants` must include a variant named 'Full INCENT' "
                          "(the paired-statistics reference point).")

    forbidden = {"balanced", "skip_shared_region_detection",
                 "use_geometric_admissibility", "mesoregion_method", "mesoregion_seed"}
    if forbidden & set(align_kwargs or {}):
        raise ValueError(
            f"align_kwargs must not set {forbidden & set(align_kwargs)} -- "
            "these are controlled per-variant by AblationVariant.align_kwargs."
        )

    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)

    if real_pairs is not None:
        instances = list(real_pairs)
        print(f"[ablation] {len(instances)} real pair(s) supplied "
              f"(no synthetic perturbation, no ground truth -- FOSCTTM will be N/A), "
              f"running {len(variants)} variant(s).")
    else:
        perturb_kwargs = dict(perturb_kwargs or {})
        instances = make_self_alignment_instances(
            section=section, reference=reference, crops=crops,
            n_instances=n_instances, perturb_kwargs=perturb_kwargs, seed=seed,
        )
        print(f"[ablation] {len(instances)} instance(s) generated, "
              f"running {len(variants)} variant(s): {[v.name for v in variants]}")

    if align_kwargs["use_gpu"] and device_ids is None:
        device_ids = [0]

    def _job(variant):
        row = _eval_variant(
            variant, instances, base_weights, align_kwargs, label_key, min_total_mass,
            aligner_override=aligner_override,
        )
        print(f"  {row['variant']:<32s} n_ok={row['n_ok']}/{len(instances)}  "
              f"({row['description']})")
        return row

    if n_jobs == 1:
        rows = [_job(v) for v in variants]
    else:
        _job_parallel = _job
        if align_kwargs["use_gpu"] and device_ids and len(device_ids) > 1:
            _job_parallel = _make_device_pool_call(_job, device_ids)
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_job_parallel, variants))
        order = {v.name: i for i, v in enumerate(variants)}
        rows.sort(key=lambda r: order[r["variant"]])

    paired_stats = _compute_all_paired_stats(rows, metrics=PRIMARY_METRICS)

    results = {
        "rows": rows,
        "paired_stats": paired_stats,
        "config": {
            "variant_names": [v.name for v in variants],
            "base_weights": base_weights,
            "n_instances": len(instances),
            "mode": "real_pairs" if real_pairs is not None
                    else "crops" if crops is not None else "section+reference",
            "perturb_kwargs": perturb_kwargs if real_pairs is None else None,
            "min_total_mass": min_total_mass,
            "seed": seed,
            "n_jobs": n_jobs,
            "device_ids": device_ids,
        },
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        json_path = os.path.join(outdir, "ablation.json")
        with open(json_path, "w") as f:
            json.dump(_json_safe(results), f, indent=2)
        print(f"[ablation] results written to {json_path}")
        try:
            _make_figure(results, outdir)
        except Exception as e:
            print(f"[ablation] figure skipped: {e}")

    return results


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to plain Python for json.dump."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# Figure: percent degradation vs Full INCENT, one panel per primary metric
# ---------------------------------------------------------------------------

def _make_figure(results: dict, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paired_stats = results["paired_stats"]
    variant_names = [n for n in results["config"]["variant_names"] if n != "Full INCENT"]
    if not variant_names:
        return

    metrics = [m for m in PRIMARY_METRICS if any(
        paired_stats.get(m, {}).get(v, {}).get("pct_degradation") is not None for v in variant_names
    )]
    if not metrics:
        return

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5), squeeze=False)
    axes = axes[0]

    for ax, metric in zip(axes, metrics):
        vals, errs, sig = [], [], []
        for v in variant_names:
            s = paired_stats[metric].get(v, {})
            pct = s.get("pct_degradation")
            vals.append(pct if pct is not None else 0.0)
            mean_full = s.get("mean_full")
            sem = s.get("sem_delta")
            if mean_full and sem is not None and abs(mean_full) > 1e-12:
                errs.append(abs(sem) / abs(mean_full) * 100.0)
            else:
                errs.append(0.0)
            q = s.get("wilcoxon_q")
            sig.append("*" if (q is not None and q < 0.05) else "")

        colors = ["tab:red" if v > 0 else "tab:blue" for v in vals]
        y_pos = np.arange(len(variant_names))
        ax.barh(y_pos, vals, xerr=errs, color=colors, alpha=0.85, capsize=3)
        for y, v, s in zip(y_pos, vals, sig):
            if s:
                offset = (max(errs) if errs else 1.0) * 0.15 + 0.5
                ax.text(v + np.sign(v) * offset if v != 0 else offset, y, s,
                        va="center", ha="left" if v >= 0 else "right", fontsize=12, color="black")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(variant_names, fontsize=9)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2, label="Full INCENT")
        ax.set_xlabel("% degradation vs Full INCENT\n(positive = worse)", fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        ax.invert_yaxis()

    fig.suptitle(
        f"Ablation: % degradation vs Full INCENT "
        f"(mean +/- s.e.m. over {results['config']['n_instances']} paired instances; "
        f"* = BH q<0.05)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    fig_path = os.path.join(outdir, "ablation.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[ablation] figure written to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import scanpy as sc

    ap = argparse.ArgumentParser(description="Component ablation study for INCENT.")
    ap.add_argument("--reference_h5ad",
                    help="Full parent slice (.h5ad). Required unless --crops_h5ad or "
                         "--real_pairs_h5ad is used.")
    ap.add_argument("--section_h5ad",
                    help="Cropped section (.h5ad); parent-frame coords whose "
                         "obs_names subset the reference. Required unless --crops_h5ad or "
                         "--real_pairs_h5ad is used.")
    ap.add_argument("--crops_h5ad", nargs="+", metavar="SECTION REF",
                    help="Explicit crop pairs as alternating section/reference .h5ad paths.")
    ap.add_argument("--real_pairs_h5ad", nargs="+", metavar="SLICE_A SLICE_B",
                    help="Genuine real slice pairs to align as-is (no synthetic "
                         "perturbation, no ground truth -- FOSCTTM will be N/A).")
    ap.add_argument("--outdir", default="results/ablation")
    ap.add_argument("--main_only", action="store_true",
                    help="Run only the 7 main-paper variants (skip the 4 supplementary ones).")
    ap.add_argument("--variants", nargs="+", metavar="NAME", default=None,
                    help="Explicit subset of variant names to run (must include 'Full INCENT'). "
                         f"Available: {[v.name for v in ALL_VARIANTS]}")
    ap.add_argument("--n_instances", type=int, default=5,
                    help="Synthetic pairs per variant in section+reference mode (default: 5). "
                         "Ignored when --crops_h5ad or --real_pairs_h5ad is used.")
    ap.add_argument("--min_total_mass", type=float, default=0.05)
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel threads across variants (default: 1).")
    ap.add_argument("--device_ids", type=int, nargs="+", default=None)
    ap.add_argument("--use_gpu", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.use_gpu == "auto":
        use_gpu = gpu_available()
    else:
        use_gpu = args.use_gpu == "true"

    if args.variants:
        selected = [VARIANTS_BY_NAME[n] for n in args.variants]
    elif args.main_only:
        selected = MAIN_VARIANTS
    else:
        selected = ALL_VARIANTS

    real_pairs = None
    crops = None
    section = reference = None
    if args.real_pairs_h5ad is not None:
        paths = args.real_pairs_h5ad
        if len(paths) % 2 != 0:
            ap.error("--real_pairs_h5ad requires an even number of paths (sliceA sliceB pairs).")
        real_pairs = [(sc.read_h5ad(paths[i]), sc.read_h5ad(paths[i + 1]))
                      for i in range(0, len(paths), 2)]
    elif args.crops_h5ad is not None:
        paths = args.crops_h5ad
        if len(paths) % 2 != 0:
            ap.error("--crops_h5ad requires an even number of paths (section ref pairs).")
        crops = [(sc.read_h5ad(paths[i]), sc.read_h5ad(paths[i + 1]))
                 for i in range(0, len(paths), 2)]
    else:
        if not args.section_h5ad or not args.reference_h5ad:
            ap.error("Provide --real_pairs_h5ad, --crops_h5ad, or both "
                     "--section_h5ad and --reference_h5ad.")
        section = sc.read_h5ad(args.section_h5ad)
        reference = sc.read_h5ad(args.reference_h5ad)

    res = run_ablation(
        section, reference,
        crops=crops, real_pairs=real_pairs,
        variants=selected,
        n_instances=args.n_instances,
        align_kwargs={"use_gpu": use_gpu},
        min_total_mass=args.min_total_mass,
        n_jobs=args.n_jobs,
        device_ids=args.device_ids,
        seed=args.seed,
        outdir=args.outdir,
    )
    print(f"\nResults written to {args.outdir}")
