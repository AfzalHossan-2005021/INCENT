"""
benchmark/run_weight_benchmark.py
=================================
Synthetic, ground-truth-anchored benchmark that (a) selects robust default
alignment weights, (b) reports their generalization on a held-out instance
split, (c) measures robustness to every nuisance severity axis, and (d)
validates the label-free deployment selector by correlating the label-free
metric (GPR / spatial coherence) with the ground-truth metrics (LTA, FOSCTTM)
across a weight grid.

Everything is built on :func:`perturb.simulate_adjacent_slice` (exact ground
truth) + the metric battery in :mod:`evaluation` + the staged selector in
:mod:`tuning`. The aligner is pluggable (``aligner`` argument) so competing
methods (PASTE/PASTE2/GPSA/STalign) can be dropped in later on identical
instances and metrics; the registry below is the (deferred) hook.

Run as a script::

    python -m benchmark.run_weight_benchmark --input_h5ad slice.h5ad --outdir results/bench

or call :func:`run_weight_benchmark` directly with an in-memory AnnData.
"""

from __future__ import annotations

import json
import os
from typing import Callable, Optional

import numpy as np

from src.core import hierarchical_pairwise_align
from src.tuning import (
    select_alignment_weights,
    make_self_alignment_instances,
    gpu_available,
    DEFAULT_INIT,
    _quiet,
)
from src.evaluation import evaluate_alignment


# Deferred baseline hook: register competing methods as
#   name -> callable(sliceA, sliceB, **weights, **align_kwargs) -> pi (nA x nB).
# A method that ignores the INCENT weights should accept/discard them.
BASELINE_REGISTRY: dict[str, Callable] = {}


def _default_aligner(sliceA, sliceB, **kwargs):
    return hierarchical_pairwise_align(sliceA, sliceB, **kwargs)


def _align(aligner, sliceA, sliceB, weights, align_kwargs, quiet):
    try:
        with _quiet(quiet):
            pi = aligner(sliceA, sliceB, **weights, **align_kwargs)
        return np.asarray(pi, dtype=np.float64)
    except Exception:
        return None


def grid_sweep_full(
    instances,
    weight_list,
    *,
    aligner=_default_aligner,
    align_kwargs,
    label_key="cell_type_annot",
    spatial_key="spatial",
    quiet=True,
):
    """Evaluate a list of weight dicts on the instances; return one row per weight
    with the full metric battery averaged over instances (NaN-safe).

    Non-scalar metric values (lta_detail, gpr_per_k) are silently skipped so the
    row dict contains only float-valued entries suitable for JSON serialisation.
    """
    rows = []
    for w in weight_list:
        mets = []
        for sim, ref in instances:
            pi = _align(aligner, sim, ref, w, align_kwargs, quiet)
            if pi is None:
                continue
            mets.append(evaluate_alignment(pi, sim, ref, sim_axis=0,
                                           label_key=label_key, spatial_key=spatial_key))
        row = {**w, "n_ok": len(mets)}
        if mets:
            keys = set().union(*[m.keys() for m in mets])
            for k in keys:
                vals = []
                for m in mets:
                    v = m.get(k, np.nan)
                    # skip non-scalar values (lta_detail is a dict, gpr_per_k is a dict)
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        vals.append(float(v))
                    elif v is None:
                        vals.append(np.nan)
                if vals:
                    row[k] = float(np.nanmean(vals))
        rows.append(row)
    return rows


def robustness_curves(
    section,
    reference,
    best_weights,
    severity_axes,
    *,
    crops=None,
    baseline_perturb,
    aligner=_default_aligner,
    n_instances=2,
    align_kwargs,
    label_key="cell_type_annot",
    spatial_key="spatial",
    seed=0,
):
    """For each severity axis, sweep its values (others at baseline), regenerate
    instances from the supplied crop(s), align at ``best_weights``, and record mean
    GPR and LTA across instances."""
    curves = {}
    for ai, (axis, values) in enumerate(severity_axes.items()):
        pts = []
        for v in values:
            pk = dict(baseline_perturb)
            pk[axis] = v
            inst = make_self_alignment_instances(
                section=section, reference=reference, crops=crops,
                n_instances=n_instances, perturb_kwargs=pk, seed=seed + 100 * ai)
            gpr_scores, lta_scores = [], []
            for sim, ref in inst:
                pi = _align(aligner, sim, ref, best_weights, align_kwargs, True)
                if pi is None:
                    gpr_scores.append(0.0)
                    continue
                m = evaluate_alignment(pi, sim, ref, sim_axis=0,
                                       label_key=label_key, spatial_key=spatial_key)
                gpr_scores.append(float(m.get("gpr", 0.0)))
                lta_scores.append(float(m.get("lta", np.nan) or np.nan))
            pts.append({
                "value": float(v),
                "gpr": float(np.nanmean(gpr_scores)) if gpr_scores else 0.0,
                "lta": float(np.nanmean(lta_scores)) if lta_scores else float("nan"),
            })
        curves[axis] = pts
    return curves


def metric_agreement(sweep_rows):
    """Rank-correlate LTA and expression correlation against GPR across a weight grid.

    GPR is the label-free deployment selector; high positive Spearman with LTA and
    expr_corr validates that the label-free optimum agrees with label-based ground truth.
    FOSCTTM is also included (negated, since lower FOSCTTM = better alignment).
    """
    from scipy.stats import spearmanr

    rows = [r for r in sweep_rows
            if r.get("n_ok", 0) > 0 and r.get("gpr") is not None]
    if len(rows) < 3:
        return {"n_points": len(rows)}
    gpr = np.array([r["gpr"] for r in rows], dtype=float)
    out = {"n_points": len(rows)}
    for name, key, negate in [
        ("lta", "lta", False),
        ("foscttm", "foscttm", True),   # lower FOSCTTM = better; negate for positive correlation
        ("expr_corr", "expr_corr", False),
    ]:
        valid = [r for r in rows if r.get(key) is not None]
        if len(valid) < 3:
            continue
        gpr_v = np.array([r["gpr"] for r in valid], dtype=float)
        vals = np.array([r[key] for r in valid], dtype=float)
        if negate:
            vals = -vals
        rho = spearmanr(gpr_v, vals, nan_policy="omit").correlation
        out[f"spearman_gpr_{name}"] = float(rho) if rho is not None else float("nan")
    return out


def _offset_simplex_grid(step, offset):
    """Simplex grid whose individual values run from offset to <1, stepped by step.

    Neither beta nor gamma will equal exactly 0 or 1 as long as offset > 0 and
    offset + step < 1. With step=0.2 and offset=0.1 the values are
    {0.1, 0.3, 0.5, 0.7, 0.9}, giving 15 interior points on the simplex.
    """
    n = round(1.0 / step)
    pts = []
    for i in range(n + 1):
        beta = offset + i * step
        if beta >= 1.0 - 1e-9:
            break
        for j in range(n + 1):
            gamma = offset + j * step
            if gamma >= 1.0 - 1e-9:
                break
            if beta + gamma <= 1.0 + 1e-9:
                pts.append((round(beta, 9), round(gamma, 9)))
    return pts


def _sensitivity_weight_list(best, alpha_grid, simplex_step, delta_value):
    """Weight dicts that vary alpha (others at best) and the feature simplex (others
    at best) — a focused, interpretable sweep for sensitivity + agreement."""
    wl = []
    for a in alpha_grid:
        wl.append({**best, "alpha": float(a)})
    for (beta, gamma) in _offset_simplex_grid(simplex_step, offset=0.1):
        wl.append({**best, "beta": float(beta), "gamma": float(gamma),
                   "delta": float(delta_value)})
    # de-duplicate
    seen, uniq = set(), []
    for w in wl:
        key = tuple(round(w[k], 6) for k in ("alpha", "beta", "gamma", "alpha_cluster", "delta"))
        if key not in seen:
            seen.add(key)
            uniq.append(w)
    return uniq


def _is_stable_pi(pi, min_total_mass=0.05):
    """Return True if the transport plan is numerically usable.

    Two failure modes at small reg_m:
    - Solver divergence: pi contains NaN or Inf.
    - Mass collapse: FUGW dropped nearly all mass so pi.sum() << expected (~1.0).
      We flag anything below min_total_mass (default 5% of a balanced plan).
    """
    if not np.isfinite(pi).all():
        return False
    if float(pi.sum()) < min_total_mass:
        return False
    return True


def reg_m_sensitivity_sweep(
    instances,
    reg_m_grid,
    best_weights,
    *,
    aligner=_default_aligner,
    base_align_kwargs,
    label_key="cell_type_annot",
    spatial_key="spatial",
    min_total_mass=0.05,
):
    """Sweep reg_m at fixed best_weights; return one row per value with averaged metrics.

    Each row contains all metric keys from :func:`grid_sweep_full` plus ``reg_m``,
    ``n_unstable`` (instances whose plan failed the stability check), and
    ``numerically_stable`` (True only when every instance passed).

    Small reg_m values allow FUGW to drop mass freely; this can produce NaN/Inf
    or near-zero total mass. Such plans are excluded from metric averaging and
    counted in ``n_unstable`` rather than silently biasing the scores.
    """
    rows = []
    for reg_m in reg_m_grid:
        kwargs = {**base_align_kwargs, "reg_m": float(reg_m)}
        n_unstable = 0
        mets = []
        for sim, ref in instances:
            pi = _align(aligner, sim, ref, best_weights, kwargs, quiet=True)
            if pi is None:
                n_unstable += 1
                continue
            if not _is_stable_pi(pi, min_total_mass=min_total_mass):
                n_unstable += 1
                continue
            mets.append(evaluate_alignment(
                pi, sim, ref, sim_axis=0,
                label_key=label_key, spatial_key=spatial_key,
            ))

        row: dict = {"n_ok": len(mets), "n_unstable": n_unstable}
        if mets:
            keys = set().union(*[m.keys() for m in mets])
            for k in keys:
                vals = []
                for m in mets:
                    v = m.get(k, np.nan)
                    if isinstance(v, (int, float, np.floating, np.integer)):
                        vals.append(float(v))
                    elif v is None:
                        vals.append(np.nan)
                if vals:
                    row[k] = float(np.nanmean(vals))

        row["reg_m"] = float(reg_m)
        row["numerically_stable"] = (n_unstable == 0)
        rows.append(row)
    return rows


def run_weight_benchmark(
    section,
    reference,
    *,
    crops=None,
    dev_perturb: dict,
    test_perturb: dict,
    severity_axes: dict,
    baseline_perturb: dict,
    aligner=_default_aligner,
    n_dev_instances: int = 3,
    n_test_instances: int = 3,
    n_robust_instances: int = 2,
    selection_kwargs: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    sensitivity_alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    sensitivity_simplex_step: float = 0.2,
    reg_m_grid=(0.1, 0.5, 1.0, 5.0, 10.0),
    outdir: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """
    Full benchmark. Returns a JSON-serializable results dict with sections:
    ``selection`` (robust defaults + landscape), ``generalization`` (held-out
    metric battery), ``robustness`` (per-axis curves), ``sensitivity`` (full-battery
    weight sweep), and ``metric_agreement`` (GPR vs label-based proxy agreement).
    If ``outdir`` is given, writes ``results.json`` and figures.
    """
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    label_key = align_kwargs.get("label_key", "cell_type_annot")
    spatial_key = align_kwargs.get("spatial_key", "spatial")
    if align_kwargs["use_gpu"]:
        print("[benchmark] GPU (CUDA) enabled for alignment OT.")
    else:
        print("[benchmark] running on CPU (no CUDA device detected or use_gpu=False).")

    # A. select robust defaults on the development split
    sel = select_alignment_weights(
        section=section, reference=reference, crops=crops,
        n_instances=n_dev_instances, perturb_kwargs=dev_perturb,
        align_kwargs=align_kwargs, seed=seed, **(selection_kwargs or {}))
    best = sel["best"]

    # B. generalization on a held-out instance split (different seed + severities)
    test_inst = make_self_alignment_instances(
        section=section, reference=reference, crops=crops,
        n_instances=n_test_instances, perturb_kwargs=test_perturb, seed=seed + 7)
    gen_row = grid_sweep_full(test_inst, [best], aligner=aligner, align_kwargs=align_kwargs,
                              label_key=label_key, spatial_key=spatial_key)[0]

    # C. robustness to each nuisance severity axis
    curves = robustness_curves(
        section, reference, best, severity_axes, crops=crops,
        baseline_perturb=baseline_perturb, aligner=aligner,
        n_instances=n_robust_instances, align_kwargs=align_kwargs,
        label_key=label_key, spatial_key=spatial_key, seed=seed + 13)

    # D. sensitivity + metric agreement on the held-out instances
    wl = _sensitivity_weight_list(best, sensitivity_alpha_grid, sensitivity_simplex_step,
                                  best["delta"])
    sweep = grid_sweep_full(test_inst, wl, aligner=aligner, align_kwargs=align_kwargs,
                            label_key=label_key, spatial_key=spatial_key)
    agree = metric_agreement(sweep)

    # E. reg_m sensitivity: sweep marginal-relaxation at fixed best weights
    reg_m_sens = reg_m_sensitivity_sweep(
        test_inst, reg_m_grid, best,
        aligner=aligner,
        base_align_kwargs=align_kwargs,
        label_key=label_key,
        spatial_key=spatial_key,
    )

    results = {
        "selection": {"best": best, "best_score": sel["best_score"],
                      "objective_key": sel["objective_key"], "landscape": sel["landscape"]},
        "generalization": gen_row,
        "robustness": curves,
        "sensitivity": sweep,
        "metric_agreement": agree,
        "reg_m_sensitivity": reg_m_sens,
        "config": {"dev_perturb": dev_perturb, "test_perturb": test_perturb,
                   "baseline_perturb": baseline_perturb, "severity_axes": severity_axes,
                   "n_dev_instances": n_dev_instances, "n_test_instances": n_test_instances,
                   "n_robust_instances": n_robust_instances, "seed": seed,
                   "reg_m_grid": list(reg_m_grid)},
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
        try:
            make_benchmark_figures(results, outdir)
        except Exception as e:
            print(f"[benchmark] figure generation skipped: {e}")

    return results


def make_benchmark_figures(results, outdir):
    """Render headline figures (robustness curves, weight sensitivity, and the
    GPR vs LTA agreement scatter) as PNGs in ``outdir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # robustness curves: GPR and LTA vs perturbation severity
    curves = results["robustness"]
    if curves:
        n = len(curves)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
        for ax, (axis, pts) in zip(axes[0], curves.items()):
            xs = [p["value"] for p in pts]
            ax.plot(xs, [p["gpr"] for p in pts], "o-", label="GPR")
            ax.plot(xs, [p.get("lta", float("nan")) for p in pts],
                    "s--", label="LTA", alpha=0.7)
            ax.set_xlabel(axis)
            ax.set_ylabel("score")
            ax.set_ylim(0, 1.02)
            ax.set_title(f"robustness: {axis}", fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "robustness_curves.png"), dpi=150)
        plt.close(fig)

    sweep = results["sensitivity"]
    ok = [r for r in sweep if r.get("n_ok", 0) > 0]
    if ok:
        best = results["selection"]["best"]
        alpha_rows = sorted(
            [r for r in ok if abs(r["beta"] - best["beta"]) < 1e-9
             and abs(r["gamma"] - best["gamma"]) < 1e-9
             and abs(r["delta"] - best["delta"]) < 1e-9],
            key=lambda r: r["alpha"])
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        if alpha_rows:
            ax[0].plot([r["alpha"] for r in alpha_rows],
                       [r["gpr"] for r in alpha_rows], "o-")
            ax[0].set_xlabel("alpha")
            ax[0].set_ylabel("GPR")
            ax[0].set_ylim(0, 1.02)
            ax[0].set_title("alpha sensitivity", fontsize=9)
        # GPR vs LTA agreement scatter
        valid = [r for r in ok if r.get("lta") is not None]
        if valid:
            ax[1].scatter([r["lta"] for r in valid], [r["gpr"] for r in valid], s=18)
        rho = results["metric_agreement"].get("spearman_gpr_lta", float("nan"))
        ax[1].set_xlabel("LTA (label-based)")
        ax[1].set_ylabel("GPR (label-free)")
        ax[1].set_title(f"GPR vs LTA  rho={rho:.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "sensitivity_agreement.png"), dpi=150)
        plt.close(fig)

    reg_m_rows = results.get("reg_m_sensitivity", [])
    if reg_m_rows:
        stable = [r for r in reg_m_rows if r.get("numerically_stable", True) and r.get("n_ok", 0) > 0]
        unstable = [r for r in reg_m_rows if not r.get("numerically_stable", True)]
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.4))
        if stable:
            xs = [r["reg_m"] for r in stable]
            ax.semilogx(xs, [r["gpr"] for r in stable], "o-", label="GPR")
            lta_vals = [r.get("lta", float("nan")) for r in stable]
            if any(v == v for v in lta_vals):  # any non-NaN
                ax.semilogx(xs, lta_vals, "s--", label="LTA", alpha=0.7)
        for r in unstable:
            ax.axvline(r["reg_m"], color="red", linestyle=":", linewidth=0.8, alpha=0.6)
        if unstable:
            ax.axvline(unstable[0]["reg_m"], color="red", linestyle=":", linewidth=0.8,
                       alpha=0.6, label="unstable")
        ax.set_xlabel("reg_m (log scale)")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1.02)
        ax.set_title("reg_m sensitivity", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "reg_m_sensitivity.png"), dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------------
# default configuration + CLI
# ----------------------------------------------------------------------------

DEFAULT_DEV_PERTURB = dict(dropout_rate=0.10, warp_amplitude=None, jitter_sigma=None,
                           rotation_range=(-30.0, 30.0), expr_alpha=1.0,
                           label_flip_rate=0.05, birth_rate=0.05)
DEFAULT_TEST_PERTURB = dict(dropout_rate=0.20, warp_amplitude=None, jitter_sigma=None,
                            rotation_range=(-45.0, 45.0), reflect=True, expr_alpha=1.5,
                            label_flip_rate=0.10, birth_rate=0.10)
DEFAULT_BASELINE_PERTURB = dict(dropout_rate=0.10, expr_alpha=1.0,
                                label_flip_rate=0.05, birth_rate=0.05)
DEFAULT_SEVERITY_AXES = {
    "dropout_rate": [0.0, 0.1, 0.2, 0.3, 0.4],
    "expr_alpha": [0.0, 0.5, 1.0, 2.0, 3.0],
    "label_flip_rate": [0.0, 0.1, 0.2, 0.3],
    "birth_rate": [0.0, 0.1, 0.2, 0.3],
}


if __name__ == "__main__":
    import argparse
    import scanpy as sc

    ap = argparse.ArgumentParser(description="Synthetic weight-selection benchmark for INCENT.")
    ap.add_argument("--reference_h5ad", required=True, help="Full parent slice (.h5ad).")
    ap.add_argument("--section_h5ad", required=True,
                    help="Manually cropped section (.h5ad, e.g. from synthesize.py); "
                         "parent-frame coords with obs_names subsetting the reference.")
    ap.add_argument("--outdir", default="results/weight_benchmark")
    ap.add_argument("--use_gpu", choices=["auto", "true", "false"], default="auto",
                    help="Use CUDA for the alignment OT. 'auto' (default) uses the GPU if available.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--reg_m_grid", type=float, nargs="+", default=[0.1, 0.5, 1.0, 5.0, 10.0],
        metavar="V",
        help="Space-separated reg_m values for the marginal-relaxation sensitivity sweep "
             "(default: 0.1 0.5 1.0 5.0 10.0).",
    )
    args = ap.parse_args()

    if args.use_gpu == "auto":
        use_gpu = gpu_available()
    else:
        use_gpu = (args.use_gpu == "true")

    reference = sc.read_h5ad(args.reference_h5ad)
    section = sc.read_h5ad(args.section_h5ad)
    res = run_weight_benchmark(
        section,
        reference,
        dev_perturb=DEFAULT_DEV_PERTURB,
        test_perturb=DEFAULT_TEST_PERTURB,
        severity_axes=DEFAULT_SEVERITY_AXES,
        baseline_perturb=DEFAULT_BASELINE_PERTURB,
        align_kwargs={"use_gpu": use_gpu},
        reg_m_grid=tuple(args.reg_m_grid),
        outdir=args.outdir,
        seed=args.seed,
    )
    print("Selected defaults:", res["selection"]["best"])
    print("Held-out GPR:", res["generalization"].get("gpr"))
    print("Held-out LTA:", res["generalization"].get("lta"))
    print("Metric agreement:", res["metric_agreement"])
    print(f"Results + figures written to {args.outdir}")
