"""
benchmark/run_weight_benchmark.py
=================================
Synthetic, ground-truth-anchored benchmark that (a) selects robust default
alignment weights, (b) reports their generalization on a held-out instance
split, (c) measures robustness to every nuisance severity axis, and (d)
validates the label-free deployment selector by correlating the label-free
metric (spatial coherence) with the exact registration metric across a weight
grid.

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
    simplex_grid,
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
    with the full metric battery averaged over instances (NaN-safe)."""
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
                row[k] = float(np.nanmean([m.get(k, np.nan) for m in mets]))
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
    registration + coherence."""
    curves = {}
    for ai, (axis, values) in enumerate(severity_axes.items()):
        pts = []
        for v in values:
            pk = dict(baseline_perturb)
            pk[axis] = v
            inst = make_self_alignment_instances(
                section=section, reference=reference, crops=crops,
                n_instances=n_instances, perturb_kwargs=pk, seed=seed + 100 * ai)
            regs, cohs = [], []
            for sim, ref in inst:
                pi = _align(aligner, sim, ref, best_weights, align_kwargs, True)
                if pi is None:
                    regs.append(0.0)
                    continue
                m = evaluate_alignment(pi, sim, ref, sim_axis=0,
                                       label_key=label_key, spatial_key=spatial_key)
                regs.append(float(m.get("reg_soft_corr_mass", 0.0)))
                cohs.append(float(m.get("coherence", np.nan)))
            pts.append({"value": float(v),
                        "reg_soft_corr_mass": float(np.mean(regs)) if regs else 0.0,
                        "coherence": float(np.nanmean(cohs)) if cohs else float("nan")})
        curves[axis] = pts
    return curves


def metric_agreement(sweep_rows):
    """Rank-correlate the label-free metrics against the exact registration metric
    across a weight grid. High positive Spearman => coherence is a valid label-free
    proxy, justifying the deployment selector."""
    from scipy.stats import spearmanr

    rows = [r for r in sweep_rows if r.get("n_ok", 0) > 0 and "reg_soft_corr_mass" in r]
    if len(rows) < 3:
        return {"n_points": len(rows)}
    reg = np.array([r["reg_soft_corr_mass"] for r in rows], dtype=float)
    out = {"n_points": len(rows)}
    for name, key in (("coherence", "coherence"), ("ltari", "ltari"),
                      ("expr_corr", "expr_corr")):
        if all(key in r for r in rows):
            vals = np.array([r[key] for r in rows], dtype=float)
            rho = spearmanr(reg, vals, nan_policy="omit").correlation
            out[f"spearman_reg_{name}"] = float(rho) if rho is not None else float("nan")
    return out


def _sensitivity_weight_list(best, alpha_grid, simplex_step, delta_value):
    """Weight dicts that vary alpha (others at best) and the feature simplex (others
    at best) -- a focused, interpretable sweep for sensitivity + agreement."""
    wl = []
    for a in alpha_grid:
        wl.append({**best, "alpha": float(a)})
    for (beta, gamma) in simplex_grid(simplex_step):
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
    sensitivity_simplex_step: float = 0.25,
    outdir: Optional[str] = None,
    seed: int = 0,
) -> dict:
    """
    Full benchmark. Returns a JSON-serializable results dict with sections:
    ``selection`` (robust defaults + landscape), ``generalization`` (held-out
    metric battery), ``robustness`` (per-axis curves), ``sensitivity`` (full-battery
    weight sweep), and ``metric_agreement`` (registration vs label-free proxies).
    If ``outdir`` is given, writes ``results.json`` and figures.
    """
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())  # use CUDA for the FGW OT if present
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

    results = {
        "selection": {"best": best, "best_score": sel["best_score"],
                      "objective_key": sel["objective_key"], "landscape": sel["landscape"]},
        "generalization": gen_row,
        "robustness": curves,
        "sensitivity": sweep,
        "metric_agreement": agree,
        "config": {"dev_perturb": dev_perturb, "test_perturb": test_perturb,
                   "baseline_perturb": baseline_perturb, "severity_axes": severity_axes,
                   "n_dev_instances": n_dev_instances, "n_test_instances": n_test_instances,
                   "n_robust_instances": n_robust_instances, "seed": seed},
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
        try:
            make_benchmark_figures(results, outdir)
        except Exception as e:  # plotting is best-effort
            print(f"[benchmark] figure generation skipped: {e}")

    return results


def make_benchmark_figures(results, outdir):
    """Render the headline figures (robustness curves, weight sensitivity, and the
    registration-vs-coherence agreement scatter) as PNGs in ``outdir``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    # robustness curves
    curves = results["robustness"]
    if curves:
        n = len(curves)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
        for ax, (axis, pts) in zip(axes[0], curves.items()):
            xs = [p["value"] for p in pts]
            ax.plot(xs, [p["reg_soft_corr_mass"] for p in pts], "o-", label="registration")
            ax.plot(xs, [p["coherence"] for p in pts], "s--", label="coherence", alpha=0.7)
            ax.set_xlabel(axis); ax.set_ylabel("score"); ax.set_ylim(0, 1.02)
            ax.set_title(f"robustness: {axis}", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "robustness_curves.png"), dpi=150)
        plt.close(fig)

    sweep = results["sensitivity"]
    ok = [r for r in sweep if r.get("n_ok", 0) > 0]
    if ok:
        # alpha sensitivity (rows where only alpha varies relative to best)
        best = results["selection"]["best"]
        alpha_rows = sorted(
            [r for r in ok if abs(r["beta"] - best["beta"]) < 1e-9
             and abs(r["gamma"] - best["gamma"]) < 1e-9 and abs(r["delta"] - best["delta"]) < 1e-9],
            key=lambda r: r["alpha"])
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
        if alpha_rows:
            ax[0].plot([r["alpha"] for r in alpha_rows],
                       [r["reg_soft_corr_mass"] for r in alpha_rows], "o-")
            ax[0].set_xlabel("alpha"); ax[0].set_ylabel("registration"); ax[0].set_ylim(0, 1.02)
            ax[0].set_title("alpha sensitivity (plateau?)", fontsize=9)
        # registration vs coherence agreement
        reg = [r["reg_soft_corr_mass"] for r in ok if "coherence" in r]
        coh = [r["coherence"] for r in ok if "coherence" in r]
        ax[1].scatter(coh, reg, s=18)
        rho = results["metric_agreement"].get("spearman_reg_coherence", float("nan"))
        ax[1].set_xlabel("spatial coherence (label-free)"); ax[1].set_ylabel("registration (GT)")
        ax[1].set_title(f"agreement  rho={rho:.2f}", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "sensitivity_agreement.png"), dpi=150)
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
        outdir=args.outdir,
        seed=args.seed,
    )
    print("Selected defaults:", res["selection"]["best"])
    print("Held-out registration:", res["generalization"].get("reg_soft_corr_mass"))
    print("Metric agreement:", res["metric_agreement"])
    print(f"Results + figures written to {args.outdir}")
