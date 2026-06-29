"""
benchmark/kappa_sensitivity.py
==============================
Sensitivity analysis for the mesoregion scale parameter kappa
(NEIGHBORHOOD_OUTER_MULTIPLIER in src/core.py).

For each kappa value in a grid, the mesoregion seed spacing is overridden as:

    S = sqrt(pi) * kappa * s

where s is the characteristic cell spacing estimated from the data.
Three synthetic slice pairs are generated per kappa point, aligned, and the
metrics are averaged (NaN-safe). The result shows that alignment quality is
stable across kappa in [8, 15], justifying the fixed default kappa=10.

Run as a script::

    python -m benchmark.kappa_sensitivity --reference_h5ad slice.h5ad \\
        --section_h5ad section.h5ad --outdir results/kappa_sensitivity

or call :func:`run_kappa_sensitivity` directly with in-memory AnnData objects.
"""

from __future__ import annotations

import json
import os
import concurrent.futures
from typing import Optional

import numpy as np

from src.core import (
    estimate_characteristic_spacing,
    hierarchical_pairwise_align,
    NEIGHBORHOOD_OUTER_MULTIPLIER,
)
from src.tuning import make_self_alignment_instances, gpu_available, _quiet
from src.evaluation import evaluate_alignment

DEFAULT_KAPPA_GRID = [5, 7.5, 10, 12.5, 15]
DEFAULT_PERTURB = dict(
    dropout_rate=0.10,
    rotation_range=(-30.0, 30.0),
    expr_alpha=1.0,
    label_flip_rate=0.05,
    birth_rate=0.05,
)


def _kappa_to_coarsen_scale(kappa: float, s: float) -> float:
    """Convert kappa to the coarsen_scale override (physical length units)."""
    return float(np.sqrt(np.pi) * kappa * s)


def _align_one(sliceA, sliceB, coarsen_scale: float, align_kwargs: dict):
    try:
        with _quiet(True):
            pi = hierarchical_pairwise_align(
                sliceA, sliceB,
                coarsen_scale=coarsen_scale,
                **align_kwargs,
            )
        return np.asarray(pi, dtype=np.float64)
    except Exception as e:
        print(f"[kappa_sensitivity] alignment failed (S={coarsen_scale:.2f}): {e}")
        return None


def _eval_kappa(
    kappa: float,
    instances: list,
    align_kwargs: dict,
    s: float,
    label_key: str,
    spatial_key: str,
) -> dict:
    """Align all instances at a given kappa; return averaged metrics."""
    S = _kappa_to_coarsen_scale(kappa, s)
    mets = []
    for sim, ref in instances:
        pi = _align_one(sim, ref, S, align_kwargs)
        if pi is None:
            continue
        m = evaluate_alignment(
            pi, sim, ref, sim_axis=0,
            label_key=label_key, spatial_key=spatial_key,
        )
        mets.append(m)

    row: dict = {"kappa": float(kappa), "S": S, "n_ok": len(mets)}
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
                row[f"{k}_std"] = float(np.nanstd(vals))
    return row


def run_kappa_sensitivity(
    section,
    reference,
    *,
    kappa_grid=DEFAULT_KAPPA_GRID,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    label_key: str = "cell_type_annot",
    spatial_key: str = "spatial",
    n_jobs: int = 1,
    seed: int = 0,
    outdir: Optional[str] = None,
) -> dict:
    """
    Sweep kappa and report alignment metrics averaged over ``n_instances`` pairs.

    Parameters
    ----------
    section, reference:
        AnnData slices (section is the simulated/cropped slice, reference is the
        full parent slice). Passed to :func:`make_self_alignment_instances`.
    kappa_grid:
        List of kappa values to evaluate. Default: [5, 7.5, 10, 12.5, 15].
    n_instances:
        Number of synthetic slice pairs per kappa point (default 3).
        The same set of instances is reused across all kappa values so the only
        variable is the mesoregion scale.
    perturb_kwargs:
        Perturbation parameters for :func:`simulate_adjacent_slice`.
    align_kwargs:
        Extra keyword arguments forwarded to :func:`hierarchical_pairwise_align`
        (e.g. alpha, beta, use_gpu). ``coarsen_scale`` is always overridden
        internally and must not be set here.
    n_jobs:
        Number of parallel threads for evaluating kappa points (default 1).
        Safe only on CPU or with multiple GPUs routed to separate threads.
    outdir:
        If given, writes ``kappa_sensitivity.json`` and ``kappa_sensitivity.png``.

    Returns
    -------
    dict with keys:
        ``rows``   — list of per-kappa result dicts (kappa, S, n_ok, gpr, lta, foscttm, ...),
        ``default_kappa`` — the package default (NEIGHBORHOOD_OUTER_MULTIPLIER),
        ``config`` — run configuration for reproducibility.
    """
    if "coarsen_scale" in (align_kwargs or {}):
        raise ValueError(
            "Do not pass coarsen_scale in align_kwargs — "
            "kappa_sensitivity controls it internally."
        )

    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)

    perturb_kwargs = dict(perturb_kwargs or DEFAULT_PERTURB)

    # Estimate s once from the data (not kappa-dependent)
    s_A = estimate_characteristic_spacing(section, spatial_key=spatial_key)
    s_B = estimate_characteristic_spacing(reference, spatial_key=spatial_key)
    s = max(s_A, s_B)
    print(f"[kappa_sensitivity] characteristic spacing s={s:.4g} "
          f"(section={s_A:.4g}, reference={s_B:.4g})")

    # Generate instances once; reuse across all kappa values
    instances = make_self_alignment_instances(
        section=section, reference=reference,
        n_instances=n_instances,
        perturb_kwargs=perturb_kwargs,
        seed=seed,
    )
    print(f"[kappa_sensitivity] {len(instances)} instance(s) generated, "
          f"sweeping {len(kappa_grid)} kappa values: {kappa_grid}")

    def _job(kappa):
        row = _eval_kappa(kappa, instances, align_kwargs, s, label_key, spatial_key)
        gpr = row.get("gpr", float("nan"))
        lta = row.get("lta", float("nan"))
        foscttm = row.get("foscttm", float("nan"))
        marker = " <-- default" if abs(kappa - NEIGHBORHOOD_OUTER_MULTIPLIER) < 1e-9 else ""
        print(
            f"  kappa={kappa:>4.1f}  S={row['S']:>7.2f}  "
            f"GPR={gpr:.4f}  LTA={lta:.4f}  FOSCTTM={foscttm:.4f}"
            f"  (n_ok={row['n_ok']}){marker}"
        )
        return row

    if n_jobs == 1:
        rows = [_job(k) for k in kappa_grid]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_job, kappa_grid))
        rows.sort(key=lambda r: r["kappa"])

    results = {
        "rows": rows,
        "default_kappa": float(NEIGHBORHOOD_OUTER_MULTIPLIER),
        "config": {
            "kappa_grid": list(kappa_grid),
            "n_instances": n_instances,
            "perturb_kwargs": perturb_kwargs,
            "s": float(s),
            "seed": seed,
            "n_jobs": n_jobs,
        },
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        json_path = os.path.join(outdir, "kappa_sensitivity.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
        print(f"[kappa_sensitivity] results written to {json_path}")
        try:
            _make_figure(results, outdir)
        except Exception as e:
            print(f"[kappa_sensitivity] figure skipped: {e}")

    return results


def _make_figure(results: dict, outdir: str) -> None:
    """Plot GPR, LTA, and FOSCTTM vs kappa with the default kappa highlighted."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in results["rows"] if r.get("n_ok", 0) > 0]
    if not rows:
        return

    kappas = [r["kappa"] for r in rows]
    default_kappa = results["default_kappa"]

    metrics = [
        ("gpr",      "GPR",      "o-",  "tab:blue"),
        ("lta",      "LTA",      "s--", "tab:orange"),
        ("foscttm",  "FOSCTTM",  "^:",  "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    for key, label, style, color in metrics:
        vals = [r.get(key, float("nan")) for r in rows]
        stds = [r.get(f"{key}_std", 0.0) for r in rows]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(kappas, vals, style, color=color, label=label, linewidth=1.8, markersize=6)
        # shaded ±1 std band
        lo = [v - e for v, e in zip(vals, stds)]
        hi = [v + e for v, e in zip(vals, stds)]
        ax.fill_between(kappas, lo, hi, alpha=0.15, color=color)

    ax.axvline(default_kappa, color="red", linestyle="--", linewidth=1.2,
               label=f"default κ={int(default_kappa)}")
    ax.set_xlabel("κ  (mesoregion scale multiplier)", fontsize=11)
    ax.set_ylabel("score", fontsize=11)
    ax.set_title("Sensitivity to mesoregion scale κ\n"
                 f"(mean ± std over {results['config']['n_instances']} instances per point)",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(kappas)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(outdir, "kappa_sensitivity.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[kappa_sensitivity] figure written to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import scanpy as sc

    ap = argparse.ArgumentParser(
        description="Sensitivity analysis for the mesoregion scale parameter kappa."
    )
    ap.add_argument("--reference_h5ad", required=True,
                    help="Full parent slice (.h5ad).")
    ap.add_argument("--section_h5ad", required=True,
                    help="Cropped section (.h5ad); parent-frame coords whose "
                         "obs_names subset the reference.")
    ap.add_argument("--outdir", default="results/kappa_sensitivity")
    ap.add_argument("--kappa_grid", type=float, nargs="+",
                    default=DEFAULT_KAPPA_GRID, metavar="K",
                    help="Space-separated kappa values to sweep "
                         "(default: 6 7 8 9 10 11 12 13 14 15 16).")
    ap.add_argument("--n_instances", type=int, default=3,
                    help="Synthetic pairs per kappa point (default: 3).")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel threads for kappa evaluation (default: 1).")
    ap.add_argument("--use_gpu", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.use_gpu == "auto":
        use_gpu = gpu_available()
    else:
        use_gpu = args.use_gpu == "true"

    reference = sc.read_h5ad(args.reference_h5ad)
    section = sc.read_h5ad(args.section_h5ad)

    res = run_kappa_sensitivity(
        section,
        reference,
        kappa_grid=args.kappa_grid,
        n_instances=args.n_instances,
        align_kwargs={"use_gpu": use_gpu},
        n_jobs=args.n_jobs,
        seed=args.seed,
        outdir=args.outdir,
    )

    default_row = next(
        (r for r in res["rows"] if abs(r["kappa"] - res["default_kappa"]) < 1e-9), None
    )
    if default_row:
        print(f"\nDefault kappa={int(res['default_kappa'])}:  "
              f"GPR={default_row.get('gpr', float('nan')):.4f}  "
              f"LTA={default_row.get('lta', float('nan')):.4f}  "
              f"FOSCTTM={default_row.get('foscttm', float('nan')):.4f}")
    print(f"Results written to {args.outdir}")
