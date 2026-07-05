"""
benchmark/reg_m_sensitivity.py
===============================
Sensitivity analysis for the marginal-relaxation penalty reg_m
(DEFAULT_REG_M in src/core.py, passed as ``reg_marginals`` to the unbalanced
coarse FUGW in src/hierarchical.py::run_coarse_fugw).

For each reg_m value in a grid, alignment is run with that reg_m (all other
weights left at their package defaults). Three synthetic slice pairs are
generated per reg_m point, aligned, and the metrics are averaged (NaN-safe).

Small reg_m lets FUGW drop mass freely, which can produce a numerically
degenerate plan (NaN/Inf entries, or near-zero total transported mass); such
runs are flagged unstable and excluded from the metric average rather than
silently biasing the scores. The result shows that alignment quality is
stable wherever reg_m is numerically stable, justifying the fixed default
reg_m=1.0.

Run as a script::

    python -m benchmark.reg_m_sensitivity --reference_h5ad slice.h5ad \\
        --section_h5ad section.h5ad --outdir results/reg_m_sensitivity

or call :func:`run_reg_m_sensitivity` directly with in-memory AnnData objects.
"""

from __future__ import annotations

import json
import os
import concurrent.futures
from queue import Queue
from typing import Callable, Optional

import numpy as np

from src.core import hierarchical_pairwise_align, DEFAULT_REG_M
from src.tuning import make_self_alignment_instances, gpu_available, _quiet
from src.evaluation import evaluate_alignment

DEFAULT_REG_M_GRID = [0.1, 0.5, 1.0, 5.0, 10.0]
DEFAULT_PERTURB = dict(
    dropout_rate=0.10,
    rotation_range=(-180.0, 180.0),
    expr_alpha=1.0,
    label_flip_rate=0.05,
    birth_rate=0.1,
)


def _is_stable_pi(pi: np.ndarray, min_total_mass: float = 0.05) -> bool:
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


def _make_device_aware_aligner(aligner: Callable, device_ids: list) -> Callable:
    """Wrap aligner so each thread picks a GPU from the pool via torch.cuda.device().

    Mirrors ``_make_device_pool_score`` in src/tuning.py, but wraps an
    (sliceA, sliceB, **kwargs) aligner signature instead of a score_fn.
    ``torch.cuda.device()`` is thread-local, so concurrent threads (one per
    reg_m point, via ``n_jobs``) can each hold a different device without
    cross-contamination. Returns the original aligner unchanged when CUDA is
    not available.
    """
    try:
        import torch
    except ImportError:
        return aligner
    if not torch.cuda.is_available():
        return aligner
    pool: Queue = Queue()
    for did in device_ids:
        pool.put(did)

    def _wrapped(sliceA, sliceB, **kwargs):
        did = pool.get()
        try:
            with torch.cuda.device(did):
                return aligner(sliceA, sliceB, **kwargs)
        finally:
            pool.put(did)

    return _wrapped


def _align_one(aligner: Callable, sliceA, sliceB, reg_m: float, align_kwargs: dict):
    try:
        with _quiet(True):
            pi = aligner(
                sliceA, sliceB,
                reg_m=reg_m,
                **align_kwargs,
            )
        return np.asarray(pi, dtype=np.float64)
    except Exception as e:
        print(f"[reg_m_sensitivity] alignment failed (reg_m={reg_m:.4g}): {e}")
        return None


def _eval_reg_m(
    reg_m: float,
    instances: list,
    align_kwargs: dict,
    label_key: str,
    min_total_mass: float,
    aligner: Callable = hierarchical_pairwise_align,
) -> dict:
    """Align all instances at a given reg_m; return averaged metrics."""
    mets = []
    n_unstable = 0
    for sim, ref in instances:
        pi = _align_one(aligner, sim, ref, reg_m, align_kwargs)
        if pi is None:
            n_unstable += 1
            continue
        if not _is_stable_pi(pi, min_total_mass=min_total_mass):
            n_unstable += 1
            continue
        m = evaluate_alignment(
            pi, sim, ref, sim_axis=0,
            label_key=label_key,
        )
        mets.append(m)

    row: dict = {"reg_m": float(reg_m), "n_ok": len(mets), "n_unstable": n_unstable}
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
    row["numerically_stable"] = (n_unstable == 0)
    return row


def run_reg_m_sensitivity(
    section=None,
    reference=None,
    *,
    crops=None,
    reg_m_grid=DEFAULT_REG_M_GRID,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    label_key: str = "cell_type_annot",
    min_total_mass: float = 0.05,
    n_jobs: int = 1,
    device_ids=None,
    seed: int = 0,
    outdir: Optional[str] = None,
) -> dict:
    """
    Sweep reg_m and report alignment metrics averaged over ``n_instances`` pairs.

    Instance source (priority ``crops`` > ``section``+``reference``), matching
    the convention of :func:`select_alignment_weights` and :func:`run_kappa_sensitivity`.

    Parameters
    ----------
    section, reference:
        AnnData slices (section is the simulated/cropped slice, reference is the
        full parent slice). Used when ``crops`` is None.
    crops:
        Explicit list of ``(section, reference)`` AnnData pairs. When given,
        ``section``, ``reference``, and ``n_instances`` are ignored — one
        instance is produced per crop. Mirrors the ``crops`` argument of
        :func:`select_alignment_weights`.
    reg_m_grid:
        List of reg_m values to evaluate. Default: [0.1, 0.5, 1.0, 5.0, 10.0].
    n_instances:
        Number of synthetic pairs per reg_m point when using
        ``section``+``reference`` mode (ignored when ``crops`` is given).
        The same set of instances is reused across all reg_m values so the only
        variable is the marginal-relaxation penalty.
    perturb_kwargs:
        Perturbation parameters for :func:`simulate_adjacent_slice`.
    align_kwargs:
        Extra keyword arguments forwarded to :func:`hierarchical_pairwise_align`
        (e.g. alpha, beta, use_gpu). ``reg_m`` is always overridden internally
        and must not be set here.
    min_total_mass:
        Minimum total transported mass for a plan to count as numerically
        stable (see :func:`_is_stable_pi`).
    n_jobs:
        Number of parallel threads for evaluating reg_m points (default 1).
        Safe only on CPU or with multiple GPUs routed to separate threads.
    device_ids:
        CUDA device indices to distribute reg_m points across (e.g. ``[0, 1]``),
        mirroring the ``device_ids`` argument of :func:`select_alignment_weights`.
        With a single GPU ``n_jobs`` is forced to 1 (concurrent CUDA calls on one
        device are unsafe). With ``len(device_ids) > 1`` and ``n_jobs > 1``, each
        thread grabs a free device from a pool via ``torch.cuda.device()`` so
        different reg_m points align concurrently on different GPUs. Ignored
        when ``use_gpu`` is False.
    outdir:
        If given, writes ``reg_m_sensitivity.json`` and ``reg_m_sensitivity.png``.

    Returns
    -------
    dict with keys:
        ``rows``   — list of per-reg_m result dicts (reg_m, n_ok, n_unstable,
        numerically_stable, lta, foscttm, expr_corr, ...),
        ``default_reg_m`` — the package default (DEFAULT_REG_M),
        ``config`` — run configuration for reproducibility.
    """
    if crops is None and (section is None or reference is None):
        raise ValueError("Provide either `crops` or both `section` and `reference`.")
    if "reg_m" in (align_kwargs or {}):
        raise ValueError(
            "Do not pass reg_m in align_kwargs — "
            "reg_m_sensitivity controls it internally."
        )

    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)

    aligner = hierarchical_pairwise_align
    if align_kwargs["use_gpu"]:
        if device_ids is None:
            device_ids = [0]
        n_gpu = len(device_ids)
        print(f"[reg_m_sensitivity] GPU (CUDA) enabled: {n_gpu} device(s) {device_ids}.")
        if n_gpu == 1 and n_jobs > 1:
            print("[reg_m_sensitivity] n_jobs forced to 1: single GPU, concurrent CUDA calls unsafe.")
            n_jobs = 1
        elif n_gpu > 1 and n_jobs > 1:
            print(f"[reg_m_sensitivity] multi-GPU parallel: {n_gpu} GPUs x n_jobs={n_jobs}.")
            aligner = _make_device_aware_aligner(aligner, device_ids)
    else:
        device_ids = None

    perturb_kwargs = dict(perturb_kwargs or DEFAULT_PERTURB)

    # Generate instances once; reuse across all reg_m values
    instances = make_self_alignment_instances(
        section=section, reference=reference, crops=crops,
        n_instances=n_instances,
        perturb_kwargs=perturb_kwargs,
        seed=seed,
    )
    print(f"[reg_m_sensitivity] {len(instances)} instance(s) generated, "
          f"sweeping {len(reg_m_grid)} reg_m values: {reg_m_grid}")

    def _job(reg_m):
        row = _eval_reg_m(reg_m, instances, align_kwargs, label_key, min_total_mass, aligner=aligner)
        lta = row.get("lta", float("nan"))
        foscttm = row.get("foscttm", float("nan"))
        expr_corr = row.get("expr_corr", float("nan"))
        marker = " <-- default" if abs(reg_m - DEFAULT_REG_M) < 1e-9 else ""
        stability = "" if row["numerically_stable"] else f"  [UNSTABLE x{row['n_unstable']}]"
        print(
            f"  reg_m={reg_m:>7.3g}  "
            f"LTA={lta:.4f}  FOSCTTM={foscttm:.4f}  expr_corr={expr_corr:.4f}"
            f"  (n_ok={row['n_ok']}){stability}{marker}"
        )
        return row

    if n_jobs == 1:
        rows = [_job(rm) for rm in reg_m_grid]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
            rows = list(pool.map(_job, reg_m_grid))
        rows.sort(key=lambda r: r["reg_m"])

    results = {
        "rows": rows,
        "default_reg_m": float(DEFAULT_REG_M),
        "config": {
            "reg_m_grid": list(reg_m_grid),
            "n_instances": n_instances if crops is None else len(instances),
            "mode": "crops" if crops is not None else "section+reference",
            "perturb_kwargs": perturb_kwargs,
            "min_total_mass": min_total_mass,
            "seed": seed,
            "n_jobs": n_jobs,
            "device_ids": device_ids,
        },
    }

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        json_path = os.path.join(outdir, "reg_m_sensitivity.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2,
                      default=lambda o: float(o)
                      if isinstance(o, (np.floating, np.integer)) else str(o))
        print(f"[reg_m_sensitivity] results written to {json_path}")
        try:
            _make_figure(results, outdir)
        except Exception as e:
            print(f"[reg_m_sensitivity] figure skipped: {e}")

    return results


def _make_figure(results: dict, outdir: str) -> None:
    """Plot LTA, FOSCTTM, and expression-transfer correlation vs reg_m (log scale),
    marking unstable points and the default reg_m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = results["rows"]
    stable_rows = [r for r in rows if r.get("n_ok", 0) > 0]
    if not stable_rows:
        return

    default_reg_m = results["default_reg_m"]

    metrics = [
        ("lta",       "LTA",        "o-",  "tab:blue"),
        ("foscttm",   "FOSCTTM",    "s--", "tab:orange"),
        ("expr_corr", "expr_corr",  "^:",  "tab:green"),
    ]

    fig, ax = plt.subplots(figsize=(7, 4))

    reg_ms = [r["reg_m"] for r in stable_rows]
    for key, label, style, color in metrics:
        vals = [r.get(key, float("nan")) for r in stable_rows]
        stds = [r.get(f"{key}_std", 0.0) for r in stable_rows]
        if all(np.isnan(v) for v in vals):
            continue
        ax.plot(reg_ms, vals, style, color=color, label=label, linewidth=1.8, markersize=6)
        # shaded ±1 std band
        lo = [v - e for v, e in zip(vals, stds)]
        hi = [v + e for v, e in zip(vals, stds)]
        ax.fill_between(reg_ms, lo, hi, alpha=0.15, color=color)

    unstable_rows = [r for r in rows if not r.get("numerically_stable", True)]
    for i, r in enumerate(unstable_rows):
        ax.axvline(r["reg_m"], color="red", linestyle=":", linewidth=1.0, alpha=0.6,
                   label="unstable" if i == 0 else None)

    ax.axvline(default_reg_m, color="black", linestyle="--", linewidth=1.2,
               label=f"default reg_m={default_reg_m:g}")
    ax.set_xscale("log")
    ax.set_xlabel("reg_m  (marginal-relaxation penalty, log scale)", fontsize=11)
    ax.set_ylabel("score", fontsize=11)
    ax.set_title("Sensitivity to marginal-relaxation penalty reg_m\n"
                 f"(mean ± std over {results['config']['n_instances']} instances per point)",
                 fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    fig_path = os.path.join(outdir, "reg_m_sensitivity.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"[reg_m_sensitivity] figure written to {fig_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import scanpy as sc

    ap = argparse.ArgumentParser(
        description="Sensitivity analysis for the marginal-relaxation penalty reg_m."
    )
    ap.add_argument("--reference_h5ad",
                    help="Full parent slice (.h5ad). Required unless --crops_h5ad is used.")
    ap.add_argument("--section_h5ad",
                    help="Cropped section (.h5ad); parent-frame coords whose "
                         "obs_names subset the reference. Required unless --crops_h5ad is used.")
    ap.add_argument("--crops_h5ad", nargs="+", metavar="SECTION REF",
                    help="Explicit crop pairs as alternating section/reference .h5ad paths "
                         "(e.g. sec1.h5ad ref1.h5ad sec2.h5ad ref2.h5ad). "
                         "When given, --section_h5ad and --reference_h5ad are ignored.")
    ap.add_argument("--outdir", default="results/reg_m_sensitivity")
    ap.add_argument("--reg_m_grid", type=float, nargs="+",
                    default=DEFAULT_REG_M_GRID, metavar="R",
                    help=f"Space-separated reg_m values to sweep "
                         f"(default: {DEFAULT_REG_M_GRID}).")
    ap.add_argument("--n_instances", type=int, default=3,
                    help="Synthetic pairs per reg_m point in section+reference mode (default: 3). "
                         "Ignored when --crops_h5ad is used.")
    ap.add_argument("--min_total_mass", type=float, default=0.05,
                    help="Minimum total transported mass for a plan to count as "
                         "numerically stable (default: 0.05).")
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Parallel threads for reg_m evaluation (default: 1). "
                         "Forced to 1 with a single GPU; with --device_ids 0 1 "
                         "n_jobs=2 routes each thread to a separate GPU.")
    ap.add_argument("--device_ids", type=int, nargs="+", default=None,
                    metavar="ID",
                    help="CUDA device indices to use (e.g. --device_ids 0 1 for two "
                         "GPUs). Defaults to [0] when use_gpu is True. Multi-GPU "
                         "requires n_jobs>1.")
    ap.add_argument("--use_gpu", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.use_gpu == "auto":
        use_gpu = gpu_available()
    else:
        use_gpu = args.use_gpu == "true"

    if args.crops_h5ad is not None:
        paths = args.crops_h5ad
        if len(paths) % 2 != 0:
            ap.error("--crops_h5ad requires an even number of paths (section ref pairs).")
        crops = [(sc.read_h5ad(paths[i]), sc.read_h5ad(paths[i + 1]))
                 for i in range(0, len(paths), 2)]
        section = reference = None
    else:
        if not args.section_h5ad or not args.reference_h5ad:
            ap.error("Provide either --crops_h5ad or both --section_h5ad and --reference_h5ad.")
        section = sc.read_h5ad(args.section_h5ad)
        reference = sc.read_h5ad(args.reference_h5ad)
        crops = None

    res = run_reg_m_sensitivity(
        section,
        reference,
        crops=crops,
        reg_m_grid=args.reg_m_grid,
        n_instances=args.n_instances,
        align_kwargs={"use_gpu": use_gpu},
        min_total_mass=args.min_total_mass,
        n_jobs=args.n_jobs,
        device_ids=args.device_ids,
        seed=args.seed,
        outdir=args.outdir,
    )

    default_row = next(
        (r for r in res["rows"] if abs(r["reg_m"] - res["default_reg_m"]) < 1e-9), None
    )
    if default_row:
        print(f"\nDefault reg_m={res['default_reg_m']:g}:  "
              f"LTA={default_row.get('lta', float('nan')):.4f}  "
              f"FOSCTTM={default_row.get('foscttm', float('nan')):.4f}  "
              f"expr_corr={default_row.get('expr_corr', float('nan')):.4f}")
    print(f"Results written to {args.outdir}")
