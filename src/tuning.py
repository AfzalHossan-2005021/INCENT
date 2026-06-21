"""
tuning.py
=========
Ground-truth-anchored selection of the alignment weights
(``alpha``, ``alpha_cluster``, ``beta``, ``gamma``, ``delta``).

Two entry points:

* :func:`select_alignment_weights` -- the **development-time** selector, scored by
  **registration accuracy** (exact synthetic ground truth; non-circular for every
  weight). Two-tier for efficiency:
    1. :func:`select_coarse_weights` picks ``(alpha_cluster, delta)`` on a cheap
       coarse-only objective (no cell-level OT -- seconds, not the ~10-min full
       alignment), since those two weights affect only the coarse FGW + macro-section.
    2. ``(alpha, beta, gamma)`` are then chosen with the full alignment via either a
       staged ``"grid"`` or Optuna ``"bayesopt"`` (fewer evaluations).
  The simulator's joint PCA writes a shared ``X_pca`` into both the simulated slice
  and the retained ``reference``, so each (sim, ref) pair is already comparable.

* :func:`select_weights_unsupervised` -- the **deployment-time** selector for real
  slice pairs with no ground truth; staged grid scored by label-free **spatial
  coherence**. The benchmark validates that this optimum tracks the registration one.

GPU: every cell-level alignment uses CUDA automatically when available
(``use_gpu=gpu_available()``); the coarse stage is tiny and runs on CPU.
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional

import numpy as np

from .core import hierarchical_pairwise_align, estimate_coarsen_length
from .clustering import cluster_cells_spatial
from .hierarchical import (
    build_slice_cluster_cache,
    compute_cluster_feature_costs,
    compute_cluster_structural_matrix,
    run_coarse_partial_fgw,
    extract_continuous_macro_section,
)
from .perturb import simulate_adjacent_slice
from .evaluation import evaluate_alignment, spatial_coherence


DEFAULT_INIT = {"alpha": 0.5, "beta": 0.5, "gamma": 0.25, "alpha_cluster": 0.5, "delta": 0.75}
WEIGHT_KEYS = ("alpha", "beta", "gamma", "alpha_cluster", "delta")


def gpu_available() -> bool:
    """True if a CUDA device is usable (so the FGW OT runs on GPU via the POT torch backend)."""
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------

@contextlib.contextmanager
def _quiet(enabled: bool):
    """Silence the aligner's prints during a sweep (it is very verbose)."""
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        yield


def simplex_grid(step: float = 0.25):
    """
    Grid the (gene, cell-type, neighborhood) feature simplex.

    Returns ``(beta, gamma)`` pairs with ``beta, gamma >= 0`` and
    ``beta + gamma <= 1`` (the remaining mass ``1 - beta - gamma`` is the
    gene-expression weight). Includes ``(0, 0)`` = pure expression.
    """
    n = int(round(1.0 / step))
    pts = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            pts.append((round(i * step, 6), round(j * step, 6)))
    return pts


def _subsample(adata, max_cells: Optional[int], rng):
    if max_cells is None or adata.n_obs <= max_cells:
        return adata
    idx = np.sort(rng.choice(adata.n_obs, size=max_cells, replace=False))
    return adata[idx].copy()


def make_self_alignment_instances(
    section=None,
    reference=None,
    *,
    crops=None,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    max_cells: Optional[int] = None,
    seed: int = 0,
):
    """
    Build ``(sim, reference)`` pairs with exact ground truth from manually-supplied
    crops.

    Two input modes (priority: ``crops`` > ``section``+``reference``):

    * ``section`` + ``reference``: perturb the given ``synthesize.py`` crop
      (``section``, parent-frame coords whose ``obs_names`` subset ``reference``)
      against the full slice. ``n_instances`` perturbation realizations are produced
      from this one crop (different perturbation seeds).
    * ``crops``: an explicit list of ``(section, reference)`` pairs; one instance
      per pair (``n_instances`` ignored).

    In both modes :func:`simulate_adjacent_slice` writes a **shared** ``X_pca`` into
    both ``sim`` and the (copied) reference, so each pair is in a comparable
    embedding. Instances are generated ONCE and reused across all weight candidates
    (weights do not change the PCA) -- the main speed-up of the sweep.
    """
    perturb_kwargs = dict(perturb_kwargs or {})
    rng = np.random.default_rng(seed)

    if crops is not None:
        sources = list(crops)
    elif section is not None and reference is not None:
        sources = [(section, reference)] * int(n_instances)
    else:
        raise ValueError("Provide either (`section` and `reference`) or `crops`.")

    instances = []
    for sec_src, ref_src in sources:
        ref = ref_src.copy()
        s = int(rng.integers(0, 2**31 - 1))
        with _quiet(True):
            sim = simulate_adjacent_slice(sec_src.copy(), reference=ref, seed=s, **perturb_kwargs)
        instances.append((_subsample(sim, max_cells, rng), _subsample(ref, max_cells, rng)))
    return instances


def _align_score(sliceA, sliceB, weights, align_kwargs, quiet):
    """Run the aligner once; return (pi or None)."""
    try:
        with _quiet(quiet):
            pi = hierarchical_pairwise_align(sliceA, sliceB, **weights, **align_kwargs)
        return np.asarray(pi, dtype=np.float64)
    except Exception:
        return None


# ----------------------------------------------------------------------------
# the shared staged search (coordinate ascent over the 5 weights)
# ----------------------------------------------------------------------------

def _staged_search(
    score_fn: Callable[[dict], float],
    *,
    init: dict,
    alpha_grid,
    alpha_cluster_grid,
    simplex_step: float,
    delta_grid,
    refine: bool,
):
    best = dict(init)
    landscape = []

    def grid_alpha(cur):
        loc = None
        for a in alpha_grid:
            for ac in alpha_cluster_grid:
                w = {**cur, "alpha": float(a), "alpha_cluster": float(ac)}
                s = score_fn(w)
                landscape.append({"stage": "alpha", "score": s, **w})
                if loc is None or s > loc[0]:
                    loc = (s, w)
        return loc

    # Stage A: (alpha, alpha_cluster) at the initial feature weights
    sA, wA = grid_alpha(best)
    best = wA

    # Stage B: feature simplex (beta, gamma) x delta at alpha*, alpha_cluster*
    locB = None
    for (beta, gamma) in simplex_grid(simplex_step):
        for d in delta_grid:
            w = {**best, "beta": float(beta), "gamma": float(gamma), "delta": float(d)}
            s = score_fn(w)
            landscape.append({"stage": "feature", "score": s, **w})
            if locB is None or s > locB[0]:
                locB = (s, w)
    best, best_score = locB[1], locB[0]

    # Stage C (optional): refine (alpha, alpha_cluster) at the chosen feature weights
    if refine:
        sC, wC = grid_alpha(best)
        if sC >= best_score:
            best, best_score = wC, sC

    return best, best_score, landscape


# ----------------------------------------------------------------------------
# stage decoupling: tune the COARSE-stage weights (alpha_cluster, delta) on a
# cheap coarse-only objective -- no cell-level OT, so it costs seconds, not the
# ~10-min full alignment. alpha_cluster and delta affect ONLY the coarse FGW and
# macro-section (Steps 1-5 of hierarchical_pairwise_align), never Steps 6-8.
# ----------------------------------------------------------------------------

def _f1(pred_mask, true_mask):
    pred_mask = np.asarray(pred_mask, dtype=bool)
    true_mask = np.asarray(true_mask, dtype=bool)
    tp = float(np.sum(pred_mask & true_mask))
    p = float(pred_mask.sum())
    t = float(true_mask.sum())
    if p == 0 or t == 0:
        return 0.0
    prec, rec = tp / p, tp / t
    return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)


def _coarse_precompute(sim, ref, *, spatial_key, use_rep, label_key, coarsen_scale=None):
    """Per-instance quantities that do NOT depend on (alpha_cluster, delta): the
    clustering, per-cluster caches, and centroid structural matrices. Computed once
    and reused across all coarse-weight candidates (the coarse-search speed-up)."""
    S = float(coarsen_scale) if coarsen_scale is not None else estimate_coarsen_length(
        sim, ref, spatial_key=spatial_key)[0]
    labelsA = cluster_cells_spatial(sim, spatial_key=spatial_key, coarsen_length=S)
    labelsB = cluster_cells_spatial(ref, spatial_key=spatial_key, coarsen_length=S)
    all_types = np.array(sorted(
        set(sim.obs[label_key].astype(str)) | set(ref.obs[label_key].astype(str))), dtype=str)
    cache_A = build_slice_cluster_cache(sim, labelsA, spatial_key=spatial_key,
                                        feature_key=use_rep, label_key=label_key, all_types=all_types)
    cache_B = build_slice_cluster_cache(ref, labelsB, spatial_key=spatial_key,
                                        feature_key=use_rep, label_key=label_key, all_types=all_types)
    return {
        "labelsA": labelsA, "labelsB": labelsB,
        "cache_A": cache_A, "cache_B": cache_B,
        "C_A": compute_cluster_structural_matrix(cache_A.centroids),
        "C_B": compute_cluster_structural_matrix(cache_B.centroids),
    }


def _coarse_overlap_f1(sim, ref, pc, alpha_cluster, delta, *, spatial_key, label_key):
    """Score how well the coarse stage + macro-section localizes the TRUE overlap,
    for given (alpha_cluster, delta). Runs only the cheap coarse FGW + macro-section
    (no cell OT) and returns the mean of the A- and B-side overlap F1 vs the exact
    correspondence ground truth."""
    M_cluster = compute_cluster_feature_costs(
        pc["cache_A"].mu_expr, pc["cache_A"].mu_struct,
        pc["cache_B"].mu_expr, pc["cache_B"].mu_struct, delta=delta)
    Pi_cluster = run_coarse_partial_fgw(
        M_cluster, pc["C_A"], pc["C_B"], pc["cache_A"].masses, pc["cache_B"].masses,
        alpha=alpha_cluster)
    try:
        macro = extract_continuous_macro_section(
            sim, ref, pc["labelsA"], pc["labelsB"], Pi_cluster,
            spatial_key=spatial_key, label_key=label_key,
            cluster_cache_A=pc["cache_A"], cluster_cache_B=pc["cache_B"])
    except Exception:
        return 0.0
    if not macro.ok:
        return 0.0

    is_birth = (np.asarray(sim.obs["adjacent_is_birth"], dtype=bool)
                if "adjacent_is_birth" in sim.obs.columns else np.zeros(sim.n_obs, dtype=bool))
    has_partner = np.isin(np.asarray(sim.obs_names), np.asarray(ref.obs_names))
    matched_A = has_partner & ~is_birth                       # true overlap on the A (sim) side
    matched_names = set(np.asarray(sim.obs_names)[matched_A].tolist())
    true_B = np.isin(np.asarray(ref.obs_names), list(matched_names))  # true overlap on the B (ref) side

    pred_A = np.zeros(sim.n_obs, dtype=bool); pred_A[macro.idx_A] = True
    pred_B = np.zeros(ref.n_obs, dtype=bool); pred_B[macro.idx_B] = True
    return 0.5 * (_f1(pred_A, matched_A) + _f1(pred_B, true_B))


def select_coarse_weights(
    instances,
    *,
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    spatial_key="spatial",
    use_rep="X_pca",
    label_key="cell_type_annot",
    coarsen_scale=None,
):
    """Grid-select (alpha_cluster, delta) by the coarse overlap-F1 objective. Cheap:
    no cell-level OT. Reuses a per-instance precompute so only the small coarse FGW
    + macro-section run per candidate."""
    precomps = [
        _coarse_precompute(sim, ref, spatial_key=spatial_key, use_rep=use_rep,
                           label_key=label_key, coarsen_scale=coarsen_scale)
        for sim, ref in instances
    ]
    landscape, best = [], None
    for ac in alpha_cluster_grid:
        for d in delta_grid:
            scores = [
                _coarse_overlap_f1(sim, ref, pc, float(ac), float(d),
                                   spatial_key=spatial_key, label_key=label_key)
                for (sim, ref), pc in zip(instances, precomps)
            ]
            s = float(np.mean(scores)) if scores else 0.0
            landscape.append({"alpha_cluster": float(ac), "delta": float(d), "score": s})
            if best is None or s > best[0]:
                best = (s, {"alpha_cluster": float(ac), "delta": float(d)})
    return {"best": best[1], "best_score": best[0], "landscape": landscape}


# ----------------------------------------------------------------------------
# cell-level Bayesian optimization (Optuna) over (alpha, beta, gamma)
# ----------------------------------------------------------------------------

def _cell_optuna_search(score_fn, init, *, n_trials, seed):
    """Optuna TPE search over the cell-level weights (alpha, beta, gamma) with
    (alpha_cluster, delta) fixed at ``init``. Far fewer evaluations than a grid for
    the same coverage of the continuous space. ``score_fn(weights)->float`` is the
    (GPU-accelerated) registration objective averaged over instances."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "method='bayesopt' requires optuna (`pip install optuna`). "
            "Use method='grid' for the dependency-free path."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    fixed = {"alpha_cluster": init["alpha_cluster"], "delta": init["delta"]}
    landscape = []

    def objective(trial):
        beta = trial.suggest_float("beta", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0 - beta)   # enforce beta+gamma<=1
        w = {"alpha": trial.suggest_float("alpha", 0.0, 1.0),
             "beta": beta, "gamma": gamma, **fixed}
        s = score_fn(w)
        landscape.append({"stage": "bayesopt", "score": s, **w})
        return s

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    best = {"alpha": float(bp["alpha"]), "beta": float(bp["beta"]),
            "gamma": float(bp["gamma"]), **fixed}
    return best, float(study.best_value), landscape


# ----------------------------------------------------------------------------
# development-time selector  (registration accuracy; non-circular)
# ----------------------------------------------------------------------------

def select_alignment_weights(
    section=None,
    reference=None,
    *,
    crops=None,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    max_cells: Optional[int] = None,
    objective_key: str = "reg_soft_corr_mass",
    method: str = "grid",
    tune_coarse: bool = True,
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    simplex_step: float = 0.25,
    n_trials: int = 40,
    init: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    refine: bool = True,
    quiet: bool = True,
    seed: int = 0,
) -> dict:
    """
    Select the five weights by maximizing mean registration accuracy over synthetic
    self-alignment instances.

    Two-tier, cost-aware design:
      1. **Coarse stage** (``tune_coarse=True``): pick ``(alpha_cluster, delta)`` with
         :func:`select_coarse_weights` -- a cheap coarse-only objective (no cell OT).
      2. **Cell stage**: pick ``(alpha, beta, gamma)`` with the full (GPU-accelerated)
         alignment, holding ``(alpha_cluster, delta)`` fixed. ``method``:
           * ``"grid"``   -- staged grid (alpha line, then the feature simplex).
           * ``"bayesopt"`` -- Optuna TPE over the continuous cube (fewer evals;
             needs ``optuna``).

    Instance source (priority ``crops`` > ``section``+``reference``). ``objective_key``
    is any key from :func:`evaluation.evaluate_alignment` (default the smooth
    registration objective). GPU is used automatically for every cell-level alignment
    when CUDA is available; the coarse stage is tiny and runs on CPU.

    Returns ``{"best", "best_score", "objective_key", "method", "coarse",
    "landscape", "per_instance_at_best", "n_instances"}``.
    """
    if method not in ("grid", "bayesopt"):
        raise ValueError("method must be 'grid' or 'bayesopt'.")

    init = dict(init or DEFAULT_INIT)
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())     # GPU FGW when available
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    label_key = align_kwargs.get("label_key", "cell_type_annot")
    spatial_key = align_kwargs.get("spatial_key", "spatial")
    use_rep = align_kwargs.get("use_rep", "X_pca")

    instances = make_self_alignment_instances(
        section=section, reference=reference, crops=crops,
        n_instances=n_instances, perturb_kwargs=perturb_kwargs,
        max_cells=max_cells, seed=seed,
    )

    # 1. coarse-stage weights (cheap; CPU)
    coarse_info = None
    if tune_coarse:
        coarse_info = select_coarse_weights(
            instances, alpha_cluster_grid=alpha_cluster_grid, delta_grid=delta_grid,
            spatial_key=spatial_key, use_rep=use_rep, label_key=label_key,
            coarsen_scale=align_kwargs.get("coarsen_scale"),
        )
        init["alpha_cluster"] = coarse_info["best"]["alpha_cluster"]
        init["delta"] = coarse_info["best"]["delta"]

    # registration objective over instances (one GPU-accelerated alignment per instance)
    def score_fn(weights):
        vals = []
        for sim, ref in instances:
            pi = _align_score(sim, ref, weights, align_kwargs, quiet)
            if pi is None:
                vals.append(0.0)
                continue
            mets = evaluate_alignment(
                pi, sim, ref, sim_axis=0, label_key=label_key, spatial_key=spatial_key,
            )
            v = mets.get(objective_key, 0.0)
            vals.append(0.0 if v is None or not np.isfinite(v) else float(v))
        return float(np.mean(vals)) if vals else 0.0

    # 2. cell-stage weights (expensive; GPU). alpha_cluster & delta held fixed.
    if method == "grid":
        best, best_score, landscape = _staged_search(
            score_fn, init=init, alpha_grid=alpha_grid,
            alpha_cluster_grid=(init["alpha_cluster"],),   # fixed from coarse stage
            simplex_step=simplex_step,
            delta_grid=(init["delta"],),                   # fixed from coarse stage
            refine=refine,
        )
    else:
        best, best_score, landscape = _cell_optuna_search(
            score_fn, init=init, n_trials=n_trials, seed=seed)

    # full metric battery at the chosen weights (for reporting)
    per_instance = []
    for sim, ref in instances:
        pi = _align_score(sim, ref, best, align_kwargs, quiet)
        if pi is not None:
            per_instance.append(
                evaluate_alignment(pi, sim, ref, sim_axis=0,
                                   label_key=label_key, spatial_key=spatial_key)
            )

    return {
        "best": best,
        "best_score": best_score,
        "objective_key": objective_key,
        "method": method,
        "coarse": coarse_info,
        "landscape": landscape,
        "per_instance_at_best": per_instance,
        "n_instances": len(instances),
    }


# ----------------------------------------------------------------------------
# deployment-time selector  (label-free spatial coherence; no ground truth)
# ----------------------------------------------------------------------------

def select_weights_unsupervised(
    sliceA,
    sliceB,
    *,
    alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    simplex_step: float = 0.25,
    delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    init: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    k_coherence: int = 15,
    refine: bool = True,
    quiet: bool = True,
) -> dict:
    """
    Select weights for a real (sliceA, sliceB) pair WITHOUT ground truth, by
    maximizing the label-free spatial coherence of the mapping. Same staged search
    as :func:`select_alignment_weights`. (The benchmark validates that this
    optimum agrees with the registration-optimal weights.)
    """
    init = dict(init or DEFAULT_INIT)
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    spatial_key = align_kwargs.get("spatial_key", "spatial")

    coordsA = np.asarray(sliceA.obsm[spatial_key], dtype=np.float64)[:, :2]
    coordsB = np.asarray(sliceB.obsm[spatial_key], dtype=np.float64)[:, :2]

    def score_fn(weights):
        pi = _align_score(sliceA, sliceB, weights, align_kwargs, quiet)
        if pi is None:
            return 0.0
        c = spatial_coherence(pi, coordsA, coordsB, k=k_coherence)["coherence"]
        return 0.0 if c is None or not np.isfinite(c) else float(c)

    best, best_score, landscape = _staged_search(
        score_fn, init=init, alpha_grid=alpha_grid, alpha_cluster_grid=alpha_cluster_grid,
        simplex_step=simplex_step, delta_grid=delta_grid, refine=refine,
    )
    return {
        "best": best,
        "best_score": best_score,
        "objective_key": "coherence",
        "landscape": landscape,
    }
