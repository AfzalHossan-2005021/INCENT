"""
tuning.py
=========
Ground-truth-anchored selection of the alignment weights
(``alpha``, ``alpha_cluster``, ``beta``, ``gamma``, ``delta``).

Two entry points:

* :func:`select_alignment_weights` -- the **development-time** selector, scored by
  **registration accuracy** (exact synthetic ground truth; non-circular for every
  weight). All five weights are swept with the full (GPU-accelerated) alignment via
  either a staged ``"grid"`` or Optuna ``"bayesopt"`` (fewer evaluations).
  The simulator's joint PCA writes a shared ``X_pca`` into both the simulated slice
  and the retained ``reference``, so each (sim, ref) pair is already comparable.

* :func:`select_weights_unsupervised` -- the **deployment-time** selector for real
  slice pairs with no ground truth; staged grid scored by label-free **spatial
  coherence**. The benchmark validates that this optimum tracks the registration one.

GPU: every alignment uses CUDA automatically when available (``use_gpu=gpu_available()``).
"""

from __future__ import annotations

import contextlib
import os
from typing import Callable, Optional

import numpy as np

from .core import hierarchical_pairwise_align
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
# staged grid search (coordinate ascent over all 5 weights)
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
# Bayesian optimization (Optuna) over all 5 weights
# ----------------------------------------------------------------------------

def _cell_optuna_search(score_fn, init, *, alpha_cluster_grid, delta_grid, n_trials, seed):
    """Optuna TPE search over all five alignment weights. ``score_fn(weights)->float``
    is the (GPU-accelerated) registration objective averaged over instances."""
    try:
        import optuna
    except ImportError as e:
        raise ImportError(
            "method='bayesopt' requires optuna (`pip install optuna`). "
            "Use method='grid' for the dependency-free path."
        ) from e

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ac_lo, ac_hi = float(min(alpha_cluster_grid)), float(max(alpha_cluster_grid))
    d_lo, d_hi = float(min(delta_grid)), float(max(delta_grid))
    landscape = []

    def objective(trial):
        beta = trial.suggest_float("beta", 0.0, 1.0)
        gamma = trial.suggest_float("gamma", 0.0, 1.0 - beta)
        w = {
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "beta": beta,
            "gamma": gamma,
            "alpha_cluster": trial.suggest_float("alpha_cluster", ac_lo, ac_hi),
            "delta": trial.suggest_float("delta", d_lo, d_hi),
        }
        s = score_fn(w)
        landscape.append({"stage": "bayesopt", "score": s, **w})
        return s

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    bp = study.best_params
    best = {k: float(bp[k]) for k in ("alpha", "beta", "gamma", "alpha_cluster", "delta")}
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
    Select all five weights by maximizing mean registration accuracy over synthetic
    self-alignment instances scored with the full alignment (real OT).

    ``method``:
      * ``"grid"``     -- staged coordinate-ascent grid: first (alpha, alpha_cluster),
                          then the feature simplex (beta, gamma) x delta.
      * ``"bayesopt"`` -- Optuna TPE over the full continuous 5-weight cube
                          (fewer evaluations; needs ``optuna``).

    Instance source (priority ``crops`` > ``section``+``reference``). ``objective_key``
    is any key from :func:`evaluation.evaluate_alignment` (default the smooth
    registration objective). GPU is used automatically when CUDA is available.

    Returns ``{"best", "best_score", "objective_key", "method",
    "landscape", "per_instance_at_best", "n_instances"}``.
    """
    if method not in ("grid", "bayesopt"):
        raise ValueError("method must be 'grid' or 'bayesopt'.")

    init = dict(init or DEFAULT_INIT)
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    label_key = align_kwargs.get("label_key", "cell_type_annot")
    spatial_key = align_kwargs.get("spatial_key", "spatial")

    instances = make_self_alignment_instances(
        section=section, reference=reference, crops=crops,
        n_instances=n_instances, perturb_kwargs=perturb_kwargs,
        max_cells=max_cells, seed=seed,
    )

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

    if method == "grid":
        best, best_score, landscape = _staged_search(
            score_fn, init=init, alpha_grid=alpha_grid,
            alpha_cluster_grid=alpha_cluster_grid,
            simplex_step=simplex_step,
            delta_grid=delta_grid,
            refine=refine,
        )
    else:
        best, best_score, landscape = _cell_optuna_search(
            score_fn, init=init,
            alpha_cluster_grid=alpha_cluster_grid,
            delta_grid=delta_grid,
            n_trials=n_trials, seed=seed,
        )

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
