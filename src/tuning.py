"""
tuning.py
=========
Ground-truth-anchored selection of the alignment weights
(``alpha``, ``alpha_cluster``, ``beta``, ``gamma``, ``delta``).

Two entry points:

* :func:`select_alignment_weights` -- the **development-time** selector. It builds
  synthetic self-alignment instances with :func:`perturb.simulate_adjacent_slice`
  (exact ground truth), aligns each under candidate weights, and scores by
  **registration accuracy** (:mod:`evaluation`). Because registration measures
  geometric correspondence it is *non-circular for every weight*, so the staged
  grid (coordinate ascent) is purely a compute optimization, not a leak-free
  device. The simulator's joint PCA writes a shared ``X_pca`` into both the
  simulated slice and the retained ``reference``, so each (sim, ref) pair is
  already in a comparable embedding -- no extra PCA step is needed.

* :func:`select_weights_unsupervised` -- the **deployment-time** selector for real
  slice pairs with no ground truth. Same staged search, but scored by the
  label-free **spatial coherence** of the mapping. The benchmark (step 4) is what
  validates that this label-free optimum tracks the registration optimum.

Both share :func:`_staged_search`: grid ``(alpha, alpha_cluster)`` -> grid the
feature simplex ``(beta, gamma)`` with ``delta`` -> optional refinement of
``(alpha, alpha_cluster)`` at the chosen feature weights.
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
    base,
    *,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    crop_fn: Optional[Callable] = None,
    max_cells: Optional[int] = None,
    seed: int = 0,
):
    """
    Build ``n_instances`` (sim, reference) pairs with exact ground truth.

    For each instance: ``reference`` is a clean copy of ``base`` (or a crop), the
    section to perturb is ``crop_fn(base, rng)`` (or ``base``), and
    :func:`simulate_adjacent_slice` produces ``sim`` while writing a **shared**
    ``X_pca`` into both ``sim`` and ``reference``. Generated ONCE and reused across
    all weight candidates (weights do not change the PCA), which is the main
    speed-up of the sweep.
    """
    perturb_kwargs = dict(perturb_kwargs or {})
    rng = np.random.default_rng(seed)
    instances = []
    for _ in range(n_instances):
        ref = base.copy()
        section = crop_fn(base, rng) if crop_fn is not None else base.copy()
        s = int(rng.integers(0, 2**31 - 1))
        with _quiet(True):
            sim = simulate_adjacent_slice(section, reference=ref, seed=s, **perturb_kwargs)
        sim = _subsample(sim, max_cells, rng)
        ref = _subsample(ref, max_cells, rng)
        instances.append((sim, ref))
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
# development-time selector  (registration accuracy; non-circular)
# ----------------------------------------------------------------------------

def select_alignment_weights(
    base,
    *,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    crop_fn: Optional[Callable] = None,
    max_cells: Optional[int] = None,
    objective_key: str = "reg_soft_corr_mass",
    alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    simplex_step: float = 0.25,
    delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    init: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    refine: bool = True,
    quiet: bool = True,
    seed: int = 0,
) -> dict:
    """
    Select the five weights by maximizing mean registration accuracy over synthetic
    self-alignment instances.

    ``objective_key`` is any key returned by :func:`evaluation.evaluate_alignment`
    (default ``reg_soft_corr_mass``, the smooth registration objective; higher is
    better). Each evaluation is one full alignment per instance, so keep grids /
    ``n_instances`` / ``max_cells`` modest; refine later.

    Returns ``{"best": {weights}, "best_score": float, "landscape": [...],
    "per_instance_at_best": [metrics...]}``.
    """
    init = dict(init or DEFAULT_INIT)
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    label_key = align_kwargs.get("label_key", "cell_type_annot")
    spatial_key = align_kwargs.get("spatial_key", "spatial")

    instances = make_self_alignment_instances(
        base, n_instances=n_instances, perturb_kwargs=perturb_kwargs,
        crop_fn=crop_fn, max_cells=max_cells, seed=seed,
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

    best, best_score, landscape = _staged_search(
        score_fn, init=init, alpha_grid=alpha_grid, alpha_cluster_grid=alpha_cluster_grid,
        simplex_step=simplex_step, delta_grid=delta_grid, refine=refine,
    )

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
