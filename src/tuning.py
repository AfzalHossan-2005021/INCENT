"""
tuning.py
=========
Ground-truth-anchored selection of the alignment weights
(``alpha``, ``alpha_cluster``, ``beta``, ``gamma``, ``delta``).

Three entry points:

* :func:`select_alignment_weights` -- the **development-time** selector, scored by
  **1 − FOSCTTM** (fully non-circular: FOSCTTM has no overlap with INCENT's objective
  components). All five weights are swept with the full (GPU-accelerated) alignment
  via either a staged ``"grid"`` or Optuna ``"bayesopt"`` (fewer evaluations).
  The simulator's joint PCA writes a shared ``X_pca`` into both the simulated slice
  and the retained ``reference``, so each (sim, ref) pair is already comparable.

* :func:`select_weights_real_pairs` -- uses **real section/reference pairs directly**,
  no synthetic perturbation. Scored by **LTA** (Label Transfer Accuracy): for each
  source cell the highest-weight target in the transport plan must share the same
  cell-type label. Requires ``cell_type_annot`` in both slices' ``.obs``; no
  ``sim.uns`` ground-truth provenance is needed.  Supports multi-GPU and ``n_jobs``.

* :func:`select_weights_unsupervised` -- the **deployment-time** selector for real
  slice pairs with no ground truth; staged grid scored by label-free **spatial
  coherence** (= GPR@k). The benchmark validates that this optimum tracks the
  development-time 1 − FOSCTTM optimum.

GPU: every alignment uses CUDA automatically when available (``use_gpu=gpu_available()``).
"""

from __future__ import annotations

import contextlib
import concurrent.futures
import os
from queue import Queue
from typing import Callable, Optional

import numpy as np

from .core import hierarchical_pairwise_align
from .perturb import simulate_adjacent_slice
from .evaluation import evaluate_alignment, geometric_preservation_rate


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
    """Silence the aligner's prints during a sweep.

    contextlib.redirect_stdout modifies the global sys.stdout and is NOT
    thread-safe: concurrent threads overwrite each other's saved reference,
    leaving sys.stdout pointing to a closed file after one thread exits.
    In worker threads we skip the redirect entirely — HOT prints in core.py
    are already gated by `if verbose:`, so nothing reaches stdout anyway.
    """
    if not enabled:
        yield
        return
    import threading
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull):
        yield


def simplex_grid(step: float = 0.25, offset: float = 0.0):
    """
    Grid the (gene, cell-type, neighborhood) feature simplex.

    Returns ``(beta, gamma)`` pairs with ``beta + gamma <= 1``.
    With ``offset=0.0`` (default) values start at 0 and include the corners
    (0,0), (1,0), (0,1).  With ``offset > 0`` values start at ``offset`` and
    stop before 1, so neither beta nor gamma equals exactly 0 or 1.
    For example ``simplex_grid(0.2, offset=0.1)`` yields 15 interior points
    with values in {0.1, 0.3, 0.5, 0.7, 0.9}.
    """
    n = int(round(1.0 / step))
    pts = []
    if offset == 0.0:
        for i in range(n + 1):
            for j in range(n + 1 - i):
                pts.append((round(i * step, 6), round(j * step, 6)))
    else:
        for i in range(n + 1):
            beta = round(offset + i * step, 6)
            if beta >= 1.0 - 1e-9:
                break
            for j in range(n + 1):
                gamma = round(offset + j * step, 6)
                if gamma >= 1.0 - 1e-9:
                    break
                if beta + gamma <= 1.0 + 1e-9:
                    pts.append((beta, gamma))
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

def _make_device_pool_score(score_fn: Callable, device_ids: list) -> Callable:
    """Wrap score_fn so each thread grabs a GPU from the pool via torch.cuda.device()."""
    try:
        import torch
    except ImportError:
        return score_fn
    if not torch.cuda.is_available():
        return score_fn
    pool: Queue = Queue()
    for did in device_ids:
        pool.put(did)

    def _wrapped(w):
        did = pool.get()
        try:
            with torch.cuda.device(did):
                return score_fn(w)
        finally:
            pool.put(did)

    return _wrapped


def _staged_search(
    score_fn: Callable[[dict], float],
    *,
    init: dict,
    alpha_grid,
    alpha_cluster_grid,
    simplex_step: float,
    simplex_offset: float = 0.0,
    delta_grid,
    refine: bool,
    n_jobs: int = 1,
    device_ids=None,
):
    best = dict(init)
    landscape = []

    # When multiple GPU devices are available, wrap score_fn so each thread
    # grabs a device from the pool before running; torch.cuda.device() is
    # thread-local so two threads can simultaneously use different GPUs.
    if device_ids and len(device_ids) > 1 and n_jobs > 1:
        _score = _make_device_pool_score(score_fn, device_ids)
    else:
        _score = score_fn

    def _eval_batch(combos: list, stage: str) -> list:
        """Evaluate a list of weight dicts, sequential or parallel."""
        if n_jobs == 1 or len(combos) <= 1:
            pairs = [(_score(w), w) for w in combos]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as pool:
                scores = list(pool.map(_score, combos))
            pairs = list(zip(scores, combos))
        for s, w in pairs:
            landscape.append({"stage": stage, "score": s, **w})
        return pairs

    def grid_alpha(cur):
        combos = [
            {**cur, "alpha": float(a), "alpha_cluster": float(ac)}
            for a in alpha_grid
            for ac in alpha_cluster_grid
        ]
        pairs = _eval_batch(combos, "alpha")
        loc = max(pairs, key=lambda p: p[0])
        return loc  # (best_score, best_weights)

    # Stage A: (alpha, alpha_cluster) at the initial feature weights
    _, wA = grid_alpha(best)
    best = wA

    # Stage B: feature simplex (beta, gamma) x delta at alpha*, alpha_cluster*
    combos_B = [
        {**best, "beta": float(beta), "gamma": float(gamma), "delta": float(d)}
        for (beta, gamma) in simplex_grid(simplex_step, offset=simplex_offset)
        for d in delta_grid
    ]
    pairs_B = _eval_batch(combos_B, "feature")
    best_score_B, best_B = max(pairs_B, key=lambda p: p[0])
    best, best_score = best_B, best_score_B

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
    is the ground-truth metric (neg_foscttm by default) averaged over instances."""
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
# development-time selector  (FOSCTTM; fully non-circular)
# ----------------------------------------------------------------------------

def select_alignment_weights(
    section=None,
    reference=None,
    *,
    crops=None,
    n_instances: int = 3,
    perturb_kwargs: Optional[dict] = None,
    max_cells: Optional[int] = None,
    objective_key: str = "neg_foscttm",
    method: str = "grid",
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
    alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    simplex_step: float = 0.25,
    simplex_offset: float = 0.0,
    n_trials: int = 40,
    init: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    refine: bool = True,
    quiet: bool = True,
    seed: int = 0,
    n_jobs: int = 1,
    device_ids=None,
) -> dict:
    """
    Select all five weights by maximizing a ground-truth metric over synthetic
    self-alignment instances scored with the full alignment (real OT).

    ``method``:
      * ``"grid"``     -- staged coordinate-ascent grid: first (alpha, alpha_cluster),
                          then the feature simplex (beta, gamma) x delta.
      * ``"bayesopt"`` -- Optuna TPE over the full continuous 5-weight cube
                          (fewer evaluations; needs ``optuna``).

    Instance source (priority ``crops`` > ``section``+``reference``).

    ``objective_key`` is any key from :func:`evaluation.evaluate_alignment`.
    Default ``'neg_foscttm'`` (= 1 − FOSCTTM, higher is better) is the recommended
    choice because:
      * **Fully non-circular** — FOSCTTM has zero overlap with INCENT's objective
        components (α expression, β cell-type, γ neighbourhood, α_cluster, δ).
        In contrast, GPR correlates with the γ neighbourhood term and LTA correlates
        with the β cell-type term; both could bias weight selection toward their
        respective objective component.
      * **Exact ground truth** — ``select_alignment_weights`` always creates synthetic
        instances via ``simulate_adjacent_slice``, so FOSCTTM correspondences are
        always available and ``neg_foscttm`` is never None here.
      * **Consistent with the reported metric** — the paper reports FOSCTTM on
        held-out instances; selecting by ``neg_foscttm`` on training instances is the
        most coherent methodology for reviewer scrutiny.
      ``'gpr'`` is a valid alternative when ground truth is uncertain or when you
      need a result that is also interpretable in the deployment (no-ground-truth) setting.

    GPU is used automatically when CUDA is available.

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
            simplex_offset=simplex_offset,
            delta_grid=delta_grid,
            refine=refine,
            n_jobs=n_jobs,
            device_ids=device_ids,
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
# real-pair selector  (LTA; cell-type labels; no perturbation)
# ----------------------------------------------------------------------------

def select_weights_real_pairs(
    pairs,
    *,
    objective_key="lta",
    method: str = "grid",
    alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
    simplex_step: float = 0.25,
    simplex_offset: float = 0.0,
    delta_grid=(0.25, 0.5, 0.75),
    n_trials: int = 40,
    init: Optional[dict] = None,
    align_kwargs: Optional[dict] = None,
    sim_axis: int = 0,
    refine: bool = True,
    quiet: bool = True,
    seed: int = 0,
    n_jobs: int = 1,
    device_ids=None,
) -> dict:
    """
    Select all five alignment weights using real section/reference pairs directly,
    without synthetic perturbation.

    Unlike :func:`select_alignment_weights`, no :func:`simulate_adjacent_slice` is
    called: each ``(sliceA, sliceB)`` tuple is aligned as-is, and the score is
    evaluated on the resulting transport plan.

    Because no exact ground-truth correspondences exist, ``objective_key`` defaults
    to ``"lta"`` (Label Transfer Accuracy): for each source cell the highest-weight
    target cell is found via argmax over the transport plan row, and label agreement
    is measured.  LTA requires only ``cell_type_annot`` (or the ``label_key`` kwarg)
    in both slices' ``.obs``; no ``sim.uns`` provenance is needed.

    Parameters
    ----------
    pairs : list of (AnnData, AnnData)
        Real ``(section, reference)`` pairs.  Each pair is aligned independently;
        the score averaged across all pairs.
    objective_key : str or callable
        Either a string key from :func:`evaluation.evaluate_alignment` or a
        callable ``(metrics_dict) -> float`` for combined objectives.
        String examples: ``"lta"`` (default), ``"gpr"``.
        Callable example: ``lambda m: 0.5 * (m["lta"] or 0) + 0.5 * (m["gpr"] or 0)``.
        ``"neg_foscttm"`` will be ``None`` for real pairs and must not be used here.
    method : str
        ``"grid"`` (default) or ``"bayesopt"``.
    alpha_grid, alpha_cluster_grid : tuple of float
        Grid values for Stage A and the optional Stage C refinement.
    simplex_step, simplex_offset : float
        Controls the (beta, gamma) simplex grid in Stage B.
    delta_grid : tuple of float
        Grid values for delta in Stage B.
    n_trials : int
        Number of Optuna trials (only used when ``method="bayesopt"``).
    init : dict or None
        Starting point for the staged search.  Defaults to :data:`DEFAULT_INIT`.
    align_kwargs : dict or None
        Extra kwargs forwarded to :func:`core.hierarchical_pairwise_align`.
    sim_axis : int
        Which slice acts as the LTA / GPR source.
        ``0`` = sliceA is the source (default; use when sliceA is the section).
        ``1`` = sliceB is the source.
    refine : bool
        Re-run the (alpha, alpha_cluster) grid after Stage B to fine-tune.
    quiet : bool
        Suppress alignment prints during the sweep.
    seed : int
        Random seed for the Bayesian sampler.
    n_jobs : int
        Thread-parallel weight evaluations.  Set ``>1`` together with
        ``len(device_ids) > 1`` to distribute across multiple GPUs.
    device_ids : list of int or None
        GPU indices to distribute across threads.

    Returns
    -------
    dict with keys:
        ``best``, ``best_score``, ``objective_key``, ``method``,
        ``landscape``, ``per_instance_at_best``, ``n_pairs``.
    """
    if method not in ("grid", "bayesopt"):
        raise ValueError("method must be 'grid' or 'bayesopt'.")
    pairs = list(pairs)
    if not pairs:
        raise ValueError("pairs must contain at least one (sliceA, sliceB) tuple.")

    init = dict(init or DEFAULT_INIT)
    align_kwargs = dict(align_kwargs or {})
    align_kwargs.setdefault("use_gpu", gpu_available())
    align_kwargs.setdefault("gpu_verbose", False)
    align_kwargs.setdefault("verbose", False)
    align_kwargs.setdefault("visualize_clusters", False)
    label_key = align_kwargs.get("label_key", "cell_type_annot")
    spatial_key = align_kwargs.get("spatial_key", "spatial")

    def score_fn(weights):
        vals = []
        for sliceA, sliceB in pairs:
            pi = _align_score(sliceA, sliceB, weights, align_kwargs, quiet)
            if pi is None:
                vals.append(0.0)
                continue
            mets = evaluate_alignment(
                pi, sliceA, sliceB,
                sim_axis=sim_axis,
                label_key=label_key,
                spatial_key=spatial_key,
            )
            if callable(objective_key):
                v = objective_key(mets)
            else:
                v = mets.get(objective_key, 0.0)
            vals.append(0.0 if v is None or not np.isfinite(v) else float(v))
        return float(np.mean(vals)) if vals else 0.0

    if method == "grid":
        best, best_score, landscape = _staged_search(
            score_fn, init=init,
            alpha_grid=alpha_grid,
            alpha_cluster_grid=alpha_cluster_grid,
            simplex_step=simplex_step,
            simplex_offset=simplex_offset,
            delta_grid=delta_grid,
            refine=refine,
            n_jobs=n_jobs,
            device_ids=device_ids,
        )
    else:
        best, best_score, landscape = _cell_optuna_search(
            score_fn, init=init,
            alpha_cluster_grid=alpha_cluster_grid,
            delta_grid=delta_grid,
            n_trials=n_trials, seed=seed,
        )

    per_instance = []
    for sliceA, sliceB in pairs:
        pi = _align_score(sliceA, sliceB, best, align_kwargs, quiet)
        if pi is not None:
            per_instance.append(
                evaluate_alignment(
                    pi, sliceA, sliceB,
                    sim_axis=sim_axis,
                    label_key=label_key,
                    spatial_key=spatial_key,
                )
            )

    return {
        "best": best,
        "best_score": best_score,
        "objective_key": "<callable>" if callable(objective_key) else objective_key,
        "method": method,
        "landscape": landscape,
        "per_instance_at_best": per_instance,
        "n_pairs": len(pairs),
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
        c = geometric_preservation_rate(pi, coordsA, coordsB, k_values=(k_coherence,))["gpr"]
        return 0.0 if c is None or not np.isfinite(c) else float(c)

    best, best_score, landscape = _staged_search(
        score_fn, init=init, alpha_grid=alpha_grid, alpha_cluster_grid=alpha_cluster_grid,
        simplex_step=simplex_step, delta_grid=delta_grid, refine=refine,
    )
    return {
        "best": best,
        "best_score": best_score,
        "objective_key": "gpr",
        "landscape": landscape,
    }
