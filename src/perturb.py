"""
adjacent_slice.py
===================
Simulate an *adjacent serial section* from a cut section of a MERFISH slice.

This is a **cell-preserving proxy**: it takes a section (a crop of a parent
slice), drops some cells, geometrically displaces the survivors the way a
~10 um-offset serial section would differ, and adds molecular noise -- while
retaining a known 1:1 correspondence to the parent (gold ground truth for
scoring).

Expression consistency
-----------------------
``.X`` is the single source of truth for expression: the Gaussian noise is
added to it and it is overwritten with the noised result. Any ``obsm["X_pca"]``
inherited from ``section`` is dropped in the process, since it would otherwise
be a stale embedding of the pre-noise expression. The original (pre-noise)
expression is preserved in ``layers["X_unperturbed"]`` for provenance.

The INCENT pipeline also consumes ``obsm["X_pca"]`` (``hierarchical.py``
defaults to ``feature_key="X_pca"``); once ``.X`` is final, call
``utils.compute_joint_pca`` on the returned slice and ``reference`` to write a
consistent shared embedding into both -- this is what
``tuning.make_self_alignment_instances`` does.

Pipeline (in order)
-------------------
    1. Random cell dropout              Bernoulli(keep) over cells.
    2. Smooth non-rigid warp            thin-plate-spline displacement field.
    3. Per-cell jitter                  N(0, sigma^2 I) (offset cutting plane).
    4. Collision resolution             iterative push-apart so NO two cells are
                                        closer than d_min  ->  no cell touches.
    5. Rigid transform                  rotation (+ optional reflection) + shift
                                        (distance-preserving -> no-touch holds).
    6. Cell-type- & gene-aware noise    Gaussian noise added to .X, scaled per
                                        gene *within each cell type*
                                        (sigma_{i,g} = alpha * std of gene g
                                        among cells of i's cell type).
    7. Annotation (label) noise         a fraction of cell-type labels reassigned
                                        to a different type (annotation error);
                                        the TRUE labels used by step 6 are kept
                                        in obs[celltype_key + "_clean"].
    8. Spurious (birth) cells           optional unmatched cells appended as
                                        realistic clutter; they have no parent
                                        (spatial_unperturbed = NaN) and are
                                        flagged obs["adjacent_is_birth"].

Rotation convention matches ``core.py`` / ``synthesize.py``:
    R(theta) = [[cos, -sin], [sin, cos]];  forward map on rows: X @ A.T + b.

Ground truth written onto the returned slice
--------------------------------------------
    obsm["spatial_unperturbed"]             (N,2) parent-frame coords (gold target;
                                            NaN for birth cells, which have no parent)
    obsm["adjacent_displacement_prerigid"]  (N,2) net warp+jitter+collision shift
    layers["X_unperturbed"]                 pre-noise expression
    obs ["adjacent_kept"]                    provenance flag
    obs [celltype_key + "_clean"]            true labels before annotation noise
    obs ["adjacent_label_flipped"]           which labels were corrupted (step 7)
    obs ["adjacent_is_birth"]                spurious unmatched cells (step 8); exclude
                                            these from correspondence scoring
    uns["self_alignment_test"]["adjacent_simulation"]  full provenance, including
        "dropout_kept_positions" (N_matched,): dropout_kept_positions[j] is the
        ROW INDEX INTO `reference` of the true parent of matched sim cell j
        (mapped via obs_names -- NOT a position local to `section`). This is
        what evaluate_alignment / foscttm auto-extract as FOSCTTM ground truth,
        so `reference` must contain every surviving cell of `section` under a
        matching obs_name.

Note (state in the paper): matched cells are known parent cells (a z-displaced
same-cell model). Weaker / more realistic correspondence is obtained by raising
``dropout_rate`` (deletions), ``birth_rate`` (spurious unmatched cells), and
``label_flip_rate`` (annotation error); these break the strict 1:1 mapping while
the matched subset retains exact ground truth.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
import anndata as ad
from anndata import AnnData
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .utils import compute_joint_pca


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------

def _to_dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)


def _sanitize(M: np.ndarray, name: str) -> np.ndarray:
    """Replace non-finite entries (NaN/inf) with 0, warning how many were found."""
    M = np.asarray(M, dtype=np.float64)
    bad = ~np.isfinite(M)
    nbad = int(bad.sum())
    if nbad:
        warnings.warn(
            f"{name} contained {nbad} non-finite value(s) (NaN/inf); replaced with 0. "
            f"Check upstream preprocessing.", UserWarning, stacklevel=3)
        M = np.where(bad, 0.0, M)
    return M


def _has_negatives(M: np.ndarray, tol: float = 1e-6) -> bool:
    finite = M[np.isfinite(M)]
    return bool(finite.size and finite.min() < -tol)


def _rotation_matrix(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _linear_part(angle_rad: float, reflect: bool) -> np.ndarray:
    A = _rotation_matrix(angle_rad)
    if reflect:
        A = A @ np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    return A


def invert_rigid(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert ``X' = X @ A.T + b`` -> (A_inv, b_inv) with ``X = X' @ A_inv.T + b_inv``."""
    A_inv = np.linalg.inv(A)
    return A_inv, -b @ A_inv.T


def _nn_distances(coords: np.ndarray) -> np.ndarray:
    """Return the distance to each point's nearest neighbor (excluding self, K=2)."""
    tree = cKDTree(coords)
    d, _ = tree.query(coords, k=2)
    return d[:, 1]


# ----------------------------------------------------------------------------
# Geometry primitives
# ----------------------------------------------------------------------------

def _smooth_warp(coords, amplitude, n_control, rng):
    """Thin-plate-spline warp from a coarse grid of random control displacements."""
    mins, maxs = coords.min(0), coords.max(0)
    pad = 0.05 * np.maximum(maxs - mins, 1e-9)
    gx = np.linspace(mins[0] - pad[0], maxs[0] + pad[0], n_control)
    gy = np.linspace(mins[1] - pad[1], maxs[1] + pad[1], n_control)
    GX, GY = np.meshgrid(gx, gy)
    ctrl = np.column_stack([GX.ravel(), GY.ravel()])
    disp_ctrl = rng.normal(0.0, amplitude, size=ctrl.shape)
    rbf = RBFInterpolator(ctrl, disp_ctrl, kernel="thin_plate_spline", smoothing=0.0)
    return coords + rbf(coords), ctrl, disp_ctrl


def _resolve_collisions(coords, d_min, max_iter, rng):
    """Push apart any pair closer than ``d_min`` (Jacobi soft repulsion)."""
    coords = coords.copy()
    iters = 0
    for iters in range(1, max_iter + 1):
        tree = cKDTree(coords)
        pairs = tree.query_pairs(r=d_min, output_type="ndarray")
        if pairs.shape[0] == 0:
            break
        i, j = pairs[:, 0], pairs[:, 1]
        delta = coords[j] - coords[i]
        dist = np.linalg.norm(delta, axis=1)
        zero = dist < 1e-12
        if zero.any():
            ang = rng.uniform(0, 2 * np.pi, size=int(zero.sum()))
            delta[zero] = np.column_stack([np.cos(ang), np.sin(ang)]) * 1e-6
            dist[zero] = 1e-6
        unit = delta / dist[:, None]
        push = 0.5 * (d_min - dist)[:, None] * unit * 1.05
        disp = np.zeros_like(coords)
        np.add.at(disp, i, -push)
        np.add.at(disp, j, push)
        coords = coords + disp
    final_min = float(_nn_distances(coords).min()) if len(coords) > 1 else np.inf
    return coords, iters, final_min


def _resolve_pivot(pivot, coords, meta):
    """Resolve the pivot point for the rigid transform."""
    if isinstance(pivot, str):
        if pivot == "centroid":
            return coords.mean(0)
        if pivot == "window_center":
            wc = meta.get("window_center")
            return coords.mean(0) if wc is None else np.asarray(wc, float)[:2]
        raise ValueError("pivot must be 'centroid', 'window_center', or (x, y).")
    p = np.asarray(pivot, float).ravel()
    if p.shape != (2,):
        raise ValueError(f"pivot coordinate must have length 2; got {p.shape}.")
    return p


# ----------------------------------------------------------------------------
# Expression noise: cell-type- and gene-aware (operates in .X's native space)
# ----------------------------------------------------------------------------

def _celltype_gene_aware_noise(E, cell_types, alpha, min_group_cells, rng, nonneg):
    """
    Add Gaussian noise to expression ``E`` (cells x genes) in ITS OWN units, with
    per-gene std measured *within each cell type*:

        sigma_{i,g} = alpha * std_g( cells sharing i's cell type )

    Std is measured on the clean ``E``.  Cell types smaller than
    ``min_group_cells`` (or absent annotation) fall back to the global per-gene
    std.  If ``nonneg`` the result is clipped at 0.
    """
    E = _sanitize(E, "expression matrix (noise)")
    n, g = E.shape
    global_std = E.std(axis=0, ddof=1) if n > 1 else np.zeros(g)
    sigma = np.empty((n, g), dtype=np.float64)
    info = {"groups": {}, "fallback_global_for": []}

    if cell_types is None:
        sigma[:] = alpha * global_std
        info["fallback_global_for"].append("<all: no celltype key>")
    else:
        for ct in np.unique(cell_types):
            idx = np.flatnonzero(cell_types == ct)
            if idx.size >= max(min_group_cells, 2):
                std_g = E[idx].std(axis=0, ddof=1)
            else:
                std_g = global_std
                info["fallback_global_for"].append(str(ct))
            sigma[idx] = alpha * std_g
            info["groups"][str(ct)] = int(idx.size)

    sigma = np.nan_to_num(sigma, nan=0.0)
    E_noised = E + rng.standard_normal(size=E.shape) * sigma
    if nonneg:
        E_noised = np.clip(E_noised, 0.0, None)
    info["alpha"] = float(alpha)
    info["mean_sigma"] = float(sigma.mean())
    info["nonneg_clip"] = bool(nonneg)
    return E_noised, info


# ----------------------------------------------------------------------------
# Annotation noise & spurious (birth) cells  --  ground-truth-preserving
# ----------------------------------------------------------------------------

def _apply_label_noise(labels, rng, flip_rate):
    """
    Annotation-error model: reassign a fraction of cell-type labels to a
    *different* type, uniformly at random.

    This corrupts only the annotation the aligner reads; the upstream expression
    noise is computed from the TRUE cell types, so label noise is an independent
    nuisance axis that lets the cell-type cost weight (beta) be tested against
    realistic mislabelling rather than a perfect oracle. Returns ``(labels,
    flipped_mask)`` and is a no-op (empty flip) when fewer than two types exist.
    """
    labels = np.asarray(labels, dtype=object).copy()
    flipped = np.zeros(labels.shape[0], dtype=bool)
    if flip_rate <= 0.0:
        return labels, flipped
    types = np.unique(labels)
    if types.size < 2:
        return labels, flipped
    n_flip = int(round(flip_rate * labels.shape[0]))
    if n_flip <= 0:
        return labels, flipped
    idx = rng.choice(labels.shape[0], size=n_flip, replace=False)
    for i in idx:
        others = types[types != labels[i]]
        labels[i] = others[rng.integers(others.size)]
    flipped[idx] = True
    return labels, flipped


def _make_spurious_cells(sim, rng, n_birth, spatial_key, median_nn, offset_scale, seed_tag):
    """
    Build ``n_birth`` unmatched 'birth' cells (GT-safe independent-resampling proxy).

    Each birth copies a random simulated cell (so its expression and label stay
    valid) and is scattered locally by a small Gaussian offset. Births carry
    **no parent**: ``spatial_unperturbed`` is NaN
    and ``obs['adjacent_is_birth']`` is True, so a scorer drops them from the
    correspondence metrics while they still act as realistic clutter that the
    aligner must cope with. Returns an ``AnnData`` (``n_birth`` rows) or ``None``.
    """
    if n_birth <= 0:
        return None
    src = rng.integers(0, sim.n_obs, size=n_birth)
    births = sim[src].copy()
    births.obs_names = [f"spurious_{seed_tag}_{i}" for i in range(n_birth)]

    coords = np.asarray(births.obsm[spatial_key], dtype=np.float64).copy()
    coords[:, :2] += rng.normal(0.0, max(offset_scale * median_nn, 1e-9), size=(n_birth, 2))
    births.obsm[spatial_key] = coords

    births.obsm["spatial_unperturbed"] = np.full((n_birth, 2), np.nan, dtype=np.float64)
    births.obsm["adjacent_displacement_prerigid"] = np.zeros((n_birth, 2), dtype=np.float64)
    births.obs["adjacent_kept"] = np.zeros(n_birth, dtype=bool)
    births.obs["adjacent_is_birth"] = np.ones(n_birth, dtype=bool)
    births.obs["adjacent_label_flipped"] = np.zeros(n_birth, dtype=bool)
    return births


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def simulate_adjacent_slice(
    section: AnnData,
    reference: Optional[AnnData] = None,
    *,
    spatial_key: str = "spatial",
    celltype_key: str = "cell_type_annot",
    # --- 1. dropout ---
    dropout_rate: float = 0.10,
    # --- 2. warp ---
    warp_amplitude: Optional[float] = None,     # um;  None -> 0.025 * bbox diagonal
    warp_n_control: int = 5,                    # control point grid size (n_control x n_control)
    # --- 3. jitter ---
    jitter_sigma: Optional[float] = None,       # um;  None -> 0.25 * median NN dist
    jitter_fraction: float = 1.0,
    # --- 4. no-touch ---
    min_dist: Optional[float] = None,
    min_dist_quantile: float = 0.05,
    hardcore_diameter: Optional[float] = None,
    collision_max_iter: int = 100000,
    # --- 5. rigid ---
    rotation_deg: Optional[float] = None,
    rotation_range: Tuple[float, float] = (-180.0, 180.0),
    translation: Optional[Sequence[float]] = None,
    translation_scale: float = 0.5,
    reflect: bool = False,
    pivot: Union[str, Sequence[float]] = "centroid",
    # --- 6. expression noise (added to .X in its native units) ---
    expr_alpha: float = 1.0,              # noise scale relative to gene std within cell type
    min_group_cells: int = 20,
    expression_layer: Optional[str] = None,      # source expression; None -> .X
    nonneg_clip: Union[bool, str] = "auto",       # auto: clip iff data is non-negative
    # --- 7. annotation noise & spurious (birth) cells (default off) ---
    label_flip_rate: float = 0.10,        # fraction of labels reassigned to a different type
    birth_rate: float = 0.10,             # spurious unmatched cells, as a fraction of survivors
    birth_offset_scale: float = 2.0,      # birth scatter, in units of median NN distance
    # --- bookkeeping ---
    min_cells: int = 4,
    seed: Optional[int] = None,
) -> AnnData:
    """
    Turn a cut ``section`` into a simulated adjacent serial section.

    See the module docstring for the full step list, conventions, and ground
    truth.  ``.X`` is the sole source of truth for expression; any inherited
    ``obsm["X_pca"]`` is dropped since it would go stale.  Call
    ``utils.compute_joint_pca`` on the result (with ``reference``) for a
    consistent shared embedding.  Key expression parameters:

    expr_alpha : float
        Noise scale; per cell i and gene g the std is ``expr_alpha * std_g`` over
        cells of i's cell type (std measured on the chosen expression matrix in
        its native units).
    expression_layer : str or None
        Source expression to perturb; ``None`` uses ``.X``.  The noised result is
        written back to that same location (``.X`` and, if given, the layer), and
        the clean original is saved to ``layers["X_unperturbed"]``.
    nonneg_clip : bool
        Clip noised expression at 0 (appropriate for counts / log-normalized
        data; set False if your expression representation can be negative).
    reference : AnnData or None
        Slice ``section`` is a crop of; every surviving cell of ``section`` must
        be present in it (matched by ``obs_names``) so FOSCTTM ground truth can be
        mapped into ``reference``'s row index space.  ``None`` -> clean copy of
        the input section.

    Returns
    -------
    AnnData
        Simulated section with noised ``.X`` and full ground truth.
    """
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1); got {dropout_rate}.")
    if not (0.0 <= jitter_fraction <= 1.0):
        raise ValueError(f"jitter_fraction must be in [0, 1]; got {jitter_fraction}.")
    if expr_alpha < 0:
        raise ValueError(f"expr_alpha must be >= 0; got {expr_alpha}.")
    if not (0.0 <= label_flip_rate <= 1.0):
        raise ValueError(f"label_flip_rate must be in [0, 1]; got {label_flip_rate}.")
    if birth_rate < 0:
        raise ValueError(f"birth_rate must be >= 0; got {birth_rate}.")

    if reference is None:
        reference = section.copy()

    rng = np.random.default_rng(seed)
    meta = dict(section.uns.get("self_alignment_test", {}))

    full_coords = np.asarray(section.obsm[spatial_key], dtype=np.float64)[:, :2]
    n_full = full_coords.shape[0]
    if n_full < min_cells:
        raise ValueError(f"section has {n_full} cells (< min_cells={min_cells}).")

    mins, maxs = full_coords.min(0), full_coords.max(0)
    bbox_diag = float(np.linalg.norm(maxs - mins))
    nn_full = _nn_distances(full_coords)
    median_nn = float(np.median(nn_full))

    if warp_amplitude is None:
        warp_amplitude = 0.025 * bbox_diag
    if jitter_sigma is None:
        jitter_sigma = 0.25 * median_nn
    if min_dist is None:
        min_dist = float(np.quantile(nn_full, min_dist_quantile))
    if hardcore_diameter is not None:
        min_dist = float(min(min_dist, hardcore_diameter))

    # sample rigid transform up front (independent of which cells survive)
    if rotation_deg is None:
        rotation_deg = float(rng.uniform(*rotation_range))
    angle_rad = float(np.radians(rotation_deg))
    if translation is None:
        amp = translation_scale * bbox_diag
        t = rng.uniform(-amp, amp, size=2).astype(np.float64)
    else:
        t = np.asarray(translation, float).ravel()
        if t.shape != (2,):
            raise ValueError(f"translation must have length 2; got {t.shape}.")
    A = _linear_part(angle_rad, reflect)
    p = _resolve_pivot(pivot, full_coords, meta)
    b = (p + t) - p @ A.T

    # -- 1. dropout --
    keep = (rng.random(n_full) >= dropout_rate) if dropout_rate > 0 \
        else np.ones(n_full, dtype=bool)
    kept_pos = np.flatnonzero(keep)
    if kept_pos.size < min_cells:
        raise ValueError(
            f"Only {kept_pos.size} cells survive dropout_rate={dropout_rate} "
            f"(min_cells={min_cells}). Lower dropout_rate.")
    sim = section[kept_pos].copy()
    coords0 = full_coords[kept_pos]
    n = coords0.shape[0]

    # -- 2-4. warp -> jitter -> collision-resolve (pre-rigid frame) --
    warped, ctrl_pts, ctrl_disp = _smooth_warp(coords0, warp_amplitude, warp_n_control, rng)
    jit = np.zeros((n, 2), dtype=np.float64)
    if jitter_sigma > 0 and jitter_fraction > 0:
        n_jit = int(round(jitter_fraction * n))
        if n_jit > 0:
            sel = rng.choice(n, size=n_jit, replace=False)
            jit[sel] = rng.normal(0.0, jitter_sigma, size=(n_jit, 2))
    jittered = warped + jit
    resolved, n_iter, final_min = _resolve_collisions(jittered, min_dist, collision_max_iter, rng)
    if final_min < min_dist - 1e-9:
        warnings.warn(
            f"Collision resolution did not fully converge in {collision_max_iter} "
            f"iters (min dist {final_min:.4g} < d_min {min_dist:.4g}). Increase "
            f"collision_max_iter or lower jitter/warp amplitude.", UserWarning, stacklevel=2)

    # -- 5. rigid (distance-preserving -> no-touch guarantee survives) --
    final_coords = resolved @ A.T + b
    new_spatial = np.asarray(sim.obsm[spatial_key], dtype=np.float64).copy()
    new_spatial[:, :2] = final_coords
    sim.obsm[spatial_key] = new_spatial

    # -- 6. expression noise: add to .X (single source of truth) --
    E_clean = _to_dense(sim.layers[expression_layer] if expression_layer else sim.X)
    E_clean = _sanitize(E_clean, "section .X")
    cell_types = (sim.obs[celltype_key].astype(str).to_numpy()
                  if celltype_key in sim.obs.columns else None)
    if cell_types is None:
        warnings.warn(
            f"celltype_key '{celltype_key}' not in obs; expression noise falls back "
            f"to global per-gene std (gene-aware only).", UserWarning, stacklevel=2)

    # Auto-detect the expression representation.  Negatives => .X is already
    # normalized/scaled (not counts): do NOT clip the noise.
    eff_nonneg = (not _has_negatives(E_clean)) if nonneg_clip == "auto" else bool(nonneg_clip)

    E_noised, noise_info = _celltype_gene_aware_noise(
        E_clean, cell_types, expr_alpha, min_group_cells, rng, eff_nonneg)

    sim.layers["X_unperturbed"] = E_clean.astype(np.float32)   # provenance
    sim.X = E_noised.astype(np.float32)                        # noised .X is canonical
    if expression_layer is not None:
        sim.layers[expression_layer] = E_noised.astype(np.float32)
    # any obsm['X_pca'] inherited from `section` is now a stale embedding of the
    # pre-noise expression; drop it so callers cannot silently use it.  Call
    # utils.compute_joint_pca(sim, reference) for a consistent shared embedding.
    sim.obsm.pop("X_pca", None)

    # -- 7. annotation noise: corrupt the labels the aligner READS (expression
    #       noise above used the TRUE cell types, so this is an independent axis) --
    n_flipped = 0
    if celltype_key in sim.obs.columns:
        labels_clean = sim.obs[celltype_key].astype(str).to_numpy()
        sim.obs[celltype_key + "_clean"] = labels_clean
        if label_flip_rate > 0.0:
            labels_noised, flip_mask = _apply_label_noise(labels_clean, rng, label_flip_rate)
            sim.obs[celltype_key] = labels_noised.astype(str)
            n_flipped = int(flip_mask.sum())
        else:
            flip_mask = np.zeros(n, dtype=bool)
        sim.obs["adjacent_label_flipped"] = flip_mask
    else:
        sim.obs["adjacent_label_flipped"] = np.zeros(n, dtype=bool)

    # -- ground truth & provenance --
    # `kept_pos` is a position local to `section` (0..n_full-1). When `section`
    # is a spatial crop of a larger `reference` (the common case -- see module
    # docstring), that local position is NOT a valid row index into `reference`.
    # Map it through obs_names so `dropout_kept_positions` is always a row index
    # into `reference`, matching what evaluate_alignment/foscttm assume. Without
    # this, FOSCTTM ground truth silently points at the wrong reference cells and
    # the score collapses to ~0.5 (chance) even when the alignment is correct.
    ref_kept_pos = reference.obs_names.get_indexer(section.obs_names[kept_pos])
    n_unmatched = int((ref_kept_pos < 0).sum())
    if n_unmatched:
        raise ValueError(
            f"{n_unmatched} surviving cell(s) from `section` were not found in "
            f"`reference` by obs_name. `reference` must contain every cell in "
            f"`section` (matching obs_names) so that FOSCTTM ground-truth indices "
            f"are valid row positions in `reference`.")

    gt = np.asarray(sim.obsm[spatial_key], dtype=np.float64).copy()
    gt[:, :2] = coords0
    sim.obsm["spatial_unperturbed"] = gt
    sim.obsm["adjacent_displacement_prerigid"] = (resolved - coords0).astype(np.float64)
    sim.obs["adjacent_kept"] = np.ones(n, dtype=bool)
    sim.obs["adjacent_is_birth"] = np.zeros(n, dtype=bool)

    A_inv, b_inv = invert_rigid(A, b)
    meta["adjacent_simulation"] = {
        "seed": (-1 if seed is None else seed),
        "apply_order": ["dropout", "warp", "jitter", "collision_resolve",
                        "rigid", "expr_noise(.X)", "label_noise", "birth_cells"],
        "model": "cell_preserving_z_displaced",
        "affine_A": A, "affine_b": b, "affine_A_inv": A_inv, "affine_b_inv": b_inv,
        "rotation_deg": float(rotation_deg), "rotation_radians": angle_rad,
        "translation": [float(t[0]), float(t[1])],
        "pivot": [float(p[0]), float(p[1])], "reflect": bool(reflect),
        "is_proper_rigid": bool(round(float(np.linalg.det(A))) == 1),
        "warp_amplitude": float(warp_amplitude), "warp_n_control": int(warp_n_control),
        "warp_control_points": ctrl_pts, "warp_control_displacements": ctrl_disp,
        "jitter_sigma": float(jitter_sigma), "jitter_fraction": float(jitter_fraction),
        "min_dist": float(min_dist), "min_dist_quantile": float(min_dist_quantile),
        "hardcore_diameter": ("" if hardcore_diameter is None else float(hardcore_diameter)),
        "collision_iterations": int(n_iter),
        "final_min_pairwise_distance": float(final_min),
        "no_touch_satisfied": bool(final_min >= min_dist - 1e-9),
        "dropout_rate": float(dropout_rate),
        "n_obs_input": int(n_full), "n_obs_output": int(n),
        "dropout_kept_positions": ref_kept_pos.astype(np.int64),
        "label_flip_rate": float(label_flip_rate),
        "n_labels_flipped": int(n_flipped),
        "birth_rate": float(birth_rate),
        "birth_offset_scale": float(birth_offset_scale),
        "expr_noise": noise_info,
        "expression_layer": (expression_layer if expression_layer is not None else ""),
        "ground_truth_keys": {
            "spatial_unperturbed": "obsm['spatial_unperturbed'] (NaN for birth cells)",
            "prerigid_displacement": "obsm['adjacent_displacement_prerigid']",
            "expression_pre_noise": "layers['X_unperturbed']",
            "noised_expression": ".X (canonical)",
            "shared_embedding": "call utils.compute_joint_pca(sim, reference) to populate obsm['X_pca']",
            "clean_labels": f"obs['{celltype_key}_clean']",
            "label_flipped_mask": "obs['adjacent_label_flipped']",
            "birth_mask": "obs['adjacent_is_birth'] (exclude from correspondence scoring)",
            "correspondence": "obs_names match the parent slice (birth cells: 'spurious_*', no parent)",
        },
    }

    # -- spurious (birth) cells: unmatched clutter appended after all GT is set --
    n_birth = int(round(birth_rate * n))
    meta["adjacent_simulation"]["n_birth"] = int(n_birth)
    sim.uns["self_alignment_test"] = meta
    if n_birth <= 0:
        return sim

    births = _make_spurious_cells(
        sim, rng, n_birth, spatial_key, median_nn, birth_offset_scale,
        (-1 if seed is None else seed))
    assert births is not None  # n_birth > 0 here, so _make_spurious_cells returns rows
    # anndata.concat drops .uns; reattach the provenance afterwards.
    combined = ad.concat([sim, births], axis=0, join="outer", uns_merge=None)
    combined.uns["self_alignment_test"] = meta
    return combined

def show_slice(_slice, spatial_key: str = "spatial", label_key: str = "cell_type_annot",
               point_size: float = 3.0, alpha: float = 1.0, title: Optional[str] = None):
    coords = np.asarray(_slice.obsm[spatial_key], dtype=np.float64)[:, :2]

    if label_key in _slice.obs.columns:
        labels = _slice.obs[label_key].astype(str).values
        unique = sorted(set(labels))
        cmap = plt.get_cmap("tab20", max(len(unique), 1))
        lbl2idx = {l: i for i, l in enumerate(unique)}
        colors = np.array([cmap(lbl2idx[l]) for l in labels])
        legend_handles = [
            mpatches.Patch(color=cmap(lbl2idx[l]), label=l)
            for l in unique[:20]
        ]
        if len(unique) > 20:
            legend_handles.append(
                mpatches.Patch(color="none", label=f"… +{len(unique)-20} more"))
    else:
        colors = np.full((len(coords), 4), [0.5, 0.5, 0.5, 0.6])
        legend_handles = None

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=point_size, alpha=alpha,
               linewidths=0, rasterized=True)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("X coordinate (µm)", fontsize=9)
    ax.set_ylabel("Y coordinate (µm)", fontsize=9)
    ax.set_title(title or f"Slice  ·  {_slice.n_obs:,} cells", fontsize=10)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right", fontsize=6,
                  markerscale=2, framealpha=0.6, title=label_key, title_fontsize=7)
    plt.tight_layout()
    plt.show()


# main function
if __name__ == "__main__":
    import scanpy as sc
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Simulate an adjacent slice from a cut section.")
    parser.add_argument("--input_h5ad", default="results/adata24wk_donor_id_12_slice_0_left_hemi.h5ad", help="Path to input section .h5ad file.")
    parser.add_argument("--reference_h5ad", default="adata24wk_donor_id_12_slice_0.h5ad", help="Path to reference slice .h5ad file for joint PCA.")
    parser.add_argument("--output_h5ad", default="adata24wk_donor_id_12_slice_0_cropped_perturbed.h5ad", help="Path to output simulated adjacent slice .h5ad file.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    try:
        adata = sc.read_h5ad(args.input_h5ad)
        reference = sc.read_h5ad(args.reference_h5ad)
        sim = simulate_adjacent_slice(adata, reference=reference, seed=args.seed)
        sim, reference = compute_joint_pca(sim, reference)
        show_slice(sim)
        sim.write_h5ad(args.output_h5ad)
        print(f"Simulated adjacent slice written to {args.output_h5ad}.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
