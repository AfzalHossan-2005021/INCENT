"""
perturb.py
===================
Seeded, reproducible perturbation of a cropped self-alignment-test portion.

Designed to sit downstream of ``synthesize.create_interactive_rectangular_portion``
(or ``core.create_random_rectangular_portion``) and *upstream* of
``pairwise_align`` / ``hierarchical_pairwise_align``.  It deliberately does NOT
touch the GUI/crop path, so the crop stays deterministic and the perturbation is
independently seedable and sweepable.

Three perturbations are applied, in this fixed order:

    1. Random cell dropout        — Bernoulli(keep) over cells, breaks the 1:1
                                     correspondence and forces partial alignment.
    2. Per-cell coordinate jitter — N(0, sigma^2 I) added to a *random fraction*
                                     of the surviving cells (segmentation noise).
    3. Additional rigid transform — a single global rotation theta (optional
                                     reflection) + translation t applied to all
                                     surviving coordinates.

So, for each surviving cell:

    X_perturbed = (X_unperturbed + D) @ A.T + b

where ``D`` is the (mostly zero) per-cell jitter displacement, ``A`` is the 2x2
linear part (rotation, optionally reflected; det = -1 iff reflected) and ``b`` is
the offset that folds in the rotation pivot and translation.

Rotation convention is identical to ``synthesize.py`` / ``core.py``:
    R(theta) = [[cos, -sin], [sin, cos]]
    world->local (their masking) : (coords - center) @ R
    local->world (their overlay) : local @ R.T + center
so the *forward* rotation applied here uses ``@ A.T`` (= ``@ R.T`` for the pure
rotation case), and the stored theta composes consistently with
``window_angle_radians``.

Ground truth (everything an alignment scorer needs) is written so the perturbed
portion is self-contained:

    portion.obsm["spatial_unperturbed"]          (N,2) parent-frame coords = the
                                                   gold target an aligner must
                                                   recover (equals the cell's
                                                   coordinate in the parent slice).
    portion.obsm["perturb_jitter_displacement"]  (N,2) the D vector (0 where not jittered)
    portion.obs ["perturb_jittered"]             (N,)  bool, which cells were jittered
    portion.uns["self_alignment_test"]["perturbation"]  scalar/array provenance
                                                   incl. A, b, theta, t, pivot,
                                                   reflect, dropout rate, sigma,
                                                   fraction, and the RNG seed.

The exact transform recoverable by a *rigid* aligner is (A, b); the jitter is
irreducible noise and therefore sets the floor on achievable registration error.

Example
-------
::

    from synthesize import create_interactive_rectangular_portion
    from perturb    import perturb_portion

    selector = create_interactive_rectangular_portion(adata)
    portion  = selector.extract()

    pert = perturb_portion(
        portion,
        seed=0,
        rotation_range=(-30.0, 30.0),   # theta sampled here if rotation_deg is None
        translation_scale=0.10,         # |t| ~ fraction of bbox diagonal
        dropout_rate=0.10,              # remove ~10% of cells
        jitter_fraction=0.30,           # 30% of survivors get jittered
        jitter_sigma=5.0,               # micron (sub-cell-diameter for MERFISH)
    )
    # pass `pert` as one slice and the parent `adata` as the other:
    # hierarchical_pairwise_align(sliceA=pert, sliceB=adata, ...)
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple, Union

import numpy as np
import scanpy as sc
from anndata import AnnData

from benchmarks import data


# ----------------------------------------------------------------------------
# Geometry helpers (convention identical to synthesize.py / core.py)
# ----------------------------------------------------------------------------

def _rotation_matrix(angle_rad: float) -> np.ndarray:
    """2x2 counter-clockwise rotation matrix R(theta) = [[c,-s],[s,c]]."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _linear_part(angle_rad: float, reflect: bool) -> np.ndarray:
    """
    2x2 linear map ``A`` for the rigid (or, if ``reflect``, improper-rigid) transform.

    A = R(theta)              (proper rotation,  det +1)
    A = R(theta) @ diag(1,-1) (with reflection,  det -1)

    Forward map on row-vector coordinates is ``X @ A.T``.
    """
    A = _rotation_matrix(angle_rad)
    if reflect:
        A = A @ np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    return A


def invert_rigid(A: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert the affine map ``X' = X @ A.T + b``.

    Returns (A_inv, b_inv) such that ``X = X' @ A_inv.T + b_inv``.  Useful for an
    alignment scorer: apply this to the perturbed coords to recover the parent
    frame (up to the jitter noise floor).
    """
    A_inv = np.linalg.inv(A)
    b_inv = -b @ A_inv.T
    return A_inv, b_inv


def _get_xy(adata: AnnData, spatial_key: str) -> np.ndarray:
    if spatial_key not in adata.obsm:
        raise KeyError(
            f"spatial_key '{spatial_key}' not in adata.obsm "
            f"(available: {list(adata.obsm.keys())})."
        )
    coords = np.asarray(adata.obsm[spatial_key], dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['{spatial_key}'] must be (N, 2+); got {coords.shape}."
        )
    return coords[:, :2].copy()


def _resolve_pivot(
    pivot: Union[str, Sequence[float]],
    coords: np.ndarray,
    meta: dict,
) -> np.ndarray:
    """Resolve 'centroid' | 'window_center' | (px, py) into a (2,) array."""
    if isinstance(pivot, str):
        if pivot == "centroid":
            return coords.mean(axis=0)
        if pivot == "window_center":
            wc = meta.get("window_center")
            if wc is None:
                return coords.mean(axis=0)
            return np.asarray(wc, dtype=np.float64)[:2]
        raise ValueError(
            f"pivot string must be 'centroid' or 'window_center'; got {pivot!r}."
        )
    p = np.asarray(pivot, dtype=np.float64).ravel()
    if p.shape != (2,):
        raise ValueError(f"pivot coordinate must have length 2; got shape {p.shape}.")
    return p


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------

def perturb_portion(
    portion: AnnData,
    *,
    spatial_key: str = "spatial",
    # -- 3. rigid transform ---------------------------------------------------
    rotation_deg: Optional[float] = None,
    rotation_range: Tuple[float, float] = (-30.0, 30.0),
    translation: Optional[Sequence[float]] = None,
    translation_scale: float = 0.10,
    reflect: bool = False,
    pivot: Union[str, Sequence[float]] = "centroid",
    # -- 1. dropout -----------------------------------------------------------
    dropout_rate: float = 0.05,
    # -- 2. jitter ------------------------------------------------------------
    jitter_fraction: float = 0.10,
    jitter_sigma: float = 5.0,
    # -- bookkeeping ----------------------------------------------------------
    min_cells: Optional[int] = None,
    seed: Optional[int] = None,
    inplace: bool = False,
) -> AnnData:
    """
    Apply dropout + per-cell jitter (on a fraction) + an additional rigid
    transform to a cropped self-alignment-test portion, recording full ground
    truth.

    Parameters
    ----------
    portion : AnnData
        A crop produced by the self-alignment harness.  Its coordinates are
        assumed to live in the *parent* frame (i.e. ``obsm[spatial_key]`` of a
        cropped cell equals that cell's coordinate in the full slice).
    spatial_key : str
        Key in ``obsm`` holding (N, 2+) coordinates.
    rotation_deg : float or None
        theta in **degrees**.  If ``None``, sampled uniformly from ``rotation_range``.
    rotation_range : (float, float)
        Inclusive degree range used when ``rotation_deg is None``.
    translation : (float, float) or None
        Absolute translation **t** in micron.  If ``None``, each component is
        sampled uniformly from ``+/- translation_scale * bbox_diagonal``.
    translation_scale : float
        Fraction of the coordinate bounding-box diagonal used when sampling **t**.
    reflect : bool
        If ``True``, include a reflection (improper rigid, det A = -1).  Useful for
        stress-testing methods that cannot resolve chirality.
    pivot : {'centroid', 'window_center'} or (float, float)
        Rotation pivot.  ``'window_center'`` reads
        ``uns['self_alignment_test']['window_center']`` and falls back to the
        centroid if absent.
    dropout_rate : float in [0, 1)
        Probability of *removing* each cell (Bernoulli).  ``0`` keeps all cells.
    jitter_fraction : float in [0, 1]
        Fraction of the *surviving* cells (chosen at random, without replacement)
        to which Gaussian jitter is applied.
    jitter_sigma : float
        Std-dev (micron, per axis) of the isotropic Gaussian jitter N(0, sigma^2 I).
    min_cells : int or None
        Minimum number of surviving cells required.  If ``None``, taken from
        ``uns['self_alignment_test']`` (key ``min_cells``) or defaults to 4.
    seed : int or None
        Seed for a single ``numpy.random.default_rng`` driving every random draw
        (sampled rotation/translation, dropout, jitter selection, jitter noise),
        in that order.  Pass an int for full reproducibility.
    inplace : bool
        If ``True``, modify and return ``portion`` directly; otherwise operate on
        a copy (default).

    Returns
    -------
    AnnData
        The perturbed portion with ground truth in ``obsm`` / ``obs`` / ``uns``
        (see module docstring).

    Raises
    ------
    ValueError
        On out-of-range parameters or if fewer than ``min_cells`` survive dropout.
    """
    # -- validate -------------------------------------------------------------
    if not (0.0 <= dropout_rate < 1.0):
        raise ValueError(f"dropout_rate must be in [0, 1); got {dropout_rate}.")
    if not (0.0 <= jitter_fraction <= 1.0):
        raise ValueError(f"jitter_fraction must be in [0, 1]; got {jitter_fraction}.")
    if jitter_sigma < 0.0:
        raise ValueError(f"jitter_sigma must be >= 0; got {jitter_sigma}.")

    rng = np.random.default_rng(seed)

    meta_full = dict(portion.uns.get("self_alignment_test", {}))
    if min_cells is None:
        min_cells = int(meta_full.get("min_cells", 4))

    coords_in = _get_xy(portion, spatial_key)          # parent-frame, (N0, 2)
    n_in = coords_in.shape[0]

    # -- resolve the rigid transform parameters (sample BEFORE dropout so the --
    #    transform is independent of which cells survive) ----------------------
    if rotation_deg is None:
        lo, hi = float(rotation_range[0]), float(rotation_range[1])
        rotation_deg = float(rng.uniform(lo, hi))
    angle_rad = float(np.radians(rotation_deg))

    if translation is None:
        mins = coords_in.min(axis=0)
        maxs = coords_in.max(axis=0)
        diag = float(np.linalg.norm(maxs - mins))
        amp = translation_scale * diag
        t = rng.uniform(-amp, amp, size=2).astype(np.float64)
    else:
        t = np.asarray(translation, dtype=np.float64).ravel()
        if t.shape != (2,):
            raise ValueError(f"translation must have length 2; got shape {t.shape}.")

    p = _resolve_pivot(pivot, coords_in, meta_full)
    A = _linear_part(angle_rad, reflect)               # 2x2 linear part
    # X' = (X - p) @ A.T + p + t  ==  X @ A.T + b,  with:
    b = (p + t) - p @ A.T                               # (2,)

    # -- 1. dropout (Bernoulli keep) ------------------------------------------
    if dropout_rate > 0.0:
        keep = rng.random(n_in) >= dropout_rate
    else:
        keep = np.ones(n_in, dtype=bool)
    kept_pos = np.flatnonzero(keep)
    n_out = kept_pos.size
    if n_out < min_cells:
        raise ValueError(
            f"Only {n_out} cell(s) survive dropout_rate={dropout_rate} "
            f"(minimum: {min_cells}).  Lower dropout_rate or enlarge the crop."
        )

    # subset (preserves X, layers, obs, var, obsm, obs_names -> correspondence)
    if inplace:
        out = portion
        out._inplace_subset_obs(kept_pos)
    else:
        out = portion[kept_pos].copy()

    coords_kept = coords_in[kept_pos]                   # (N_out, 2) parent frame
    n = coords_kept.shape[0]

    # -- 2. per-cell jitter on a random fraction of survivors -----------------
    n_jit = int(round(jitter_fraction * n))
    jittered = np.zeros(n, dtype=bool)
    disp = np.zeros((n, 2), dtype=np.float64)
    if n_jit > 0 and jitter_sigma > 0.0:
        jit_idx = rng.choice(n, size=n_jit, replace=False)
        disp[jit_idx] = rng.normal(0.0, jitter_sigma, size=(n_jit, 2))
        jittered[jit_idx] = True

    coords_jittered = coords_kept + disp

    # -- 3. apply the global rigid transform ----------------------------------
    coords_out = coords_jittered @ A.T + b

    # write coords back (preserve any extra columns beyond xy)
    new_spatial = np.asarray(out.obsm[spatial_key], dtype=np.float64).copy()
    new_spatial[:, :2] = coords_out
    out.obsm[spatial_key] = new_spatial

    # -- ground truth ----------------------------------------------------------
    gt_unperturbed = np.asarray(out.obsm[spatial_key], dtype=np.float64).copy()
    gt_unperturbed[:, :2] = coords_kept          # parent-frame gold target
    out.obsm["spatial_unperturbed"] = gt_unperturbed
    out.obsm["perturb_jitter_displacement"] = disp
    out.obs["perturb_jittered"] = jittered

    A_inv, b_inv = invert_rigid(A, b)
    meta_full["perturbation"] = {
        "seed": seed,
        "apply_order": ["dropout", "jitter", "rigid"],
        # forward affine  X_perturbed = X_unperturbed @ A.T + b  (jitter excluded)
        "affine_A": A,
        "affine_b": b,
        "affine_A_inv": A_inv,
        "affine_b_inv": b_inv,
        # human-readable rigid params
        "rotation_deg": rotation_deg,
        "rotation_radians": angle_rad,
        "translation": [float(t[0]), float(t[1])],
        "pivot": [float(p[0]), float(p[1])],
        "reflect": bool(reflect),
        "is_proper_rigid": bool(round(float(np.linalg.det(A))) == 1),
        # dropout
        "dropout_rate": float(dropout_rate),
        "n_obs_input": int(n_in),
        "n_obs_output": int(n_out),
        "dropout_kept_positions": kept_pos.astype(np.int64),
        # jitter
        "jitter_fraction": float(jitter_fraction),
        "jitter_sigma": float(jitter_sigma),
        "n_jittered": int(jittered.sum()),
        # where to find per-cell ground truth
        "ground_truth_keys": {
            "spatial_unperturbed": "obsm['spatial_unperturbed']",
            "jitter_displacement": "obsm['perturb_jitter_displacement']",
            "jittered_mask": "obs['perturb_jittered']",
            "correspondence": "obs_names match the parent slice",
        },
    }
    out.uns["self_alignment_test"] = meta_full
    return out


# main function
if __name__ == "__main__":
    data_dir = "../data/synthetic/"
    data_file="adata4wk_donor_id_1_slice_0_corpped.h5ad"

    adata = sc.read_h5ad(data_dir + data_file)

    perturbed = perturb_portion(adata, seed=0)

    # save the perturbed portion for later use (e.g. in pairwise_align):
    perturbed.write_h5ad(data_dir + data_file.replace(".h5ad", "_perturbed.h5ad"))
