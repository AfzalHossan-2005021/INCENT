"""
evaluation.py
=============
Ground-truth-anchored metric battery for scoring a cell-level alignment ``pi``
between a simulated adjacent slice (see :func:`perturb.simulate_adjacent_slice`)
and its clean parent/reference.

The metrics deliberately separate the distinct axes of alignment quality, so no
single (possibly misleading) number drives weight selection:

* **registration** (PRIMARY, exact, label-free) -- does each cell map to its true
  geometric partner?  Uses the exact correspondence (matching ``obs_names``) and
  the gold parent-frame coordinates in ``obsm['spatial_unperturbed']`` written by
  the simulator.  Birth/spurious cells (``obs['adjacent_is_birth']``) have no
  partner and are excluded.
* **spatial coherence** (label-free) -- does the mapping preserve local geometry
  (do a cell's neighbours stay neighbours after mapping)?  Catches the
  "right type, wrong place" failure that label-transfer metrics miss.  Runs on any
  pair, with or without ground truth.
* **label transfer** (ARI / accuracy) -- transfer cell-type labels through ``pi``
  and compare to the target's labels.  Reported, never the sole objective (it is
  spatially blind, and the cost already ingests cell type).
* **expression transfer correlation** (label-free) -- barycentric-predicted vs
  measured target expression; a cheap expression-consistency proxy.

Orientation convention: ``pi`` has shape ``(sliceA.n_obs, sliceB.n_obs)`` as
returned by ``hierarchical_pairwise_align(sliceA, sliceB)``.  Pass ``sim_axis`` to
say which axis is the simulated slice that carries the ground truth (default 0).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------

def _dense(X) -> np.ndarray:
    return np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)


def _median_nn(coords: np.ndarray) -> float:
    """Median nearest-neighbour distance; a robust length scale for normalization."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.shape[0] < 2:
        return 1.0
    d, _ = cKDTree(coords).query(coords, k=2)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(np.median(nn)) if nn.size else 1.0


def _barycentric_images(pi: np.ndarray, coords_target: np.ndarray):
    """
    Map each source cell to a target-frame location = mass-weighted mean of the
    target coordinates it transports to. Returns (images, mapped_mask, row_mass).
    Rows with zero mass (unmapped, e.g. outside the overlap shadow) are flagged.
    """
    row_mass = np.asarray(pi).sum(axis=1)
    mapped = row_mass > 0
    images = np.full((pi.shape[0], coords_target.shape[1]), np.nan, dtype=np.float64)
    if mapped.any():
        images[mapped] = (pi[mapped] @ coords_target) / row_mass[mapped, None]
    return images, mapped, row_mass


# ----------------------------------------------------------------------------
# 1. registration accuracy  (PRIMARY, exact, label-free)
# ----------------------------------------------------------------------------

def registration_scores(
    pi: np.ndarray,
    sim,
    parent,
    *,
    spatial_key: str = "spatial",
    sim_axis: int = 0,
) -> dict:
    """
    Exact registration metrics from the simulator's ground truth.

    Correspondence is by matching ``obs_names`` (each surviving simulated cell is a
    known parent cell); the gold target coordinate is ``sim.obsm['spatial_unperturbed']``.
    Birth cells and cells that received no transport mass are handled explicitly.

    Returns a dict with:
      ``soft_corr_mass``   fraction of matched-cell transported mass landing on the
                           true partner (smooth in [0,1]; the recommended objective).
      ``hard_accuracy``    fraction of mapped matched cells whose argmax partner is
                           the true partner.
      ``coverage``         fraction of matched cells that received any mass.
      ``median_reg_error`` median ||predicted - gold|| over mapped matched cells,
                           normalized by the parent median-NN spacing (raw also given).
      ``mass_on_births``   fraction of total mass absorbed by spurious source cells
                           (clutter-robustness diagnostic).
      ``n_matched`` / ``n_scored``.
    """
    pi = np.asarray(pi, dtype=np.float64)
    if sim_axis == 1:
        pi = pi.T  # orient to rows=sim, cols=parent
    elif sim_axis != 0:
        raise ValueError("sim_axis must be 0 or 1.")

    n_sim, n_par = pi.shape
    if sim.n_obs != n_sim or parent.n_obs != n_par:
        raise ValueError(
            f"pi shape {pi.shape} does not match (sim={sim.n_obs}, parent={parent.n_obs}); "
            f"check sim_axis."
        )

    coords_par = np.asarray(parent.obsm[spatial_key], dtype=np.float64)[:, :2]
    gold = np.asarray(sim.obsm["spatial_unperturbed"], dtype=np.float64)[:, :2]

    is_birth = (
        np.asarray(sim.obs["adjacent_is_birth"], dtype=bool)
        if "adjacent_is_birth" in sim.obs.columns
        else np.zeros(n_sim, dtype=bool)
    )

    par_pos = {name: j for j, name in enumerate(parent.obs_names)}
    true_j = np.array([par_pos.get(name, -1) for name in sim.obs_names], dtype=int)

    matched = (true_j >= 0) & (~is_birth)
    images, mapped, row_mass = _barycentric_images(pi, coords_par)

    total_mass = float(row_mass.sum())
    mass_on_births = float(row_mass[is_birth].sum() / total_mass) if total_mass > 0 else 0.0

    matched_idx = np.flatnonzero(matched)
    den = float(row_mass[matched_idx].sum())
    if den > 0 and matched_idx.size:
        num = float(pi[matched_idx, true_j[matched_idx]].sum())
        soft = num / den
    else:
        soft = 0.0

    scored = matched & mapped
    scored_idx = np.flatnonzero(scored)
    if scored_idx.size:
        hard = float(np.mean(pi[scored_idx].argmax(axis=1) == true_j[scored_idx]))
        err = np.linalg.norm(images[scored_idx] - gold[scored_idx], axis=1)
        raw_err = float(np.median(err))
    else:
        hard, raw_err = 0.0, float("inf")

    spacing = _median_nn(coords_par)
    coverage = float(mapped[matched_idx].mean()) if matched_idx.size else 0.0

    return {
        "soft_corr_mass": soft,
        "hard_accuracy": hard,
        "coverage": coverage,
        "median_reg_error": raw_err / spacing if np.isfinite(raw_err) else float("inf"),
        "median_reg_error_raw": raw_err,
        "spacing": spacing,
        "mass_on_births": mass_on_births,
        "n_matched": int(matched_idx.size),
        "n_scored": int(scored_idx.size),
    }


# ----------------------------------------------------------------------------
# 2. spatial coherence  (label-free, works on any pair)
# ----------------------------------------------------------------------------

def spatial_coherence(
    pi: np.ndarray,
    coords_source: np.ndarray,
    coords_target: np.ndarray,
    *,
    k: int = 15,
) -> dict:
    """
    Neighbourhood-preservation coherence of the mapping (label-free).

    Each source cell is barycentrically mapped into the target frame; we then
    measure how much each cell's k-NN set in source space overlaps its k-NN set
    among the mapped images. 1.0 = local geometry perfectly preserved; ~k/N =
    random. This penalizes mappings that scatter neighbours apart even when label
    or expression agreement looks fine.
    """
    coords_source = np.asarray(coords_source, dtype=np.float64)[:, :2]
    coords_target = np.asarray(coords_target, dtype=np.float64)[:, :2]
    images, mapped, _ = _barycentric_images(np.asarray(pi, dtype=np.float64), coords_target)

    src = coords_source[mapped]
    img = images[mapped]
    n = src.shape[0]
    if n < k + 1:
        return {"coherence": float("nan"), "n_mapped": int(n)}

    def _knn_sets(X):
        _, idx = cKDTree(X).query(X, k=k + 1)
        return idx[:, 1:]  # drop self

    src_nn = _knn_sets(src)
    img_nn = _knn_sets(img)
    overlap = np.array([
        np.intersect1d(src_nn[i], img_nn[i], assume_unique=True).size
        for i in range(n)
    ], dtype=np.float64) / float(k)
    return {"coherence": float(overlap.mean()), "n_mapped": int(n)}


# ----------------------------------------------------------------------------
# 3. label transfer  (reported; spatially blind)
# ----------------------------------------------------------------------------

def label_transfer_scores(
    pi: np.ndarray,
    labels_source: np.ndarray,
    labels_target: np.ndarray,
    *,
    valid_target: Optional[np.ndarray] = None,
) -> dict:
    """
    Transfer labels source->target by mass-weighted vote and score vs target labels.

    ``pred_target[j] = argmax_c sum_{i: labels_source[i]==c} pi[i, j]``, scored only
    on target cells that received mass (and, if given, are ``valid_target``).
    Returns adjusted Rand index (primary) and accuracy.
    """
    pi = np.asarray(pi, dtype=np.float64)
    labels_source = np.asarray(labels_source).astype(str)
    labels_target = np.asarray(labels_target).astype(str)

    col_mass = pi.sum(axis=0)
    mapped = col_mass > 0
    if valid_target is not None:
        mapped = mapped & np.asarray(valid_target, dtype=bool)
    if not mapped.any():
        return {"ari": float("nan"), "accuracy": float("nan"), "n_scored": 0}

    types = np.unique(labels_source)
    onehot = (labels_source[:, None] == types[None, :]).astype(np.float64)  # (n_src, n_types)
    votes = onehot.T @ pi  # (n_types, n_target)
    pred = types[votes.argmax(axis=0)]

    pred_m = pred[mapped]
    true_m = labels_target[mapped]
    return {
        "ari": float(adjusted_rand_score(true_m, pred_m)),
        "accuracy": float(np.mean(pred_m == true_m)),
        "n_scored": int(mapped.sum()),
    }


# ----------------------------------------------------------------------------
# 4. expression transfer correlation  (label-free proxy)
# ----------------------------------------------------------------------------

def expression_transfer_corr(
    pi: np.ndarray,
    expr_source: np.ndarray,
    expr_target: np.ndarray,
    *,
    max_genes: int = 2000,
) -> dict:
    """
    Mean per-gene Pearson correlation between barycentric-predicted and measured
    target expression (label-free expression-consistency proxy). Genes with zero
    variance in either side are skipped; up to ``max_genes`` most-variable genes
    are used for speed.
    """
    pi = np.asarray(pi, dtype=np.float64)
    A = _dense(expr_source)
    B = _dense(expr_target)
    col_mass = pi.sum(axis=0)
    mapped = col_mass > 0
    if mapped.sum() < 3:
        return {"expr_corr": float("nan"), "n_genes": 0, "n_scored": int(mapped.sum())}

    pred = np.zeros_like(B)
    pred[mapped] = (pi[:, mapped].T @ A) / col_mass[mapped, None]

    Bm, Pm = B[mapped], pred[mapped]
    if B.shape[1] > max_genes:
        keep = np.argsort(Bm.var(axis=0))[::-1][:max_genes]
        Bm, Pm = Bm[:, keep], Pm[:, keep]

    corrs = []
    for g in range(Bm.shape[1]):
        b, p = Bm[:, g], Pm[:, g]
        if b.std() > 1e-12 and p.std() > 1e-12:
            corrs.append(np.corrcoef(b, p)[0, 1])
    return {
        "expr_corr": float(np.nanmean(corrs)) if corrs else float("nan"),
        "n_genes": int(len(corrs)),
        "n_scored": int(mapped.sum()),
    }


# ----------------------------------------------------------------------------
# convenience: the full battery
# ----------------------------------------------------------------------------

def evaluate_alignment(
    pi: np.ndarray,
    sim,
    parent,
    *,
    spatial_key: str = "spatial",
    label_key: str = "cell_type_annot",
    sim_axis: int = 0,
    k_coherence: int = 15,
    include_expression: bool = True,
) -> dict:
    """
    Run the whole metric battery on one aligned (sim, parent) pair.

    Registration is the PRIMARY signal for weight selection; the rest are
    reported validation metrics. Label transfer uses the *clean* labels
    (``obs[label_key + '_clean']`` when present) so it reflects true correspondence
    rather than the deliberately-corrupted annotation.

    Returns a flat dict prefixed by metric family (e.g. ``reg_soft_corr_mass``,
    ``coherence``, ``ltari``, ``expr_corr``).
    """
    pi = np.asarray(pi, dtype=np.float64)
    pi_oriented = pi.T if sim_axis == 1 else pi  # rows=sim, cols=parent

    out: dict = {}
    reg = registration_scores(pi, sim, parent, spatial_key=spatial_key, sim_axis=sim_axis)
    out.update({f"reg_{k}": v for k, v in reg.items()})

    coords_sim = np.asarray(sim.obsm[spatial_key], dtype=np.float64)[:, :2]
    coords_par = np.asarray(parent.obsm[spatial_key], dtype=np.float64)[:, :2]
    coh = spatial_coherence(pi_oriented, coords_sim, coords_par, k=k_coherence)
    out["coherence"] = coh["coherence"]
    out["coherence_n_mapped"] = coh["n_mapped"]

    def _clean_labels(adata):
        key = label_key + "_clean"
        col = key if key in adata.obs.columns else label_key
        return adata.obs[col].astype(str).to_numpy() if col in adata.obs.columns else None

    lab_sim, lab_par = _clean_labels(sim), _clean_labels(parent)
    if lab_sim is not None and lab_par is not None:
        # transfer sim(source) -> parent(target); skip nothing on the parent side
        lt = label_transfer_scores(pi_oriented, lab_sim, lab_par)
        out["ltari"] = lt["ari"]
        out["lt_accuracy"] = lt["accuracy"]

    if include_expression:
        ec = expression_transfer_corr(pi_oriented, sim.X, parent.X)
        out["expr_corr"] = ec["expr_corr"]

    return out
