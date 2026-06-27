"""
Shared fixtures for the INCENT test suite.

All fixtures are pure-Python / NumPy — no real alignment is run; pi matrices
are hand-crafted so that expected metric values are analytically known.
"""

from __future__ import annotations

import numpy as np
import pytest
import anndata as ad


# ── sizes ─────────────────────────────────────────────────────────────────────
N_REF = 30   # reference slice cells
N_SIM = 20   # simulated slice cells  (subset after dropout)
N_GENES = 15
RNG = np.random.default_rng(42)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_adata(n_cells: int, n_genes: int, n_types: int = 3,
                seed: int = 0, label_key: str = "cell_type_annot") -> ad.AnnData:
    rng = np.random.default_rng(seed)
    X = rng.random((n_cells, n_genes)).astype(np.float32)
    spatial = rng.random((n_cells, 2)).astype(np.float64) * 100.0
    labels = (rng.integers(0, n_types, size=n_cells)).astype(str)
    adata = ad.AnnData(X=X)
    adata.obsm["spatial"] = spatial
    adata.obs[label_key] = labels
    adata.obsm["X_pca"] = rng.random((n_cells, 10)).astype(np.float32)
    return adata


def _perfect_pi(n_sim: int, n_ref: int, gt: np.ndarray) -> np.ndarray:
    """
    Coupling in the evaluate_alignment(sim_axis=0) convention:
    shape (n_sim, n_ref); pi[j, gt[j]] = 1/n_ref.
    For foscttm() direct calls (expects n_ref × n_sim) use .T.
    """
    pi = np.zeros((n_sim, n_ref), dtype=np.float64)
    for j, i in enumerate(gt):
        pi[j, i] = 1.0 / n_ref
    return pi


def _uniform_pi(n_sim: int, n_ref: int) -> np.ndarray:
    """Uniform coupling in (n_sim, n_ref) shape. Use .T for foscttm() direct calls."""
    return np.full((n_sim, n_ref), 1.0 / (n_sim * n_ref), dtype=np.float64)


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def gt_indices():
    """gt_indices[j] = reference row for simulated cell j (no repeats, sorted)."""
    rng = np.random.default_rng(7)
    return np.sort(rng.choice(N_REF, size=N_SIM, replace=False)).astype(np.int64)


@pytest.fixture
def perfect_pi(gt_indices):
    return _perfect_pi(N_SIM, N_REF, gt_indices)


@pytest.fixture
def uniform_pi():
    return _uniform_pi(N_SIM, N_REF)


@pytest.fixture
def slice_ref():
    return _make_adata(N_REF, N_GENES, seed=1)


@pytest.fixture
def slice_sim(gt_indices, slice_ref):
    """
    Simulated slice: spatial coords near the true ref positions + small jitter.
    Labels copied from ref's gt rows so that perfect_pi gives LTA = 1.0.
    Carries ground-truth in .uns as simulate_adjacent_slice would write it.
    """
    rng = np.random.default_rng(99)
    adata = _make_adata(N_SIM, N_GENES, seed=2)

    # place sim cells near their true ref counterparts
    ref_coords = slice_ref.obsm["spatial"][gt_indices]
    adata.obsm["spatial"] = ref_coords + rng.normal(0, 1.0, ref_coords.shape)

    # give sim cells the same labels as their true ref matches
    adata.obs["cell_type_annot"] = (
        slice_ref.obs["cell_type_annot"].values[gt_indices]
    )
    # also write _clean labels (matching)
    adata.obs["cell_type_annot_clean"] = adata.obs["cell_type_annot"].copy()

    # ground-truth provenance expected by evaluate_alignment / _extract_gt_indices
    adata.uns["self_alignment_test"] = {
        "adjacent_simulation": {
            "dropout_kept_positions": gt_indices.copy(),
        }
    }
    return adata
