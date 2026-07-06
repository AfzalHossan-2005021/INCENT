"""
Integration tests for the ablation-enabling kwargs threaded through
src/core.py::hierarchical_pairwise_align: ``mesoregion_method``,
``use_geometric_admissibility``, ``balanced``, ``skip_shared_region_detection``.

Unlike tests/test_clustering.py and tests/test_hierarchical_ablation.py (which
exercise the underlying pure-numpy logic in isolation and run in well under a
second), these run the *real* hierarchical OT pipeline end-to-end on a small
synthetic tissue, so each case takes a few seconds on CPU. They exist to catch
wiring mistakes (a kwarg not actually reaching the function that uses it) that
the isolated unit tests cannot see.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import anndata as ad

from src.core import hierarchical_pairwise_align, pairwise_align
from src.perturb import simulate_adjacent_slice
from src.utils import compute_joint_pca


def _make_synthetic_tissue(n_side=16, spacing=5.0, jitter=0.8, seed=0):
    """A small tissue with four spatially-clustered, expression-distinct cell types."""
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64) * spacing
    coords += rng.normal(0, jitter, coords.shape)
    n_cells = coords.shape[0]

    half = n_side * spacing / 2.0
    cell_types = np.where(coords[:, 0] < half, "A", "B")
    cell_types = np.where(coords[:, 1] < half, cell_types, np.char.add(cell_types, "2"))

    n_genes = 10
    type_means = {ct: rng.normal(0, 1, n_genes) for ct in np.unique(cell_types)}
    X = np.stack([type_means[ct] for ct in cell_types]) + rng.normal(0, 0.3, (n_cells, n_genes))
    X = np.clip(X, 0, None).astype(np.float32)

    reference = ad.AnnData(X=X)
    reference.obsm["spatial"] = coords
    reference.obs["cell_type_annot"] = cell_types
    reference.obsm["X_pca"] = X[:, :6].astype(np.float32)
    reference.obs_names = [f"cell_{i}" for i in range(n_cells)]
    return reference


@pytest.fixture(scope="module")
def sim_ref_pair():
    """A small (sim, reference) self-alignment pair, generated once and reused."""
    reference = _make_synthetic_tissue(n_side=16, seed=0)
    coords = np.asarray(reference.obsm["spatial"])
    crop_mask = coords[:, 0] < coords[:, 0].max() * 0.65
    section = reference[crop_mask].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sim = simulate_adjacent_slice(
            section, reference=reference, seed=123,
            dropout_rate=0.05, rotation_range=(-10, 10), expr_alpha=0.5,
            label_flip_rate=0.02, birth_rate=0.02,
        )
        sim, ref2 = compute_joint_pca(sim, reference.copy())
    return sim, ref2


BASE_KWARGS = dict(
    alpha=0.5, beta=0.5, gamma=0.25, alpha_cluster=0.5, delta=0.5,
    coarsen_scale=15.0, numItermax=2000,
    use_gpu=False, verbose=False, gpu_verbose=False, visualize_clusters=False,
)


def _assert_valid_pi(pi, expected_shape):
    assert pi.shape == expected_shape
    assert np.isfinite(pi).all()
    assert pi.sum() == pytest.approx(1.0, abs=1e-3)


class TestSkipSharedRegionDetection:

    def test_returns_full_size_valid_coupling(self, sim_ref_pair):
        sim, ref = sim_ref_pair
        pi = hierarchical_pairwise_align(
            sim, ref, skip_shared_region_detection=True, **BASE_KWARGS,
        )
        _assert_valid_pi(pi, (sim.n_obs, ref.n_obs))

    def test_matches_plain_pairwise_align_shape(self, sim_ref_pair):
        """Both bypass the macro-section/shadow restriction, so both must
        return a full (n_sim, n_ref) coupling."""
        sim, ref = sim_ref_pair
        pi_skip = hierarchical_pairwise_align(
            sim, ref, skip_shared_region_detection=True, **BASE_KWARGS,
        )
        pi_flat = pairwise_align(
            sim, ref, alpha=0.5, beta=0.5, gamma=0.25,
            numItermax=2000, use_gpu=False, verbose=False, gpu_verbose=False,
        )
        assert pi_skip.shape == pi_flat.shape


class TestMesoregionMethodWiring:

    @pytest.mark.parametrize("method", ["grid", "random"])
    def test_alternate_mesoregion_method_runs_end_to_end(self, sim_ref_pair, method):
        sim, ref = sim_ref_pair
        pi = hierarchical_pairwise_align(
            sim, ref, mesoregion_method=method, mesoregion_seed=1, **BASE_KWARGS,
        )
        assert np.isfinite(pi).all()
        assert pi.shape == (sim.n_obs, ref.n_obs)


class TestBalancedWiring:

    def test_balanced_true_runs_end_to_end(self, sim_ref_pair):
        sim, ref = sim_ref_pair
        pi = hierarchical_pairwise_align(sim, ref, balanced=True, **BASE_KWARGS)
        assert np.isfinite(pi).all()
        assert pi.shape == (sim.n_obs, ref.n_obs)


class TestGeometricAdmissibilityWiring:

    def test_disabled_runs_end_to_end(self, sim_ref_pair):
        sim, ref = sim_ref_pair
        pi = hierarchical_pairwise_align(
            sim, ref, use_geometric_admissibility=False, **BASE_KWARGS,
        )
        assert np.isfinite(pi).all()
        assert pi.shape == (sim.n_obs, ref.n_obs)
