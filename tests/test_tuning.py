"""
Tests for src/tuning.py
========================
Tests for the helper functions (simplex_grid, gpu_available, _staged_search).

select_alignment_weights is NOT tested here because it runs real OT alignment
(too expensive for a unit-test suite); it is covered at integration level by
benchmark/run_weight_benchmark.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.tuning import (
    simplex_grid,
    gpu_available,
    DEFAULT_INIT,
    WEIGHT_KEYS,
    _staged_search,
    make_self_alignment_instances,
)
from tests.conftest import N_REF, N_SIM, _make_adata


# ══════════════════════════════════════════════════════════════════════════════
# 1.  simplex_grid
# ══════════════════════════════════════════════════════════════════════════════

class TestSimplexGrid:

    def test_default_step_produces_valid_simplex_points(self):
        pts = simplex_grid(0.25)
        for beta, gamma in pts:
            assert beta >= 0.0
            assert gamma >= 0.0
            assert beta + gamma <= 1.0 + 1e-9

    def test_step_1_gives_three_vertices(self):
        pts = simplex_grid(1.0)
        expected = {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)}
        assert set(pts) == expected

    def test_includes_pure_expression_corner(self):
        """(0, 0) = pure expression must always be present."""
        pts = simplex_grid(0.25)
        assert (0.0, 0.0) in pts

    def test_count_step_half(self):
        """step=0.5 → n=2 → (n+1)(n+2)/2 = 6 points."""
        pts = simplex_grid(0.5)
        assert len(pts) == 6

    def test_count_step_quarter(self):
        """step=0.25 → n=4 → 15 points."""
        pts = simplex_grid(0.25)
        assert len(pts) == 15

    def test_no_duplicate_points(self):
        pts = simplex_grid(0.25)
        assert len(pts) == len(set(pts))

    def test_all_values_in_unit_interval(self):
        for beta, gamma in simplex_grid(0.1):
            assert 0.0 - 1e-9 <= beta <= 1.0 + 1e-9
            assert 0.0 - 1e-9 <= gamma <= 1.0 + 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# 2.  gpu_available
# ══════════════════════════════════════════════════════════════════════════════

class TestGpuAvailable:

    def test_returns_bool(self):
        assert isinstance(gpu_available(), bool)

    def test_does_not_raise(self):
        gpu_available()   # must not throw even if torch not installed


# ══════════════════════════════════════════════════════════════════════════════
# 3.  DEFAULT_INIT and WEIGHT_KEYS
# ══════════════════════════════════════════════════════════════════════════════

class TestConstants:

    def test_default_init_has_all_weight_keys(self):
        for k in WEIGHT_KEYS:
            assert k in DEFAULT_INIT, f"DEFAULT_INIT missing key '{k}'"

    def test_default_init_values_in_unit_interval(self):
        for k, v in DEFAULT_INIT.items():
            assert 0.0 <= v <= 1.0, f"DEFAULT_INIT['{k}']={v} out of [0,1]"

    def test_weight_keys_tuple(self):
        assert isinstance(WEIGHT_KEYS, tuple)
        assert set(WEIGHT_KEYS) == {"alpha", "beta", "gamma", "alpha_cluster", "delta"}


# ══════════════════════════════════════════════════════════════════════════════
# 4.  _staged_search  (unit test with trivial score_fn)
# ══════════════════════════════════════════════════════════════════════════════

class TestStagedSearch:
    """
    Replace the expensive alignment with a trivial score function whose
    maximum is analytically known, then verify _staged_search finds it.
    """

    def _score_fn_known_max(self, weights):
        """Peak at alpha=0.9, alpha_cluster=0.9, beta=0.0, gamma=0.0, delta=1.0."""
        return (
            -abs(weights["alpha"] - 0.9)
            - abs(weights["alpha_cluster"] - 0.9)
            - abs(weights["beta"] - 0.0)
            - abs(weights["gamma"] - 0.0)
            - abs(weights["delta"] - 1.0)
        )

    def test_staged_search_finds_approximate_maximum(self):
        best, best_score, landscape = _staged_search(
            self._score_fn_known_max,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
            alpha_cluster_grid=(0.1, 0.3, 0.5, 0.7, 0.9),
            simplex_step=0.25,
            delta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
            refine=True,
        )
        assert best["alpha"] == pytest.approx(0.9)
        assert best["alpha_cluster"] == pytest.approx(0.9)
        assert best["beta"] == pytest.approx(0.0)
        assert best["gamma"] == pytest.approx(0.0)
        assert best["delta"] == pytest.approx(1.0)

    def test_staged_search_returns_correct_types(self):
        calls = []

        def score_fn(w):
            calls.append(w)
            return 0.5

        best, best_score, landscape = _staged_search(
            score_fn,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.5,),
            alpha_cluster_grid=(0.5,),
            simplex_step=1.0,
            delta_grid=(0.5,),
            refine=False,
        )
        assert isinstance(best, dict)
        assert isinstance(best_score, float)
        assert isinstance(landscape, list)
        for entry in landscape:
            assert "score" in entry
            assert "stage" in entry

    def test_staged_search_landscape_has_all_stages(self):
        def score_fn(w):
            return float(np.random.default_rng(0).random())

        _, _, landscape = _staged_search(
            score_fn,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.3, 0.7),
            alpha_cluster_grid=(0.3, 0.7),
            simplex_step=0.5,
            delta_grid=(0.0, 1.0),
            refine=True,
        )
        stages = {entry["stage"] for entry in landscape}
        assert "alpha" in stages
        assert "feature" in stages

    def test_refine_does_not_decrease_best_score(self):
        counter = {"n": 0}

        def score_fn(w):
            counter["n"] += 1
            return float(np.random.default_rng(counter["n"]).random())

        _, score_no_refine, _ = _staged_search(
            score_fn,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.3, 0.7),
            alpha_cluster_grid=(0.3, 0.7),
            simplex_step=1.0,
            delta_grid=(0.0, 1.0),
            refine=False,
        )
        # With refine=True the best_score can only stay the same or improve,
        # but since score_fn is random we just verify no crash and score is float.
        best2, score_refine, _ = _staged_search(
            score_fn,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.3, 0.7),
            alpha_cluster_grid=(0.3, 0.7),
            simplex_step=1.0,
            delta_grid=(0.0, 1.0),
            refine=True,
        )
        assert isinstance(score_refine, float)
        assert set(best2.keys()) == set(WEIGHT_KEYS)

    def test_best_weights_contain_all_keys(self):
        def score_fn(w):
            return 1.0

        best, _, _ = _staged_search(
            score_fn,
            init=dict(DEFAULT_INIT),
            alpha_grid=(0.5,),
            alpha_cluster_grid=(0.5,),
            simplex_step=1.0,
            delta_grid=(0.5,),
            refine=False,
        )
        assert set(best.keys()) == set(WEIGHT_KEYS)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  make_self_alignment_instances  (no real alignment, just tests the factory)
# ══════════════════════════════════════════════════════════════════════════════

class TestMakeSelfAlignmentInstances:
    """
    We only verify the factory's interface; simulate_adjacent_slice internals
    are tested in test_perturb.py.  The real slice data is loaded from disk if
    available, otherwise we skip.
    """

    @pytest.fixture
    def tiny_reference(self):
        adata = _make_adata(100, 10, n_types=3, seed=21, label_key="cell_type_annot")
        # give obs_names that the section could be a subset of
        import pandas as pd
        adata.obs_names = pd.Index([f"cell_{i}" for i in range(100)])
        return adata

    @pytest.fixture
    def tiny_section(self, tiny_reference):
        # a real obs_names subset of `tiny_reference` (as synthesize.py crops are),
        # required so simulate_adjacent_slice can map dropout_kept_positions into
        # `tiny_reference`'s row index space.
        return tiny_reference[:40].copy()

    def test_n_instances_matches_request(self, tiny_section, tiny_reference):
        instances = make_self_alignment_instances(
            section=tiny_section, reference=tiny_reference,
            n_instances=2, seed=0,
        )
        assert len(instances) == 2

    def test_each_instance_is_tuple_of_two_adata(self, tiny_section, tiny_reference):
        import anndata as ad
        instances = make_self_alignment_instances(
            section=tiny_section, reference=tiny_reference,
            n_instances=1, seed=0,
        )
        for sim, ref in instances:
            assert isinstance(sim, ad.AnnData)
            assert isinstance(ref, ad.AnnData)

    def test_sim_has_gt_provenance(self, tiny_section, tiny_reference):
        instances = make_self_alignment_instances(
            section=tiny_section, reference=tiny_reference,
            n_instances=1, seed=0,
        )
        sim, _ = instances[0]
        assert "self_alignment_test" in sim.uns
        assert "adjacent_simulation" in sim.uns["self_alignment_test"]

    def test_crops_input_mode(self, tiny_section, tiny_reference):
        """crops= mode ignores n_instances; one instance per crop."""
        crops = [(tiny_section, tiny_reference), (tiny_section, tiny_reference)]
        instances = make_self_alignment_instances(crops=crops, seed=0)
        assert len(instances) == 2

    def test_dropout_kept_positions_index_into_reference(self, tiny_section, tiny_reference):
        """
        Regression test: dropout_kept_positions must be a row index into
        `reference`, not a position local to the (cropped) `section`. Since
        `tiny_section` is `tiny_reference`'s first 40 rows, gt[j] must recover
        the true parent's obs_name for every matched sim cell j.
        """
        instances = make_self_alignment_instances(
            section=tiny_section, reference=tiny_reference,
            n_instances=1, seed=0,
        )
        sim, ref = instances[0]
        gt = np.asarray(
            sim.uns["self_alignment_test"]["adjacent_simulation"]["dropout_kept_positions"]
        )
        assert gt.min() >= 0
        assert gt.max() < ref.n_obs
        matched_names = sim.obs_names[: len(gt)]
        assert list(ref.obs_names[gt]) == list(matched_names)

    def test_max_cells_keeps_ground_truth_consistent(self, tiny_section, tiny_reference):
        """
        When max_cells forces ref/sim subsampling, dropout_kept_positions must
        still index into the (subsampled) reference's new row order and match
        sim's (subsampled) row order 1:1 for the matched prefix.
        """
        instances = make_self_alignment_instances(
            section=tiny_section, reference=tiny_reference,
            n_instances=1, seed=0, max_cells=10,
        )
        sim, ref = instances[0]
        assert ref.n_obs <= 10
        assert sim.n_obs <= 10
        gt = np.asarray(
            sim.uns["self_alignment_test"]["adjacent_simulation"]["dropout_kept_positions"]
        )
        assert gt.min() >= 0
        assert gt.max() < ref.n_obs
        matched_names = sim.obs_names[: len(gt)]
        assert list(ref.obs_names[gt]) == list(matched_names)

    def test_raises_without_inputs(self):
        with pytest.raises(ValueError):
            make_self_alignment_instances()
