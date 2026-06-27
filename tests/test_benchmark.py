"""
Tests for benchmark/run_weight_benchmark.py
============================================
Only reg_m_sensitivity_sweep is tested here; run_weight_benchmark itself runs
real OT alignment (too expensive for unit tests) and is covered at integration
level by running the benchmark script directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from benchmark.run_weight_benchmark import reg_m_sensitivity_sweep, _is_stable_pi
from src.tuning import DEFAULT_INIT, WEIGHT_KEYS
from tests.conftest import _make_adata


# ---------------------------------------------------------------------------
# Shared stub aligner: returns uniform pi without running any OT
# ---------------------------------------------------------------------------

def _stub_aligner(sliceA, sliceB, **kwargs):
    nA, nB = sliceA.n_obs, sliceB.n_obs
    return np.full((nA, nB), 1.0 / (nA * nB), dtype=np.float64)


_BEST = dict(DEFAULT_INIT)
_REG_M_GRID = (0.1, 1.0, 10.0)


@pytest.fixture
def tiny_instances():
    sliceA = _make_adata(20, 8, seed=50)
    sliceB = _make_adata(20, 8, seed=51)
    return [(sliceA, sliceB)]


# ---------------------------------------------------------------------------
# TestRegMSensitivitySweep
# ---------------------------------------------------------------------------

class TestRegMSensitivitySweep:

    def test_returns_one_row_per_grid_value(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        assert len(rows) == len(_REG_M_GRID)

    def test_each_row_has_reg_m_key(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert "reg_m" in row

    def test_reg_m_values_match_grid(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        returned = [row["reg_m"] for row in rows]
        assert returned == pytest.approx(list(_REG_M_GRID))

    def test_each_row_has_n_ok(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert "n_ok" in row
            assert row["n_ok"] >= 0

    def test_n_ok_equals_instance_count_when_aligner_succeeds(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert row["n_ok"] == len(tiny_instances)

    def test_base_align_kwargs_not_mutated(self, tiny_instances):
        base = {"use_gpu": False, "verbose": False}
        original = dict(base)
        reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs=base,
        )
        assert base == original

    def test_reg_m_passed_to_aligner(self, tiny_instances):
        """Verify that each sweep iteration actually receives the correct reg_m."""
        seen_reg_m = []

        def _recording_aligner(sliceA, sliceB, **kwargs):
            seen_reg_m.append(kwargs.get("reg_m"))
            nA, nB = sliceA.n_obs, sliceB.n_obs
            return np.full((nA, nB), 1.0 / (nA * nB), dtype=np.float64)

        reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_recording_aligner,
            base_align_kwargs={},
        )
        assert seen_reg_m == pytest.approx(list(_REG_M_GRID))

    def test_empty_instances_returns_rows_with_n_ok_zero(self):
        rows = reg_m_sensitivity_sweep(
            [], _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        assert len(rows) == len(_REG_M_GRID)
        for row in rows:
            assert row["n_ok"] == 0
            assert "reg_m" in row

    def test_single_grid_value(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, (1.0,), _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        assert len(rows) == 1
        assert rows[0]["reg_m"] == pytest.approx(1.0)

    def test_numerically_stable_flag_present(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert "numerically_stable" in row

    def test_n_unstable_present(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert "n_unstable" in row

    def test_stable_aligner_marks_all_stable(self, tiny_instances):
        rows = reg_m_sensitivity_sweep(
            tiny_instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner,
            base_align_kwargs={},
        )
        for row in rows:
            assert row["numerically_stable"] is True
            assert row["n_unstable"] == 0

    def test_nan_pi_counts_as_unstable(self, tiny_instances):
        def _nan_aligner(sliceA, sliceB, **kwargs):
            nA, nB = sliceA.n_obs, sliceB.n_obs
            pi = np.full((nA, nB), np.nan)
            return pi

        rows = reg_m_sensitivity_sweep(
            tiny_instances, (1.0,), _BEST,
            aligner=_nan_aligner,
            base_align_kwargs={},
        )
        assert rows[0]["numerically_stable"] is False
        assert rows[0]["n_unstable"] == len(tiny_instances)
        assert rows[0]["n_ok"] == 0

    def test_zero_mass_pi_counts_as_unstable(self, tiny_instances):
        def _zero_aligner(sliceA, sliceB, **kwargs):
            nA, nB = sliceA.n_obs, sliceB.n_obs
            return np.zeros((nA, nB))

        rows = reg_m_sensitivity_sweep(
            tiny_instances, (1.0,), _BEST,
            aligner=_zero_aligner,
            base_align_kwargs={},
        )
        assert rows[0]["numerically_stable"] is False
        assert rows[0]["n_unstable"] == len(tiny_instances)

    def test_mixed_stability_partial_n_ok(self):
        """One instance returns valid pi, another returns NaN — n_ok=1, n_unstable=1."""
        sliceA = _make_adata(20, 8, seed=60)
        sliceB = _make_adata(20, 8, seed=61)
        call_count = {"n": 0}

        def _mixed_aligner(sliceA, sliceB, **kwargs):
            call_count["n"] += 1
            nA, nB = sliceA.n_obs, sliceB.n_obs
            if call_count["n"] % 2 == 0:
                return np.full((nA, nB), np.nan)
            return np.full((nA, nB), 1.0 / (nA * nB))

        instances = [(sliceA, sliceB), (sliceA, sliceB)]
        rows = reg_m_sensitivity_sweep(
            instances, (1.0,), _BEST,
            aligner=_mixed_aligner,
            base_align_kwargs={},
        )
        assert rows[0]["n_ok"] == 1
        assert rows[0]["n_unstable"] == 1
        assert rows[0]["numerically_stable"] is False


# ---------------------------------------------------------------------------
# TestIsStablePi
# ---------------------------------------------------------------------------

class TestIsStablePi:

    def test_uniform_pi_is_stable(self):
        pi = np.full((10, 10), 1.0 / 100)
        assert _is_stable_pi(pi) is True

    def test_nan_pi_is_unstable(self):
        pi = np.full((5, 5), np.nan)
        assert _is_stable_pi(pi) is False

    def test_inf_pi_is_unstable(self):
        pi = np.zeros((5, 5))
        pi[0, 0] = np.inf
        assert _is_stable_pi(pi) is False

    def test_zero_mass_pi_is_unstable(self):
        assert _is_stable_pi(np.zeros((5, 5))) is False

    def test_below_threshold_is_unstable(self):
        pi = np.full((10, 10), 1e-6)  # sum ≈ 1e-4, well below default 0.05
        assert _is_stable_pi(pi) is False

    def test_custom_threshold(self):
        pi = np.full((10, 10), 0.001)  # sum = 0.1
        assert _is_stable_pi(pi, min_total_mass=0.05) is True
        assert _is_stable_pi(pi, min_total_mass=0.2) is False
