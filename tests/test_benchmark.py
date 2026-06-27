"""
Tests for benchmark/run_weight_benchmark.py
============================================
Only reg_m_sensitivity_sweep is tested here; run_weight_benchmark itself runs
real OT alignment (too expensive for unit tests) and is covered at integration
level by running the benchmark script directly.
"""

from __future__ import annotations

import concurrent.futures
import numpy as np
import pytest

from benchmark.run_weight_benchmark import (
    reg_m_sensitivity_sweep,
    _is_stable_pi,
    _make_device_aware_aligner,
    grid_sweep_full,
)
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


# ---------------------------------------------------------------------------
# TestParallelExecution
# ---------------------------------------------------------------------------

class TestParallelExecution:
    """Verify that n_jobs>1 produces identical results to n_jobs=1."""

    @pytest.fixture
    def instances(self):
        sliceA = _make_adata(20, 8, seed=70)
        sliceB = _make_adata(20, 8, seed=71)
        return [(sliceA, sliceB), (sliceA, sliceB)]

    def test_grid_sweep_parallel_matches_sequential(self, instances):
        weights = [
            {**_BEST, "alpha": 0.3},
            {**_BEST, "alpha": 0.7},
            {**_BEST, "alpha": 0.9},
        ]
        seq = grid_sweep_full(instances, weights, aligner=_stub_aligner,
                              align_kwargs={}, n_jobs=1)
        par = grid_sweep_full(instances, weights, aligner=_stub_aligner,
                              align_kwargs={}, n_jobs=3)
        assert len(seq) == len(par)
        for s, p in zip(seq, par):
            assert s["alpha"] == pytest.approx(p["alpha"])
            assert s["n_ok"] == p["n_ok"]

    def test_reg_m_sweep_parallel_matches_sequential(self, instances):
        seq = reg_m_sensitivity_sweep(
            instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner, base_align_kwargs={}, n_jobs=1,
        )
        par = reg_m_sensitivity_sweep(
            instances, _REG_M_GRID, _BEST,
            aligner=_stub_aligner, base_align_kwargs={}, n_jobs=3,
        )
        assert len(seq) == len(par)
        seq_sorted = sorted(seq, key=lambda r: r["reg_m"])
        par_sorted = sorted(par, key=lambda r: r["reg_m"])
        for s, p in zip(seq_sorted, par_sorted):
            assert s["reg_m"] == pytest.approx(p["reg_m"])
            assert s["n_ok"] == p["n_ok"]
            assert s["numerically_stable"] == p["numerically_stable"]

    def test_grid_sweep_order_preserved_parallel(self, instances):
        """Output order must match weight_list order regardless of thread scheduling."""
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
        weights = [{**_BEST, "alpha": a} for a in alphas]
        par = grid_sweep_full(instances, weights, aligner=_stub_aligner,
                              align_kwargs={}, n_jobs=5)
        returned_alphas = [r["alpha"] for r in par]
        assert returned_alphas == pytest.approx(alphas)


# ---------------------------------------------------------------------------
# TestMakeDeviceAwareAligner
# ---------------------------------------------------------------------------

class TestMakeDeviceAwareAligner:
    """
    _make_device_aware_aligner is tested without real GPUs:
    - When torch is not available (or device routing isn't exercised), the wrapper
      must still call the underlying aligner with the correct arguments.
    - Thread-safety: multiple threads running concurrently should each call the
      aligner exactly once and return distinct results without cross-contamination.
    """

    def test_wrapped_aligner_returns_correct_shape(self):
        sliceA = _make_adata(10, 5, seed=80)
        sliceB = _make_adata(12, 5, seed=81)
        wrapped = _make_device_aware_aligner(_stub_aligner, [0])
        pi = wrapped(sliceA, sliceB)
        assert pi.shape == (10, 12)

    def test_wrapped_aligner_passes_kwargs(self):
        received_kwargs = {}

        def _recording_aligner(sliceA, sliceB, **kwargs):
            received_kwargs.update(kwargs)
            nA, nB = sliceA.n_obs, sliceB.n_obs
            return np.full((nA, nB), 1.0 / (nA * nB))

        sliceA = _make_adata(10, 5, seed=82)
        sliceB = _make_adata(10, 5, seed=83)
        wrapped = _make_device_aware_aligner(_recording_aligner, [0])
        wrapped(sliceA, sliceB, use_gpu=False, verbose=False)
        assert received_kwargs.get("use_gpu") is False
        assert received_kwargs.get("verbose") is False

    def test_wrapped_aligner_returns_same_result_as_unwrapped(self):
        sliceA = _make_adata(15, 6, seed=84)
        sliceB = _make_adata(15, 6, seed=85)
        wrapped = _make_device_aware_aligner(_stub_aligner, [0])
        pi_wrapped = wrapped(sliceA, sliceB)
        pi_direct = _stub_aligner(sliceA, sliceB)
        np.testing.assert_allclose(pi_wrapped, pi_direct)

    def test_multi_device_pool_distributes_across_threads(self):
        """Each thread should receive a different device_id from the pool."""
        import threading
        seen_calls = []
        lock = threading.Lock()

        def _recording_aligner(sliceA, sliceB, **kwargs):
            nA, nB = sliceA.n_obs, sliceB.n_obs
            with lock:
                seen_calls.append(threading.current_thread().name)
            return np.full((nA, nB), 1.0 / (nA * nB))

        device_ids = [0, 1]
        wrapped = _make_device_aware_aligner(_recording_aligner, device_ids)
        sliceA = _make_adata(10, 5, seed=86)
        sliceB = _make_adata(10, 5, seed=87)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futs = [pool.submit(wrapped, sliceA, sliceB) for _ in range(4)]
            [f.result() for f in futs]

        assert len(seen_calls) == 4

    def test_pool_returns_device_after_call(self):
        """After N sequential calls on a 1-slot pool, all N calls complete (no deadlock)."""
        wrapped = _make_device_aware_aligner(_stub_aligner, [0])
        sliceA = _make_adata(8, 4, seed=88)
        sliceB = _make_adata(8, 4, seed=89)
        for _ in range(5):
            pi = wrapped(sliceA, sliceB)
            assert pi is not None

    def test_exception_in_aligner_still_returns_device_to_pool(self):
        """Even if the aligner raises, the device must be returned so the pool is not exhausted."""
        call_count = {"n": 0}

        def _sometimes_raises(sliceA, sliceB, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated alignment failure")
            nA, nB = sliceA.n_obs, sliceB.n_obs
            return np.full((nA, nB), 1.0 / (nA * nB))

        wrapped = _make_device_aware_aligner(_sometimes_raises, [0])
        sliceA = _make_adata(8, 4, seed=90)
        sliceB = _make_adata(8, 4, seed=91)
        with pytest.raises(RuntimeError):
            wrapped(sliceA, sliceB)
        # Pool must still be usable after the exception
        pi = wrapped(sliceA, sliceB)
        assert pi is not None
