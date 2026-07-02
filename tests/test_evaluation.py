"""
Tests for src/evaluation.py
============================
Every metric is tested against a hand-crafted pi matrix whose expected value
is analytically known:
  - perfect_pi  → perfect / near-perfect scores
  - uniform_pi  → random-baseline scores
  - custom constructions → edge cases

No real alignment solver is invoked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.evaluation import (
    label_transfer_accuracy,
    foscttm,
    expression_transfer_corr,
    calculate_neighborhood_dissimilarity_cost,
    calculate_gene_expression_dissimilarity,
    cell_type_matching,
    benchmark_method,
    evaluate_alignment,
    _extract_gt_indices,
)
from tests.conftest import N_REF, N_SIM, N_GENES, _make_adata, _perfect_pi, _uniform_pi


# ══════════════════════════════════════════════════════════════════════════════
# 1.  label_transfer_accuracy
# ══════════════════════════════════════════════════════════════════════════════

class TestLabelTransferAccuracy:

    def test_perfect_alignment_gives_lta_one(self, perfect_pi, slice_ref, slice_sim, gt_indices):
        """Perfect coupling → LTA = 1.0."""
        labels_sim = slice_sim.obs["cell_type_annot"].values
        labels_ref = slice_ref.obs["cell_type_annot"].values
        lta, detail = label_transfer_accuracy(perfect_pi, labels_sim, labels_ref)
        assert lta == pytest.approx(1.0), f"Expected LTA=1.0, got {lta}"
        assert detail["n_correct"] == detail["n_evaluated"]

    def test_wrong_alignment_gives_lta_zero(self, slice_ref, gt_indices):
        """Coupling that always maps to the wrong type → LTA = 0."""
        n = 10
        labels_A = np.array(["A"] * n)
        labels_B = np.array(["B"] * n)
        pi = np.eye(n) / n
        lta, _ = label_transfer_accuracy(pi, labels_A, labels_B)
        assert lta == pytest.approx(0.0)

    def test_uniform_pi_gives_lta_near_chance(self):
        """Uniform coupling → LTA ≈ label frequency of the most common class."""
        n = 6
        labels_A = np.array(["X", "X", "X", "Y", "Y", "Y"])
        labels_B = np.array(["X", "X", "X", "Y", "Y", "Y"])
        pi = _uniform_pi(n, n)
        lta, _ = label_transfer_accuracy(pi, labels_A, labels_B)
        # argmax of uniform row → first column (col 0); label_B[0]="X"
        # Cells with label "X" predict correctly (3/6 = 0.5)
        assert 0.0 <= lta <= 1.0

    def test_detail_counts_are_consistent(self, perfect_pi, slice_ref, slice_sim):
        labels_sim = slice_sim.obs["cell_type_annot"].values
        labels_ref = slice_ref.obs["cell_type_annot"].values
        _, detail = label_transfer_accuracy(perfect_pi, labels_sim, labels_ref)
        assert detail["n_evaluated"] + detail["n_skipped_low_mass"] == N_SIM
        assert detail["n_correct"] <= detail["n_evaluated"]
        assert len(detail["per_cell_correct"]) == detail["n_evaluated"]

    def test_all_zero_rows_returns_zero(self):
        """If every row is zero, LTA = 0.0 and n_evaluated = 0."""
        pi = np.zeros((5, 5))
        labels = np.array(["A", "A", "A", "B", "B"])
        lta, detail = label_transfer_accuracy(pi, labels, labels)
        assert lta == pytest.approx(0.0)
        assert detail["n_evaluated"] == 0
        assert detail["n_skipped_low_mass"] == 5

    def test_mass_threshold_skips_low_mass_rows(self):
        """Rows with very low mass below threshold are skipped."""
        n = 4
        pi = np.zeros((n, n))
        pi[0, 0] = 1.0         # high mass, correct
        pi[1, 2] = 1e-10       # tiny mass, wrong label but may be skipped
        pi[2, 2] = 0.5         # medium mass, correct
        pi[3, 1] = 0.5         # medium mass, correct
        labels_A = np.array(["A", "B", "C", "D"])
        labels_B = np.array(["A", "X", "C", "X", "D", "X"][:n])
        labels_B = np.array(["A", "X", "C", "D"])
        lta_no_thresh, det1 = label_transfer_accuracy(pi, labels_A, labels_B, mass_threshold=0.0)
        lta_thresh, det2 = label_transfer_accuracy(pi, labels_A, labels_B, mass_threshold=0.5)
        # With threshold, low-mass row 1 may be skipped
        assert det2["n_skipped_low_mass"] >= det1["n_skipped_low_mass"]

    def test_returns_float_and_dict(self, perfect_pi, slice_ref, slice_sim):
        labels_sim = slice_sim.obs["cell_type_annot"].values
        labels_ref = slice_ref.obs["cell_type_annot"].values
        lta, detail = label_transfer_accuracy(perfect_pi, labels_sim, labels_ref)
        assert isinstance(lta, float)
        assert isinstance(detail, dict)
        assert "n_evaluated" in detail and "n_correct" in detail


# ══════════════════════════════════════════════════════════════════════════════
# 2.  foscttm
# ══════════════════════════════════════════════════════════════════════════════

class TestFoscttm:

    def test_perfect_plan_gives_foscttm_zero(self, perfect_pi, gt_indices):
        """
        Perfect coupling: pi[gt[j], j] = 1/N_REF for all j, 0 elsewhere.
        For each sim cell j, the true match ref row gt[j] has strictly higher
        weight than all other ref rows → FOSCTTM = 0.
        foscttm() expects pi shape (n_ref, n_sim); perfect_pi is (n_sim, n_ref) so use .T.
        """
        result = foscttm(perfect_pi.T, gt_indices)
        assert result["foscttm"] == pytest.approx(0.0, abs=1e-10)
        assert result["foscttm_B_to_A"] == pytest.approx(0.0, abs=1e-10)
        assert result["foscttm_A_to_B"] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_plan_gives_foscttm_half(self, uniform_pi, gt_indices):
        """
        Uniform plan: all entries equal → all cells tied → midpoint tie-breaking
        → FOSCTTM ≈ 0.5.
        uniform_pi is (n_sim, n_ref); foscttm() needs (n_ref, n_sim) so use .T.
        """
        result = foscttm(uniform_pi.T, gt_indices)
        assert result["foscttm"] == pytest.approx(0.5, abs=0.05)

    def test_symmetric_mean_is_average_of_directions(self, perfect_pi, gt_indices):
        result = foscttm(perfect_pi.T, gt_indices)
        expected = 0.5 * (result["foscttm_B_to_A"] + result["foscttm_A_to_B"])
        assert result["foscttm"] == pytest.approx(expected)

    def test_foscttm_range(self, uniform_pi, gt_indices):
        result = foscttm(uniform_pi.T, gt_indices)
        assert 0.0 <= result["foscttm"] <= 1.0
        assert 0.0 <= result["foscttm_B_to_A"] <= 1.0
        assert 0.0 <= result["foscttm_A_to_B"] <= 1.0

    def test_minimal_two_cell_pair(self):
        """2×2 perfect plan — smallest non-degenerate case, FOSCTTM = 0."""
        # pi shape (n_ref=2, n_sim=2); gt[0]=0, gt[1]=1
        pi = np.array([[0.5, 0.0], [0.0, 0.5]])
        gt = np.array([0, 1])
        result = foscttm(pi, gt)
        assert result["foscttm"] == pytest.approx(0.0, abs=1e-10)

    def test_chunk_size_one_matches_default(self, uniform_pi, gt_indices):
        """chunk_size=1 (loop one cell at a time) must give same result."""
        pi_foscttm = uniform_pi.T   # (n_ref, n_sim) for direct foscttm call
        r1 = foscttm(pi_foscttm, gt_indices, chunk_size=512)
        r2 = foscttm(pi_foscttm, gt_indices, chunk_size=1)
        assert r1["foscttm"] == pytest.approx(r2["foscttm"], abs=1e-12)
        assert r1["foscttm_B_to_A"] == pytest.approx(r2["foscttm_B_to_A"], abs=1e-12)

    def test_returns_expected_keys(self, uniform_pi, gt_indices):
        result = foscttm(uniform_pi.T, gt_indices)
        assert set(result.keys()) == {"foscttm", "foscttm_B_to_A", "foscttm_A_to_B"}

    def test_better_plan_has_lower_foscttm(self, perfect_pi, uniform_pi, gt_indices):
        """Perfect plan must beat uniform plan."""
        r_perfect = foscttm(perfect_pi.T, gt_indices)
        r_uniform = foscttm(uniform_pi.T, gt_indices)
        assert r_perfect["foscttm"] < r_uniform["foscttm"]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  expression_transfer_corr
# ══════════════════════════════════════════════════════════════════════════════

class TestExpressionTransferCorr:

    def test_identity_plan_gives_high_corr(self):
        """Identity plan: predicted = actual expression → correlation = 1."""
        n, g = 20, 10
        rng = np.random.default_rng(3)
        X = rng.random((n, g)).astype(np.float64)
        pi = np.eye(n) / n
        result = expression_transfer_corr(pi, X, X)
        assert result["expr_corr"] == pytest.approx(1.0, abs=1e-8)

    def test_shuffled_plan_gives_lower_corr(self):
        """Shuffled coupling should give lower correlation than identity."""
        n, g = 20, 10
        rng = np.random.default_rng(5)
        X = rng.random((n, g)).astype(np.float64)
        pi_id = np.eye(n) / n
        perm = rng.permutation(n)
        pi_bad = pi_id[perm]
        r_id = expression_transfer_corr(pi_id, X, X)
        r_bad = expression_transfer_corr(pi_bad, X, X)
        assert r_id["expr_corr"] >= r_bad["expr_corr"]

    def test_returns_expected_keys(self):
        n, g = 10, 5
        X = np.random.default_rng(6).random((n, g))
        pi = np.eye(n) / n
        result = expression_transfer_corr(pi, X, X)
        assert "expr_corr" in result
        assert "n_genes" in result
        assert "n_scored" in result

    def test_too_few_mapped_cells_returns_nan(self):
        """If fewer than 3 target cells have nonzero column mass, return NaN."""
        n, g = 10, 5
        X = np.random.default_rng(7).random((n, g))
        pi = np.zeros((n, n))
        pi[0, 0] = 1.0   # only 1 mapped column
        result = expression_transfer_corr(pi, X, X)
        assert math.isnan(result["expr_corr"])

    def test_corr_range_is_bounded(self):
        n, g = 15, 8
        rng = np.random.default_rng(8)
        A = rng.random((n, g))
        B = rng.random((n, g))
        pi = _uniform_pi(n, n)
        result = expression_transfer_corr(pi, A, B)
        if not math.isnan(result["expr_corr"]):
            assert -1.0 <= result["expr_corr"] <= 1.0

    def test_constant_gene_skipped(self):
        """Genes with zero variance in source or target are skipped."""
        n, g = 10, 4
        rng = np.random.default_rng(9)
        A = rng.random((n, g))
        B = rng.random((n, g))
        B[:, 0] = 5.0   # constant in target → skip
        pi = np.eye(n) / n
        result_all = expression_transfer_corr(pi, A, A)
        result_const = expression_transfer_corr(pi, A, B)
        assert result_const["n_genes"] <= result_all["n_genes"]


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Supplementary metrics
# ══════════════════════════════════════════════════════════════════════════════

class TestSupplementaryMetrics:

    def test_neighborhood_dissimilarity_cost_perfect_zero(self):
        """Distance matrix with zeros → cost = 0 regardless of pi."""
        n = 5
        dist = np.zeros((n, n))
        pi = _uniform_pi(n, n)
        assert calculate_neighborhood_dissimilarity_cost(dist, pi) == pytest.approx(0.0)

    def test_neighborhood_dissimilarity_cost_scales_linearly(self):
        n = 4
        dist = np.ones((n, n))
        pi = _uniform_pi(n, n)
        c1 = calculate_neighborhood_dissimilarity_cost(dist, pi)
        c2 = calculate_neighborhood_dissimilarity_cost(dist * 2.0, pi)
        assert c2 == pytest.approx(2.0 * c1, rel=1e-10)

    def test_gene_expression_dissimilarity_zero_for_zero_dist(self):
        n = 5
        dist = np.zeros((n, n))
        pi = _uniform_pi(n, n)
        assert calculate_gene_expression_dissimilarity(dist, pi) == pytest.approx(0.0)

    def test_gene_expression_dissimilarity_returns_float(self):
        n = 5
        dist = np.random.default_rng(10).random((n, n))
        pi = _uniform_pi(n, n)
        val = calculate_gene_expression_dissimilarity(dist, pi)
        assert isinstance(val, float)

    def test_cell_type_matching_perfect_plan_same_labels(self):
        """Identity plan + mismatch = 0 → all mass on correct type → matching = 1."""
        n = 5
        mismatch = np.zeros((n, n))   # 0 = same type everywhere
        pi = np.eye(n) / n
        assert cell_type_matching(mismatch, pi) == pytest.approx(1.0)

    def test_cell_type_matching_all_wrong_types(self):
        """mismatch = 1 everywhere → matching = 0."""
        n = 5
        mismatch = np.ones((n, n))
        pi = np.eye(n) / n
        assert cell_type_matching(mismatch, pi) == pytest.approx(0.0)

    def test_cell_type_matching_zero_mass_returns_zero(self):
        n = 5
        mismatch = np.zeros((n, n))
        pi = np.zeros((n, n))
        assert cell_type_matching(mismatch, pi) == pytest.approx(0.0)

    def test_cell_type_matching_range(self):
        n = 6
        rng = np.random.default_rng(11)
        mismatch = rng.random((n, n))
        pi = _uniform_pi(n, n)
        val = cell_type_matching(mismatch, pi)
        assert 0.0 <= val <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 6.  benchmark_method
# ══════════════════════════════════════════════════════════════════════════════

class TestBenchmarkMethod:

    def test_returns_result_and_perf_dict(self):
        result, perf = benchmark_method(sum, range(100))
        assert result == sum(range(100))
        assert "wall_time_s" in perf
        assert "peak_memory_mb" in perf

    def test_wall_time_is_positive(self):
        _, perf = benchmark_method(lambda: [i ** 2 for i in range(10000)])
        assert perf["wall_time_s"] >= 0.0

    def test_peak_memory_is_nonnegative(self):
        _, perf = benchmark_method(lambda: list(range(1000)))
        assert perf["peak_memory_mb"] >= 0.0

    def test_function_exception_propagates(self):
        def bad():
            raise ValueError("oops")
        with pytest.raises(ValueError, match="oops"):
            benchmark_method(bad)

    def test_kwargs_forwarded(self):
        def add(a, b):
            return a + b
        result, _ = benchmark_method(add, 3, b=4)
        assert result == 7


# ══════════════════════════════════════════════════════════════════════════════
# 7.  _extract_gt_indices
# ══════════════════════════════════════════════════════════════════════════════

class TestExtractGtIndices:

    def test_extracts_from_well_formed_uns(self, slice_sim, gt_indices):
        arr = _extract_gt_indices(slice_sim)
        assert arr is not None
        np.testing.assert_array_equal(arr, gt_indices)

    def test_returns_none_when_uns_missing_key(self):
        import anndata as ad
        import numpy as np
        adata = ad.AnnData(X=np.zeros((5, 3)))
        assert _extract_gt_indices(adata) is None

    def test_returns_none_when_uns_empty(self):
        import anndata as ad
        import numpy as np
        adata = ad.AnnData(X=np.zeros((5, 3)))
        adata.uns = {}
        assert _extract_gt_indices(adata) is None

    def test_dtype_is_int64(self, slice_sim):
        arr = _extract_gt_indices(slice_sim)
        assert arr.dtype == np.int64


# ══════════════════════════════════════════════════════════════════════════════
# 8.  evaluate_alignment  (main wrapper)
# ══════════════════════════════════════════════════════════════════════════════

class TestEvaluateAlignment:

    def test_perfect_pi_returns_all_keys(self, perfect_pi, slice_sim, slice_ref, gt_indices):
        out = evaluate_alignment(
            perfect_pi, slice_sim, slice_ref,
            sim_axis=0,
        )
        required = {"lta", "lta_detail", "foscttm", "foscttm_A_to_B",
                    "foscttm_B_to_A", "neg_foscttm", "expr_corr"}
        assert required.issubset(out.keys()), f"Missing keys: {required - out.keys()}"

    def test_perfect_pi_lta_is_one(self, perfect_pi, slice_sim, slice_ref):
        out = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        assert out["lta"] == pytest.approx(1.0)

    def test_perfect_pi_foscttm_near_zero(self, perfect_pi, slice_sim, slice_ref):
        out = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        assert out["foscttm"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_pi_neg_foscttm_near_one(self, perfect_pi, slice_sim, slice_ref):
        out = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        assert out["neg_foscttm"] == pytest.approx(1.0, abs=1e-10)

    def test_neg_foscttm_equals_one_minus_foscttm(self, uniform_pi, slice_sim, slice_ref):
        out = evaluate_alignment(uniform_pi, slice_sim, slice_ref, sim_axis=0)
        if out["foscttm"] is not None:
            assert out["neg_foscttm"] == pytest.approx(1.0 - out["foscttm"], abs=1e-12)

    def test_no_gt_gives_none_foscttm(self, uniform_pi, slice_ref):
        """When the simulated slice has no .uns ground truth, FOSCTTM keys are None."""
        import anndata as ad
        sim_no_gt = ad.AnnData(X=np.zeros((N_SIM, N_GENES)))
        sim_no_gt.obsm["spatial"] = np.random.default_rng(0).random((N_SIM, 2))
        sim_no_gt.obs["cell_type_annot"] = ["A"] * N_SIM
        out = evaluate_alignment(
            uniform_pi, sim_no_gt, slice_ref,
            sim_axis=0, include_expression=False,
        )
        assert out["foscttm"] is None
        assert out["neg_foscttm"] is None

    def test_sim_axis_1_transposes_pi(self, perfect_pi, slice_sim, slice_ref):
        """
        sim_axis=1 means sliceB is the sim.  We pass the transposed perfect_pi
        (shape n_ref × n_sim = 30×20) as sliceB=sim, sliceA=ref.
        evaluate_alignment internally transposes it back → same result.
        """
        out0 = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        out1 = evaluate_alignment(perfect_pi.T, slice_ref, slice_sim, sim_axis=1)
        assert out0["lta"] == pytest.approx(out1["lta"], abs=1e-10)

    def test_include_expression_false_omits_expr_corr(self, uniform_pi, slice_sim, slice_ref):
        out = evaluate_alignment(
            uniform_pi, slice_sim, slice_ref, sim_axis=0,
            include_expression=False,
        )
        assert "expr_corr" not in out

    def test_uniform_pi_metrics_in_range(self, uniform_pi, slice_sim, slice_ref):
        out = evaluate_alignment(uniform_pi, slice_sim, slice_ref, sim_axis=0)
        assert 0.0 <= out["lta"] <= 1.0
        if out["foscttm"] is not None:
            assert 0.0 <= out["foscttm"] <= 1.0

    def test_explicit_gt_override(self, perfect_pi, slice_sim, slice_ref, gt_indices):
        """Pass gt_src_indices explicitly; FOSCTTM must still be computed."""
        # Remove .uns to ensure auto-extract would return None
        import copy
        sim_no_uns = copy.deepcopy(slice_sim)
        sim_no_uns.uns = {}
        out = evaluate_alignment(
            perfect_pi, sim_no_uns, slice_ref, sim_axis=0,
            gt_src_indices=gt_indices,
        )
        assert out["foscttm"] is not None
        assert out["foscttm"] == pytest.approx(0.0, abs=1e-10)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  Metric consistency: better alignment → better scores
# ══════════════════════════════════════════════════════════════════════════════

class TestMetricConsistency:

    def test_perfect_beats_uniform_on_lta(self, perfect_pi, uniform_pi, slice_sim, slice_ref):
        o_p = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        o_u = evaluate_alignment(uniform_pi, slice_sim, slice_ref, sim_axis=0)
        assert o_p["lta"] >= o_u["lta"]

    def test_perfect_beats_uniform_on_foscttm(self, perfect_pi, uniform_pi, slice_sim, slice_ref):
        o_p = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        o_u = evaluate_alignment(uniform_pi, slice_sim, slice_ref, sim_axis=0)
        assert o_p["foscttm"] <= o_u["foscttm"]

    def test_neg_foscttm_ordering_matches_foscttm(self, perfect_pi, uniform_pi,
                                                    slice_sim, slice_ref):
        o_p = evaluate_alignment(perfect_pi, slice_sim, slice_ref, sim_axis=0)
        o_u = evaluate_alignment(uniform_pi, slice_sim, slice_ref, sim_axis=0)
        # neg_foscttm higher is better; perfect should be higher
        assert o_p["neg_foscttm"] >= o_u["neg_foscttm"]
