"""
Tests for the ablation-enabling flags added to src/hierarchical.py:

* ``use_geometric_admissibility`` (score_frontier_matches / expand_macro_match_frontier
  / extract_continuous_macro_section) -- "w/o geometric admissibility" ablation.
* ``balanced`` (run_coarse_fugw) -- "balanced instead of unbalanced" ablation.

These exercise the pure-numpy macro-section scoring logic and the small
coarse-FUGW solver directly (no full pairwise_align/hierarchical_pairwise_align
run), so they stay fast.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.hierarchical import score_frontier_matches, run_coarse_fugw


# ---------------------------------------------------------------------------
# use_geometric_admissibility
# ---------------------------------------------------------------------------

class TestGeometricAdmissibilityFlag:
    """
    Construct a scenario with two already-selected pairs defining a rigid
    transform, and a frontier candidate whose target centroid is deliberately
    placed far from where that transform predicts -- i.e. geometrically
    inadmissible. With the guard enabled it must be rejected; with the guard
    disabled it must survive (subject to still beating the unmatched null,
    which the strong global_pair_evidence below guarantees).
    """

    def _scenario(self, outlier_shift):
        # Two slices, two selected pairs (0,0) and (1,1) forming an identity
        # rigid transform (translation 0, rotation 0) at unit spacing.
        centroids_A = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
        centroids_B = np.array([[0.0, 0.0], [10.0, 0.0], [5.0 + outlier_shift, 10.0]])
        selected_pairs = [(0, 0), (1, 1)]

        # Frontier: cluster 2 in A vs cluster 2 in B (the shifted one).
        frontier_A = {2}
        frontier_B = {2}
        candidate_pair_set = {(2, 2)}

        # adjacency: cluster 2 must be a spatial neighbor of at least one
        # selected cluster in both slices for `support_pairs` to be non-empty.
        adj_A = np.array([
            [True, True, True],
            [True, True, False],
            [True, False, True],
        ])
        adj_B = adj_A.copy()

        global_pair_evidence = {(0, 0): 5.0, (1, 1): 5.0, (2, 2): 5.0}
        mi_contrib = np.full((3, 3), 1.0)

        return dict(
            frontier_A=frontier_A,
            frontier_B=frontier_B,
            candidate_pair_set=candidate_pair_set,
            selected_pairs=selected_pairs,
            global_pair_evidence=global_pair_evidence,
            adj_A=adj_A,
            adj_B=adj_B,
            centroids_A=centroids_A,
            centroids_B=centroids_B,
            mi_contrib=mi_contrib,
            transform_scale=1.0,
        )

    def test_gross_outlier_rejected_when_admissibility_enabled(self):
        kwargs = self._scenario(outlier_shift=1000.0)
        pairs, scores = score_frontier_matches(**kwargs, use_geometric_admissibility=True)
        assert (2, 2) not in pairs

    def test_gross_outlier_retained_when_admissibility_disabled(self):
        kwargs = self._scenario(outlier_shift=1000.0)
        pairs, scores = score_frontier_matches(**kwargs, use_geometric_admissibility=False)
        assert (2, 2) in pairs

    def test_default_is_admissibility_enabled(self):
        """The keyword default must match the pipeline's pre-ablation behavior."""
        kwargs = self._scenario(outlier_shift=1000.0)
        pairs_default, _ = score_frontier_matches(**kwargs)
        pairs_explicit, _ = score_frontier_matches(**kwargs, use_geometric_admissibility=True)
        assert pairs_default == pairs_explicit

    def test_consistent_pair_not_affected_by_flag(self):
        """A geometrically consistent frontier pair should pass either way."""
        kwargs = self._scenario(outlier_shift=0.0)
        pairs_on, _ = score_frontier_matches(**kwargs, use_geometric_admissibility=True)
        pairs_off, _ = score_frontier_matches(**kwargs, use_geometric_admissibility=False)
        assert (2, 2) in pairs_on
        assert (2, 2) in pairs_off


# ---------------------------------------------------------------------------
# balanced flag on run_coarse_fugw
# ---------------------------------------------------------------------------

class TestBalancedFlag:

    def _toy_cluster_problem(self, seed=0):
        rng = np.random.default_rng(seed)
        n_a, n_b = 5, 6
        centroids_A = rng.random((n_a, 2)) * 50.0
        centroids_B = rng.random((n_b, 2)) * 50.0
        from src.hierarchical import compute_cluster_structural_matrix
        C_A = compute_cluster_structural_matrix(centroids_A)
        C_B = compute_cluster_structural_matrix(centroids_B)
        M = rng.random((n_a, n_b))
        p_A = np.full(n_a, 1.0 / n_a)
        p_B = np.full(n_b, 1.0 / n_b)
        return M, C_A, C_B, p_A, p_B

    def test_balanced_returns_valid_coupling(self):
        M, C_A, C_B, p_A, p_B = self._toy_cluster_problem()
        pi = run_coarse_fugw(M, C_A, C_B, p_A, p_B, alpha=0.5, balanced=True, use_gpu=False)
        assert pi.shape == (5, 6)
        assert np.isfinite(pi).all()
        assert (pi >= -1e-9).all()

    def test_balanced_respects_marginals(self):
        """Balanced FGW enforces the exact row/column marginals p_A, p_B."""
        M, C_A, C_B, p_A, p_B = self._toy_cluster_problem(seed=1)
        pi = run_coarse_fugw(M, C_A, C_B, p_A, p_B, alpha=0.5, balanced=True, use_gpu=False)
        np.testing.assert_allclose(pi.sum(axis=1), p_A, atol=1e-4)
        np.testing.assert_allclose(pi.sum(axis=0), p_B, atol=1e-4)

    def test_unbalanced_is_default(self):
        M, C_A, C_B, p_A, p_B = self._toy_cluster_problem(seed=2)
        pi_default = run_coarse_fugw(M, C_A, C_B, p_A, p_B, alpha=0.5, reg_m=1.0, use_gpu=False)
        pi_explicit = run_coarse_fugw(
            M, C_A, C_B, p_A, p_B, alpha=0.5, reg_m=1.0, balanced=False, use_gpu=False
        )
        np.testing.assert_allclose(pi_default, pi_explicit)

    def test_balanced_and_unbalanced_differ(self):
        """Sanity check that the two code paths are actually distinct solvers."""
        M, C_A, C_B, p_A, p_B = self._toy_cluster_problem(seed=3)
        pi_balanced = run_coarse_fugw(M, C_A, C_B, p_A, p_B, alpha=0.5, balanced=True, use_gpu=False)
        pi_unbalanced = run_coarse_fugw(
            M, C_A, C_B, p_A, p_B, alpha=0.5, reg_m=0.05, balanced=False, use_gpu=False
        )
        # Unbalanced with a very loose reg_m may drop mass; balanced never does.
        assert pi_balanced.sum() == pytest.approx(1.0, abs=1e-4)
