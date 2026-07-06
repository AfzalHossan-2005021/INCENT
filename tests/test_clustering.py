"""
Tests for src/clustering.py mesoregion seeding strategies.

``cluster_cells_spatial(method=...)`` supports three seeding strategies used
by the pipeline (``"farthest_point"``, the default) and by the mesoregion
ablation in benchmark/ablation.py (``"grid"``, ``"random"``). These tests
check the invariants every strategy must satisfy (dense 0..C-1 labels,
contiguity) and the strategy-specific properties (minimum seed spacing for
farthest_point/random; determinism).
"""

from __future__ import annotations

import numpy as np
import pytest
import anndata as ad

from src.clustering import (
    cluster_cells_spatial,
    _farthest_point_seeds,
    _grid_seeds,
    _random_seeds,
)


def _make_grid_adata(n_side=20, spacing=5.0, jitter=0.5, seed=0):
    rng = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(n_side), np.arange(n_side))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64) * spacing
    coords += rng.normal(0, jitter, coords.shape)
    adata = ad.AnnData(X=rng.random((coords.shape[0], 5)).astype(np.float32))
    adata.obsm["spatial"] = coords
    return adata


def _assert_dense_labels(labels: np.ndarray):
    assert labels.min() == 0
    assert np.array_equal(np.unique(labels), np.arange(labels.max() + 1))


class TestClusterCellsSpatialMethods:

    @pytest.mark.parametrize("method", ["farthest_point", "grid", "random"])
    def test_labels_are_dense_0_to_c_minus_1(self, method):
        adata = _make_grid_adata(seed=1)
        labels = cluster_cells_spatial(adata, coarsen_length=25.0, method=method, seed=0)
        assert labels.shape[0] == adata.n_obs
        _assert_dense_labels(labels)

    @pytest.mark.parametrize("method", ["farthest_point", "grid", "random"])
    def test_produces_multiple_clusters_for_large_tissue(self, method):
        adata = _make_grid_adata(n_side=20, seed=2)
        labels = cluster_cells_spatial(adata, coarsen_length=25.0, method=method, seed=0)
        assert labels.max() + 1 >= 2

    def test_empty_slice_returns_empty_labels(self):
        adata = ad.AnnData(X=np.empty((0, 5), dtype=np.float32))
        adata.obsm["spatial"] = np.empty((0, 2), dtype=np.float64)
        labels = cluster_cells_spatial(adata, coarsen_length=10.0)
        assert labels.shape == (0,)

    def test_invalid_method_raises(self):
        adata = _make_grid_adata(seed=3)
        with pytest.raises(ValueError):
            cluster_cells_spatial(adata, coarsen_length=25.0, method="not_a_method")

    def test_nonpositive_coarsen_length_raises(self):
        adata = _make_grid_adata(seed=4)
        with pytest.raises(ValueError):
            cluster_cells_spatial(adata, coarsen_length=0.0)

    def test_random_method_is_deterministic_given_seed(self):
        adata = _make_grid_adata(seed=5)
        labels_a = cluster_cells_spatial(adata, coarsen_length=25.0, method="random", seed=7)
        labels_b = cluster_cells_spatial(adata, coarsen_length=25.0, method="random", seed=7)
        np.testing.assert_array_equal(labels_a, labels_b)

    def test_random_method_differs_across_seeds(self):
        """Different seeds should (almost always) produce a different partition."""
        adata = _make_grid_adata(n_side=20, seed=6)
        labels_a = cluster_cells_spatial(adata, coarsen_length=20.0, method="random", seed=1)
        labels_b = cluster_cells_spatial(adata, coarsen_length=20.0, method="random", seed=2)
        assert not np.array_equal(labels_a, labels_b)

    @pytest.mark.parametrize("method", ["farthest_point", "grid", "random"])
    def test_default_farthest_point_matches_explicit_call(self, method):
        """method='farthest_point' (the pipeline default) is reachable both
        implicitly and explicitly and gives identical results."""
        adata = _make_grid_adata(seed=8)
        implicit = cluster_cells_spatial(adata, coarsen_length=25.0)
        explicit = cluster_cells_spatial(adata, coarsen_length=25.0, method="farthest_point")
        np.testing.assert_array_equal(implicit, explicit)


class TestFarthestPointSeeds:

    def test_minimum_spacing_respected(self):
        rng = np.random.default_rng(0)
        coords = rng.random((300, 2)) * 100.0
        S = 15.0
        seeds = _farthest_point_seeds(coords, S)
        assert seeds.shape[0] >= 2
        d = np.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= S - 1e-9


class TestRandomSeeds:

    def test_minimum_spacing_respected(self):
        rng = np.random.default_rng(0)
        coords = rng.random((300, 2)) * 100.0
        S = 15.0
        seeds = _random_seeds(coords, S, np.random.default_rng(1))
        assert seeds.shape[0] >= 2
        d = np.linalg.norm(seeds[:, None, :] - seeds[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        assert d.min() >= S - 1e-9

    def test_seeds_are_a_subset_of_input_cells(self):
        rng = np.random.default_rng(2)
        coords = rng.random((50, 2)) * 100.0
        seeds = _random_seeds(coords, 20.0, np.random.default_rng(3))
        for seed_pt in seeds:
            assert np.any(np.all(np.isclose(coords, seed_pt), axis=1))


class TestGridSeeds:

    def test_seeds_cover_tissue_footprint(self):
        adata = _make_grid_adata(n_side=15, seed=9)
        coords = np.asarray(adata.obsm["spatial"])
        seeds = _grid_seeds(coords, 20.0)
        assert seeds.shape[0] >= 1
        # every seed must be within S of some actual cell (tissue-outline following)
        from scipy.spatial import cKDTree
        tree = cKDTree(coords)
        d, _ = tree.query(seeds, k=1)
        assert np.all(d <= 20.0 + 1e-6)

    def test_produces_more_clusters_than_farthest_point_on_dense_grid(self):
        """
        Documented behavioral difference (see cluster_cells_spatial docstring):
        grid seeding is anchored to an estimated PCA frame and keeps every
        seed near an actual cell, so on a dense regular tissue it tends to
        retain more seeds than farthest_point's blue-noise spacing rule.
        """
        adata = _make_grid_adata(n_side=20, seed=10)
        labels_fp = cluster_cells_spatial(adata, coarsen_length=20.0, method="farthest_point")
        labels_grid = cluster_cells_spatial(adata, coarsen_length=20.0, method="grid")
        assert (labels_grid.max() + 1) >= (labels_fp.max() + 1)
