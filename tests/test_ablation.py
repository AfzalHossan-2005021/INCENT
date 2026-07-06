"""
Tests for benchmark/ablation.py.

Uses a stub aligner (returns a uniform coupling, no real OT) so the whole
sweep runs in well under a second, mirroring the convention in
tests/test_benchmark.py. Real alignment behavior for the new core.py/
hierarchical.py/clustering.py flags is covered separately in
tests/test_hierarchical_ablation.py, tests/test_clustering.py, and
tests/test_core_ablation_integration.py.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import anndata as ad

from benchmark.ablation import (
    ALL_VARIANTS,
    MAIN_VARIANTS,
    SUPPLEMENTARY_VARIANTS,
    VARIANTS_BY_NAME,
    AblationVariant,
    _without_cell_type,
    _without_neighborhood,
    _without_gene_expression,
    _paired_stats,
    _bh_fdr,
    run_ablation,
)
from src.tuning import DEFAULT_INIT
from tests.conftest import _make_adata


def _stub_aligner(sliceA, sliceB, **kwargs):
    nA, nB = sliceA.n_obs, sliceB.n_obs
    return np.full((nA, nB), 1.0 / (nA * nB), dtype=np.float64)


def _make_reference_and_section(n_cells=60, seed=0):
    reference = _make_adata(n_cells, 8, seed=seed)
    reference.obs_names = [f"c{i}" for i in range(n_cells)]
    coords = np.asarray(reference.obsm["spatial"])
    section = reference[coords[:, 0] < np.median(coords[:, 0]) + 20].copy()
    return section, reference


# ---------------------------------------------------------------------------
# Variant table
# ---------------------------------------------------------------------------

class TestVariantTable:

    def test_all_variants_is_main_plus_supplementary(self):
        assert ALL_VARIANTS == MAIN_VARIANTS + SUPPLEMENTARY_VARIANTS

    def test_full_incent_is_present(self):
        assert any(v.name == "Full INCENT" for v in ALL_VARIANTS)

    def test_variant_names_are_unique(self):
        names = [v.name for v in ALL_VARIANTS]
        assert len(names) == len(set(names))

    def test_variants_by_name_matches_all_variants(self):
        assert set(VARIANTS_BY_NAME) == {v.name for v in ALL_VARIANTS}

    def test_hierarchy_variant_uses_flat_aligner_and_three_weight_keys(self):
        v = VARIANTS_BY_NAME["w/o hierarchy"]
        assert v.aligner_kind == "flat"
        assert set(v.weight_keys) == {"alpha", "beta", "gamma"}

    def test_hierarchical_variants_use_five_weight_keys(self):
        for v in ALL_VARIANTS:
            if v.name == "w/o hierarchy":
                continue
            assert set(v.weight_keys) == {"alpha", "beta", "gamma", "alpha_cluster", "delta"}


# ---------------------------------------------------------------------------
# Weight-transform arithmetic
# ---------------------------------------------------------------------------

class TestWeightTransforms:

    def test_without_cell_type_zeroes_beta_and_delta(self):
        w = _without_cell_type(DEFAULT_INIT)
        assert w["beta"] == 0.0
        assert w["delta"] == 0.0

    def test_without_cell_type_renormalizes_gamma(self):
        base = {"alpha": 0.5, "beta": 0.4, "gamma": 0.3, "alpha_cluster": 0.5, "delta": 0.5}
        w = _without_cell_type(base)
        assert w["gamma"] == pytest.approx(0.3 / 0.6)

    def test_without_cell_type_leaves_alpha_and_alpha_cluster_untouched(self):
        w = _without_cell_type(DEFAULT_INIT)
        assert w["alpha"] == DEFAULT_INIT["alpha"]
        assert w["alpha_cluster"] == DEFAULT_INIT["alpha_cluster"]

    def test_without_cell_type_raises_when_beta_is_one(self):
        base = {"alpha": 0.5, "beta": 1.0, "gamma": 0.0, "alpha_cluster": 0.5, "delta": 0.5}
        with pytest.raises(ValueError):
            _without_cell_type(base)

    def test_without_neighborhood_zeroes_gamma_only(self):
        w = _without_neighborhood(DEFAULT_INIT)
        assert w["gamma"] == 0.0
        assert w["delta"] == DEFAULT_INIT["delta"]  # coarse term untouched

    def test_without_neighborhood_renormalizes_beta(self):
        base = {"alpha": 0.5, "beta": 0.4, "gamma": 0.3, "alpha_cluster": 0.5, "delta": 0.5}
        w = _without_neighborhood(base)
        assert w["beta"] == pytest.approx(0.4 / 0.7)

    def test_without_gene_expression_sets_delta_to_one(self):
        w = _without_gene_expression(DEFAULT_INIT)
        assert w["delta"] == 1.0

    def test_without_gene_expression_beta_gamma_sum_to_one(self):
        w = _without_gene_expression(DEFAULT_INIT)
        assert (w["beta"] + w["gamma"]) == pytest.approx(1.0)

    def test_without_gene_expression_preserves_beta_gamma_ratio(self):
        base = {"alpha": 0.5, "beta": 0.4, "gamma": 0.2, "alpha_cluster": 0.5, "delta": 0.5}
        w = _without_gene_expression(base)
        assert (w["beta"] / w["gamma"]) == pytest.approx(0.4 / 0.2)

    def test_without_gene_expression_raises_when_beta_gamma_zero(self):
        base = {"alpha": 0.5, "beta": 0.0, "gamma": 0.0, "alpha_cluster": 0.5, "delta": 0.5}
        with pytest.raises(ValueError):
            _without_gene_expression(base)


# ---------------------------------------------------------------------------
# Paired statistics
# ---------------------------------------------------------------------------

class TestPairedStats:

    def test_identical_arrays_give_zero_delta(self):
        vals = np.array([0.1, 0.2, 0.3, 0.4])
        s = _paired_stats(vals, vals.copy(), "foscttm")
        assert s["mean_delta"] == pytest.approx(0.0)
        assert s["pct_degradation"] == pytest.approx(0.0)

    def test_foscttm_worse_ablation_gives_positive_degradation(self):
        """foscttm is lower-is-better; an ablation with HIGHER foscttm is worse (positive %)."""
        full = np.array([0.1, 0.1, 0.1, 0.1])
        ablation = np.array([0.2, 0.2, 0.2, 0.2])
        s = _paired_stats(full, ablation, "foscttm")
        assert s["pct_degradation"] > 0

    def test_lta_worse_ablation_gives_positive_degradation(self):
        """lta is higher-is-better; an ablation with LOWER lta is worse (positive %)."""
        full = np.array([0.8, 0.8, 0.8, 0.8])
        ablation = np.array([0.6, 0.6, 0.6, 0.6])
        s = _paired_stats(full, ablation, "lta")
        assert s["pct_degradation"] > 0

    def test_lta_better_ablation_gives_negative_degradation(self):
        full = np.array([0.6, 0.6, 0.6, 0.6])
        ablation = np.array([0.8, 0.8, 0.8, 0.8])
        s = _paired_stats(full, ablation, "lta")
        assert s["pct_degradation"] < 0

    def test_empty_arrays_return_none_stats(self):
        s = _paired_stats(np.array([]), np.array([]), "foscttm")
        assert s["n"] == 0
        assert s["mean_delta"] is None
        assert s["pct_degradation"] is None

    def test_n_is_number_of_pairs(self):
        full = np.array([0.1, 0.2, 0.3])
        ablation = np.array([0.15, 0.25, 0.35])
        s = _paired_stats(full, ablation, "foscttm")
        assert s["n"] == 3

    def test_wilcoxon_p_is_none_for_single_pair(self):
        s = _paired_stats(np.array([0.1]), np.array([0.2]), "foscttm")
        assert s["wilcoxon_p"] is None

    def test_sem_zero_for_single_pair(self):
        s = _paired_stats(np.array([0.1]), np.array([0.2]), "foscttm")
        assert s["sem_delta"] == 0.0


class TestBhFdr:

    def test_monotone_input_gives_monotone_q(self):
        p = [0.001, 0.01, 0.02, 0.5]
        q = _bh_fdr(p)
        assert np.all(np.diff(q) >= -1e-9)

    def test_all_significant_when_all_p_tiny(self):
        q = _bh_fdr([1e-6, 1e-6, 1e-6])
        assert np.all(q < 0.05)

    def test_none_and_nan_pass_through(self):
        q = _bh_fdr([0.01, None, float("nan")])
        assert np.isfinite(q[0])
        assert np.isnan(q[1])
        assert np.isnan(q[2])

    def test_single_pvalue_equals_itself(self):
        q = _bh_fdr([0.03])
        assert q[0] == pytest.approx(0.03)


# ---------------------------------------------------------------------------
# run_ablation orchestration (stub aligner: fast, no real OT)
# ---------------------------------------------------------------------------

class TestRunAblation:

    def test_runs_all_variants_by_default(self, tmp_path):
        section, reference = _make_reference_and_section(seed=10)
        res = run_ablation(
            section=section, reference=reference, n_instances=2,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        assert {r["variant"] for r in res["rows"]} == {v.name for v in ALL_VARIANTS}

    def test_respects_explicit_variant_subset(self, tmp_path):
        section, reference = _make_reference_and_section(seed=11)
        subset = [VARIANTS_BY_NAME["Full INCENT"], VARIANTS_BY_NAME["w/o cell type"]]
        res = run_ablation(
            section=section, reference=reference, n_instances=2, variants=subset,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        assert {r["variant"] for r in res["rows"]} == {"Full INCENT", "w/o cell type"}

    def test_requires_full_incent_in_variants(self, tmp_path):
        section, reference = _make_reference_and_section(seed=12)
        with pytest.raises(ValueError):
            run_ablation(
                section=section, reference=reference, n_instances=2,
                variants=[VARIANTS_BY_NAME["w/o cell type"]],
                align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
                outdir=str(tmp_path),
            )

    def test_rejects_forbidden_align_kwargs(self, tmp_path):
        section, reference = _make_reference_and_section(seed=13)
        with pytest.raises(ValueError):
            run_ablation(
                section=section, reference=reference, n_instances=2,
                align_kwargs={"use_gpu": False, "balanced": True},
                aligner_override=_stub_aligner, outdir=str(tmp_path),
            )

    def test_writes_json_and_figure(self, tmp_path):
        section, reference = _make_reference_and_section(seed=14)
        run_ablation(
            section=section, reference=reference, n_instances=2,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        assert (tmp_path / "ablation.json").exists()
        assert (tmp_path / "ablation.png").exists()

    def test_json_is_valid_and_matches_returned_dict(self, tmp_path):
        section, reference = _make_reference_and_section(seed=15)
        res = run_ablation(
            section=section, reference=reference, n_instances=2,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        with open(tmp_path / "ablation.json") as f:
            loaded = json.load(f)
        assert set(loaded["config"]["variant_names"]) == {r["variant"] for r in res["rows"]}

    def test_paired_stats_has_zero_delta_for_identical_stub_output(self, tmp_path):
        """The stub aligner ignores its kwargs, so every variant produces the
        exact same coupling as Full INCENT -- paired deltas must be exactly 0."""
        section, reference = _make_reference_and_section(seed=16)
        res = run_ablation(
            section=section, reference=reference, n_instances=3,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        for metric, per_variant in res["paired_stats"].items():
            for name, s in per_variant.items():
                if s["n"] > 0:
                    assert s["mean_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_no_outdir_skips_file_writes(self, tmp_path):
        section, reference = _make_reference_and_section(seed=17)
        run_ablation(
            section=section, reference=reference, n_instances=2,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=None, seed=0,
        )
        assert list(tmp_path.iterdir()) == []

    def test_instances_reused_identically_across_variants(self, tmp_path):
        """Paired design: every variant must see the same n instances."""
        section, reference = _make_reference_and_section(seed=18)
        res = run_ablation(
            section=section, reference=reference, n_instances=3,
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        n_ok = {r["n_ok"] for r in res["rows"]}
        assert n_ok == {3}

    def test_real_pairs_mode_has_no_foscttm(self, tmp_path):
        """real_pairs aligns as-is (no simulate_adjacent_slice), so no FOSCTTM ground truth."""
        section, reference = _make_reference_and_section(seed=19)
        res = run_ablation(
            real_pairs=[(section, reference)],
            align_kwargs={"use_gpu": False}, aligner_override=_stub_aligner,
            outdir=str(tmp_path), seed=0,
        )
        full_row = next(r for r in res["rows"] if r["variant"] == "Full INCENT")
        for m in full_row["per_instance"]:
            assert m is None or m.get("foscttm") is None
