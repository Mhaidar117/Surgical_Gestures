"""End-to-end smoke test for the GW skill-manifold pipeline.

Runs the full orchestrator (steps 1-10) on the synthetic dataset fixture
with a tiny permutation count, and asserts that the JSON results file is
written and the verification checklist items pass.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
# Make sure both `src/` (for the package) and the pipeline orchestrator are
# importable as modules.
sys.path.insert(0, str(REPO / "src"))


def _load_orchestrator():
    """Import pipeline/skill_manifold_gw.py as a module (it isn't on PYTHONPATH)."""
    spec = importlib.util.spec_from_file_location(
        "skill_manifold_gw", REPO / "pipeline" / "skill_manifold_gw.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pipeline_end_to_end(synthetic_data_root: Path, tmp_path, monkeypatch):
    orch = _load_orchestrator()

    # Redirect the reports/plots to a tmp dir so tests don't pollute the repo.
    monkeypatch.setattr(orch, "ensure_output_dirs",
                        lambda: ((tmp_path / "reports").resolve(),
                                 (tmp_path / "reports" / "plots").resolve()))
    (tmp_path / "reports" / "plots").mkdir(parents=True, exist_ok=True)

    # Load the shipped config but override n_perms and subsample_per_tier for speed.
    from skill_manifold.io import CONFIG_DIR, load_config
    cfg = load_config("skill_manifold.yaml")
    cfg["n_perms"] = 20
    cfg["subsample_per_tier"] = 10
    cfg["trial_null_n_subsamples"] = 3
    cfg["modality_split_n_perms"] = 20
    cfg["modality_split_n_bootstraps"] = 2

    results = orch.run(
        cfg,
        task_modules_yaml=CONFIG_DIR / "skill_manifold_task_modules.yaml",
        data_root_path=synthetic_data_root,
        n_perms_override=20,
        subsample_override=10,
        smoke_fraction=None,
    )

    # Checklist items from the prompt:
    assert results["coverage"]["jigsaws_trials"] > 0
    assert results["coverage"]["eeg_eye_trials"] > 0

    # 3x3 RDMs symmetric with zero diagonal.
    import numpy as np
    rdm_j = np.asarray(results["rdm_jigsaws"])
    rdm_e = np.asarray(results["rdm_eeg_eye"])
    assert rdm_j.shape == (3, 3) and rdm_e.shape == (3, 3)
    assert np.allclose(rdm_j, rdm_j.T)
    assert np.allclose(rdm_e, rdm_e.T)
    assert np.allclose(np.diag(rdm_j), 0)
    assert np.allclose(np.diag(rdm_e), 0)

    # Coupling marginals are uniform.
    T = np.asarray(results["headline_gw"]["coupling"])
    assert np.allclose(T.sum(axis=0), 1.0 / 3.0, atol=1e-6)
    assert np.allclose(T.sum(axis=1), 1.0 / 3.0, atol=1e-6)

    # Null distribution has exactly n_perms valid samples.
    assert results["null"]["n_permutations"] == 20

    # OSATS table has a row per axis.
    assert len(results["osats_axis_alignment"]) == 6

    # Each named check reports a boolean pass/fail.
    names = {c["name"] for c in results["checks"]}
    assert {"rdm_j_valid", "rdm_e_valid", "coupling_marginals_uniform"}.issubset(names)
    for c in results["checks"]:
        if c["name"] in {"rdm_j_valid", "rdm_e_valid",
                         "coupling_marginals_uniform"}:
            assert c["pass"], f"{c['name']} should pass on synthetic data"

    # Report files were written.
    reports_dir = tmp_path / "reports"
    assert (reports_dir / "results_comparison_B.json").exists()
    assert (reports_dir / "report_comparison_B.md").exists()
    for plot in ("rdm_jigsaws.png", "rdm_eeg_eye.png", "coupling_headline.png",
                 "null_histogram.png", "osats_axis_alignment.png",
                 "mds_jigsaws.png", "mds_eeg_eye.png",
                 "trial_level_block_null.png", "trial_level_robustness.png",
                 "trial_level_per_cell_bootstrap.png",
                 "trial_level_epsilon_sensitivity.png"):
        assert (reports_dir / "plots" / plot).exists(), f"{plot} not written"

    # Trial-level PRIMARY (all-trials) sub-dict is populated with expected keys.
    primary = results["trial_level_null_primary"]
    assert primary is not None
    B = np.asarray(primary["block_mass"])
    assert B.shape == (3, 3)
    assert abs(float(B.sum()) - 1.0) < 1e-9
    nj = np.asarray(primary["tier_counts_j"]); ne = np.asarray(primary["tier_counts_e"])
    NJ = int(nj.sum()); NE = int(ne.sum())
    # Block marginals should match tier-count proportions (NOT 1/3 when
    # tiers are unbalanced -- which they will be under tertile binning).
    assert np.allclose(B.sum(axis=1), nj / NJ, atol=1e-6)
    assert np.allclose(B.sum(axis=0), ne / NE, atol=1e-6)
    assert tuple(primary["coupling_shape"]) == (NJ, NE)
    assert primary["n_permutations"] == 20
    assert 0.0 <= primary["p_value"] <= 1.0
    expected = primary["expected_trace_under_null"]
    assert abs(expected - (nj @ ne) / (NJ * NE)) < 1e-12

    # New checks appear in the orchestrator's verification list.
    names = {c["name"] for c in results["checks"]}
    for required in ("trial_null_primary_coupling_shape",
                     "block_mass_marginals_match_tier_counts",
                     "trial_null_mean_near_expected_trace"):
        assert required in names
    # Correctness-critical checks pass on synthetic data. The 1000-perm check
    # is skipped on smoke because we override n_perms = 20 for speed.
    critical = {
        "rdm_j_valid", "rdm_e_valid", "coupling_marginals_uniform",
        "trial_null_primary_coupling_shape",
        "block_mass_marginals_match_tier_counts",
        "trial_null_mean_near_expected_trace",
    }
    for c in results["checks"]:
        if c["name"] in critical:
            assert c["pass"], f"{c['name']} should pass on synthetic data"

    # Balanced sensitivity is populated too (but demoted).
    assert results["trial_level_null_balanced"] is not None

    # Subsample-robustness summary: now includes per-cell stats.
    rb = results["trial_level_robustness"]["summary"]
    assert rb["n_runs"] == 3
    assert len(rb["per_cell_median"]) == 3
    assert len(rb["frac_significant_per_cell_bonf"]) == 3
    assert all(0.0 <= f <= 1.0 for f in rb["frac_significant_per_cell_bonf"])
    assert len(results["trial_level_robustness"]["per_run"]) == 3
    assert all(len(r["per_cell_z"]) == 3
               for r in results["trial_level_robustness"]["per_run"])

    # Epsilon sensitivity: per-cell z is now per row.
    eps_rows = results["epsilon_sensitivity"]
    assert len(eps_rows) == 4
    epsilons = [r["epsilon"] for r in eps_rows]
    assert epsilons == [0.005, 0.01, 0.02, 0.05]
    for r in eps_rows:
        assert len(r["per_cell_z"]) == 3
        assert len(r["per_cell_observed"]) == 3

    # Fixed-cutoff sensitivity either ran or was skipped with a reason.
    fc = results["fixed_cutoff_sensitivity"]
    assert fc is not None

    # Step 10 modality split populated, with the expected three modalities.
    ms = results.get("modality_split") or {}
    if ms:
        mods = {k for k in ms.keys() if k != "_meta"}
        assert mods == {"eeg_baseline", "eeg_predictive_coding", "eye"}
        assert ms["eeg_baseline"]["mimic_feature_dim"] == 64
        assert ms["eeg_predictive_coding"]["mimic_feature_dim"] == 64
        assert ms["eye"]["mimic_feature_dim"] == 18
        for m in mods:
            for binning in ("fixed", "tertile"):
                bs = ms[m][binning]["bootstrap"]
                pcz = np.asarray(bs["per_cell_z"])
                assert pcz.ndim == 2 and pcz.shape[1] == 3
                assert np.isfinite(pcz).all()

    # Residualization diagnostics indicate nuisance was removed.
    diag = results["residualization_diagnostics"]
    # The synthetic data is noisy, but post-fit R^2 should be tiny.
    assert diag["max_post_fit_r2_jigsaws"] < 0.05
    assert diag["max_post_fit_r2_eeg_eye"] < 0.05
