"""Tests for the Mimic-side modality split (Step 10)."""
from __future__ import annotations

import numpy as np

from skill_manifold.features_eeg_eye import (
    EEG_BASELINE_DIM, EEG_PC_DIM, EYE_DIM,
    feature_column_names, mimic_modality_columns,
)


def test_mimic_modality_columns_partition():
    """The three modality subsets must be disjoint, union to the full column
    set, and have the expected dimensionality (64, 64, 18)."""
    mod = mimic_modality_columns()
    assert set(mod.keys()) == {"eeg_baseline", "eeg_predictive_coding", "eye"}

    assert len(mod["eeg_baseline"]) == EEG_BASELINE_DIM == 64
    assert len(mod["eeg_predictive_coding"]) == EEG_PC_DIM == 64
    assert len(mod["eye"]) == EYE_DIM == 18

    all_cols = feature_column_names()
    # Union covers the full feature name list.
    assert set().union(*mod.values()) == set(all_cols)
    # Disjoint.
    assert (len(mod["eeg_baseline"]) + len(mod["eeg_predictive_coding"])
            + len(mod["eye"])) == len(all_cols)
    # Prefixes are the advertised ones.
    assert all(c.startswith("eeg_base_") for c in mod["eeg_baseline"])
    assert all(c.startswith("eeg_pc_") for c in mod["eeg_predictive_coding"])
    assert all(c.startswith("eye_") for c in mod["eye"])
    # Order within each subset matches feature_column_names order.
    for subset in mod.values():
        indices = [all_cols.index(c) for c in subset]
        assert indices == sorted(indices)
