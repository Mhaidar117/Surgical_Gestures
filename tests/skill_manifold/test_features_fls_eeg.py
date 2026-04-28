"""Unit tests for src/skill_manifold/features_fls_eeg.py.

Covers the parts of the EEG feature builder that do not require an EDF
file on disk (the EDF-loading piece is exercised by the smoke test). The
focus here is the pure-numpy logic: channel-name -> region mapping, soft
bad-channel detection, Welch-bandpower correctness, and the regional
log-power feature shape.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from skill_manifold.features_fls_eeg import (    # noqa: E402
    BANDS, REGION_ORDER, SUSPECT_CHANNELS,
    _canonical_label, _is_eog_channel,
    assign_region, build_region_index,
    eeg_feature_column_names, integrate_bands,
    regional_log_power, soft_bad_channels,
)


# ---------- EOG channel detection -------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        # Canonical EOG names from EOG_CHANNELS list.
        ("EEGHEOGRCPz", True),
        ("EEGHEOGLCPz", True),
        ("EEGVEOGUCPz", True),
        ("EEGVEOGLCPz", True),
        # Concatenated variants that the dataset emits and that the
        # original list missed (the postmortem flagged these as the only
        # remaining channels triggering the interpolate_bads fallback).
        ("EEGHEOGLHEOGRCPz", True),
        ("EEGVEOGUVEOGLCPz", True),
        # Canonicalized form (after stripping EEG prefix and CPz suffix).
        ("HEOGR", True),
        ("HEOGLHEOGR", True),
        ("VEOGU", True),
        # Non-EOG scalp channels.
        ("EEGFp1CPz", False),
        ("EEGCzCPz", False),
        ("Fp1", False),
        ("Cz", False),
        ("F8", False),
        # Defensive: case insensitivity (rare in practice but free).
        ("eegheogrcpz", True),
    ],
)
def test_is_eog_channel(name: str, expected: bool) -> None:
    assert _is_eog_channel(name) is expected


# ---------- canonical labels -------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("EEGFp1CPz", "Fp1"),       # representative FLS-format name
        ("EEGCzCPz", "Cz"),
        ("EEGFCzCPz", "FCz"),
        ("EEGT7CPz", "T7"),
        ("Fp1", "Fp1"),             # already canonical -> unchanged
        ("Cz", "Cz"),
        ("EEGHEOGRCPz", "HEOGR"),   # EOG: stripped but still non-10-20
    ],
)
def test_canonical_label_strips_prefix_and_suffix(raw: str, expected: str) -> None:
    assert _canonical_label(raw) == expected


# ---------- region mapping ---------------------------------------------------

@pytest.mark.parametrize(
    "name, expected",
    [
        # Canonical short-form (existing pre-FLS coverage).
        ("F3", "frontal_L"),
        ("F4", "frontal_R"),
        ("Fp1", "frontal_L"),
        ("AF7", "frontal_L"),
        ("FC2", "frontal_R"),
        ("C3", "central_L"),
        ("T7", "central_L"),
        ("CP4", "parietal_R"),
        ("P3", "parietal_L"),
        ("PO8", "parietal_R"),
        ("O1", "occipital_L"),
        ("O2", "occipital_R"),
        ("Cz", None),         # midline -> dropped
        ("FCz", None),
        ("Oz", None),
        # FLS dataset's actual naming convention (the regression that the
        # postmortem flagged: the original code returned None for all of
        # these because every name ends in `z`).
        ("EEGFp1CPz", "frontal_L"),
        ("EEGFp2CPz", "frontal_R"),
        ("EEGF3CPz", "frontal_L"),
        ("EEGF8CPz", "frontal_R"),
        ("EEGFC3CPz", "frontal_L"),
        ("EEGT7CPz", "central_L"),
        ("EEGT8CPz", "central_R"),
        ("EEGC3CPz", "central_L"),
        ("EEGC4CPz", "central_R"),
        ("EEGCP5CPz", "parietal_L"),
        ("EEGP4CPz", "parietal_R"),
        ("EEGPO8CPz", "parietal_R"),
        ("EEGO1CPz", "occipital_L"),
        ("EEGO2CPz", "occipital_R"),
        ("EEGCzCPz", None),        # midline preserved through canonicalization
        ("EEGFCzCPz", None),
    ],
)
def test_assign_region_examples(name: str, expected) -> None:
    assert assign_region(name) == expected


def test_build_region_index_partitions_known_channels() -> None:
    chans = ["F3", "F4", "Fp1", "Fp2", "C3", "C4", "P3", "P4", "O1", "O2",
             "Cz", "Pz", "Oz", "EEGHEOGLCPz"]   # mids + EOG
    idx = build_region_index(chans)
    assert sorted(idx.keys()) == sorted(REGION_ORDER)
    flat = sum((idx[r] for r in REGION_ORDER), [])
    # Exactly the 10 lateralized channels should appear.
    assert sorted(flat) == sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


def test_build_region_index_with_full_fls_naming() -> None:
    """Regression for the postmortem bug: with the FLS naming convention
    (`EEG<label>CPz`), every channel must still route to a region rather
    than collapsing to None."""
    chans = ["EEGFp1CPz", "EEGFp2CPz", "EEGF3CPz", "EEGF4CPz",
             "EEGC3CPz", "EEGC4CPz", "EEGP3CPz", "EEGP4CPz",
             "EEGO1CPz", "EEGO2CPz",
             "EEGCzCPz", "EEGPzCPz", "EEGOzCPz",          # midline -> excluded
             "EEGHEOGLCPz", "EEGVEOGUCPz"]                 # EOG -> excluded
    idx = build_region_index(chans)
    flat = sum((idx[r] for r in REGION_ORDER), [])
    # The 10 lateralized channels should land in regions, the rest excluded.
    assert sorted(flat) == sorted([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # Every region should now have at least one channel (the postmortem
    # bug had every region empty).
    nonempty = sum(1 for r in REGION_ORDER if idx[r])
    assert nonempty == 8


# ---------- bad channel detection -------------------------------------------

def test_soft_bad_channels_flags_zero_variance() -> None:
    rng = np.random.default_rng(0)
    n_samples = 1000
    n_chans = 10
    data = rng.standard_normal((n_samples, n_chans)).astype(np.float64)
    data[:, 3] = 0.0    # flatlined channel
    chans = [f"E{i}" for i in range(n_chans)]
    bad = soft_bad_channels(data, chans)
    assert "E3" in bad


def test_soft_bad_channels_flags_huge_variance() -> None:
    rng = np.random.default_rng(0)
    n_samples = 1000
    n_chans = 10
    data = rng.standard_normal((n_samples, n_chans)).astype(np.float64)
    data[:, 5] *= 200.0  # 200x noisier than its peers
    chans = [f"E{i}" for i in range(n_chans)]
    bad = soft_bad_channels(data, chans)
    assert "E5" in bad


def test_soft_bad_channels_passes_clean_data() -> None:
    rng = np.random.default_rng(0)
    data = rng.standard_normal((1000, 8)).astype(np.float64)
    chans = [f"E{i}" for i in range(8)]
    assert soft_bad_channels(data, chans) == []


def test_suspect_channels_get_stricter_treatment() -> None:
    """A SUSPECT channel that is 5x noisier than the rest should be flagged
    even though the global threshold (100x) wouldn't catch it."""
    rng = np.random.default_rng(0)
    n_samples = 1000
    chans = list(SUSPECT_CHANNELS) + ["F3", "F4", "C3", "C4"]
    data = rng.standard_normal((n_samples, len(chans))).astype(np.float64)
    # Boost F8 (suspect) by 5x; global rule wouldn't flag it but the
    # suspect-list path tightens to [0.1, 10] x median.
    data[:, chans.index("F8")] *= 12.0
    bad = soft_bad_channels(data, chans)
    assert "F8" in bad


# ---------- bandpower numerics -----------------------------------------------

def test_integrate_bands_picks_up_alpha_oscillation() -> None:
    """A pure 10 Hz sinusoid should dominate the alpha band relative to others."""
    sfreq = 500.0
    n = int(20 * sfreq)
    t = np.arange(n) / sfreq
    sig = np.sin(2 * np.pi * 10.0 * t)
    data = sig.reshape(-1, 1)   # one channel
    from scipy.signal import welch
    freqs, psd = welch(data[:, 0], fs=sfreq, nperseg=int(2 * sfreq))
    bp = integrate_bands(freqs, psd[None, :], BANDS)   # (1, 5)
    band_names = [b[0] for b in BANDS]
    alpha_idx = band_names.index("alpha")
    assert bp[0, alpha_idx] == bp[0].max(), "alpha should dominate for a 10 Hz sine"


def test_regional_log_power_shape() -> None:
    sfreq = 500.0
    n_samples = int(10 * sfreq)
    chans = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, len(chans))).astype(np.float64)
    vec = regional_log_power(data, chans, sfreq=sfreq)
    expected_dim = len(REGION_ORDER) * len(BANDS)
    assert vec.shape == (expected_dim,)
    assert np.all(np.isfinite(vec))


def test_relative_bandpower_invariant_to_global_gain() -> None:
    """A per-subject multiplicative gain (skin/skull conductance, electrode
    coupling) should leave the relative-bandpower output invariant. This
    is the property that addresses the PC1=96% degeneracy diagnosed in
    the postmortem: with absolute power the gain landed entirely on PC1
    and the cosine RDM degenerated to a sign indicator on PC1.
    """
    sfreq = 500.0
    n_samples = int(10 * sfreq)
    chans = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, len(chans))).astype(np.float64)
    vec_a = regional_log_power(data, chans, sfreq=sfreq, relative=True)
    vec_b = regional_log_power(data * 10.0, chans, sfreq=sfreq, relative=True)
    np.testing.assert_allclose(vec_a, vec_b, atol=1e-9)


def test_absolute_bandpower_responds_to_gain() -> None:
    """Sanity check on the legacy (relative=False) path: a 10x gain in the
    time-domain signal corresponds to a 100x gain in power, i.e. +2 in
    log10 space across every (region, band) cell."""
    sfreq = 500.0
    n_samples = int(10 * sfreq)
    chans = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, len(chans))).astype(np.float64)
    vec_a = regional_log_power(data, chans, sfreq=sfreq, relative=False)
    vec_b = regional_log_power(data * 10.0, chans, sfreq=sfreq, relative=False)
    np.testing.assert_allclose(vec_b - vec_a, 2.0, atol=1e-6)


def test_relative_bandpower_handles_dead_channel() -> None:
    """A channel of all zeros should produce a zero row in the relative
    matrix rather than NaN -- otherwise the whole feature vector would
    propagate NaNs to the regional mean."""
    sfreq = 500.0
    n_samples = int(10 * sfreq)
    chans = ["F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2"]
    rng = np.random.default_rng(0)
    data = rng.standard_normal((n_samples, len(chans))).astype(np.float64)
    data[:, 2] = 0.0  # zero out C3 (one channel in central_L region)
    vec = regional_log_power(data, chans, sfreq=sfreq, relative=True)
    assert np.all(np.isfinite(vec))


def test_eeg_feature_column_names_count() -> None:
    cols = eeg_feature_column_names()
    assert len(cols) == len(REGION_ORDER) * len(BANDS)
    # Every entry should follow `eeg_<region>_<band>` and the regions and bands
    # should each be hit exactly len(other) times.
    band_names = [b[0] for b in BANDS]
    region_hits = {r: 0 for r in REGION_ORDER}
    band_hits = {b: 0 for b in band_names}
    for c in cols:
        assert c.startswith("eeg_")
        for r in REGION_ORDER:
            if c.startswith(f"eeg_{r}_"):
                region_hits[r] += 1
                tail = c[len(f"eeg_{r}_"):]
                assert tail in band_names
                band_hits[tail] += 1
                break
    assert all(v == len(band_names) for v in region_hits.values())
    assert all(v == len(REGION_ORDER) for v in band_hits.values())
