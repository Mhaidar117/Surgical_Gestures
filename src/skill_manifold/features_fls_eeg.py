"""Per-trial EEG feature builder for the NIBIB-RPCCC-FLS dataset.

Each FLS trial is one EDF under
``data/laparoscopic-surgery-fls-tasks/EEG_FLS/{subject}_{task}_{try}.edf``,
recorded at 500 Hz with 128 channels (AntNeuro WaveGuard) plus four EOG
leads. The pipeline is intentionally simple (readability over complexity):

    1. Load EDF with MNE.
    2. Drop the four EOG leads (`EEGHEOGRCPz`, `EEGHEOGLCPz`,
       `EEGVEOGUCPz`, `EEGVEOGLCPz`).
    3. Soft bad-channel handling: any channel whose variance is < 1/100 or
       > 100x the trial median is marked bad and interpolated from
       neighbours. The known suspect list (F8, POz, AF4, AF8, F6, FC3) is
       included in this check rather than blanket-excluded -- per the
       conversation, lenient is fine for the first pass.
    4. Bandpass 1-40 Hz + 60 Hz notch (Roswell Park is on US mains).
    5. Welch PSD on the whole trial; integrate within five canonical bands.
    6. Aggregate channels into eight scalp regions (frontal/central/parietal
       /occipital x left/right), log-power, and flatten to a 40-d trial
       feature vector. Midline (`z`-suffix) channels are dropped from the
       regional means -- they don't have a clean L/R assignment and adding
       a separate "midline" group would force a one-off code path the rest
       of the pipeline doesn't need.

The output feature matrix is concatenated with metadata and fed into the
existing residualization / RDM / GW machinery via `subject_aggregation.py`.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Modality-specific constants. These mirror the dataset's documented setup.
EEG_RATE_HZ = 500.0
NOTCH_FREQ_HZ = 60.0
BANDPASS_LOW_HZ = 1.0
BANDPASS_HIGH_HZ = 40.0

# EOG channels per the dataset README. They get dropped before any
# region/bandpower computation; they can still be used as a sync diagnostic
# elsewhere (the eye_loader does that on the gaze side).
EOG_CHANNELS = (
    "EEGHEOGRCPz",
    "EEGHEOGLCPz",
    "EEGVEOGUCPz",
    "EEGVEOGLCPz",
)

# Some FLS recordings emit concatenated EOG names like `EEGHEOGLHEOGRCPz`
# (left+right horizontal in one channel) or `EEGVEOGUVEOGLCPz`. They aren't
# in EOG_CHANNELS verbatim and -- worse -- they survive the canonicalization
# step (stripping `EEG` prefix and `CPz` suffix yields `HEOGLHEOGR`, which
# isn't in any 10-20 montage), so without this regex they trigger the
# `interpolate_bads` zero-fill fallback unnecessarily.
EOG_PATTERN = re.compile(r"^(?:EEG)?[HV]EOG", re.IGNORECASE)


def _is_eog_channel(name: str) -> bool:
    """True if the channel name (raw or canonical) looks like an EOG lead.

    Catches the documented EOG_CHANNELS list and any concatenated variants
    the dataset emits (e.g. `EEGHEOGLHEOGRCPz`, `HEOGLHEOGR`).
    """
    return bool(EOG_PATTERN.match(name))

# Suspect-channel list flagged in the README. We do NOT blanket-drop these;
# instead `soft_bad_channels` adds them to the bad list when a per-trial
# variance check fails.
SUSPECT_CHANNELS = ("F8", "POz", "AF4", "AF8", "F6", "FC3")

# Five canonical EEG bands (lower inclusive, upper exclusive at the high
# bound, matching scipy convention via `freqs >= lo & freqs < hi`).
BANDS: Tuple[Tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta",  13.0, 30.0),
    ("gamma", 30.0, 40.0),
)

# Eight scalp regions: anteroposterior (frontal, central, parietal,
# occipital) x hemisphere (left, right). Ordering is fixed for the column
# layout to be deterministic.
REGION_ORDER: Tuple[str, ...] = (
    "frontal_L", "frontal_R",
    "central_L", "central_R",
    "parietal_L", "parietal_R",
    "occipital_L", "occipital_R",
)


# ---------- channel-name → region mapping ------------------------------------

def _canonical_label(name: str) -> str:
    """Strip the dataset's `EEG` prefix and `CPz` Cz-reference suffix to
    recover a standard 10-20 label. Idempotent on already-canonical names.

    The FLS EDFs name every channel as `EEG<label>CPz` (e.g. `EEGFp1CPz`,
    `EEGCzCPz`); the prefix is the modality code and the suffix encodes
    the recording reference (Cz). Region/hemisphere routing and MNE's
    `set_montage("standard_1020")` both need the bare label, so we strip
    once at the boundary and let everything downstream operate on canon.

    Examples:
        EEGFp1CPz   -> Fp1
        EEGCzCPz    -> Cz
        Fp1         -> Fp1            (no prefix/suffix to strip)
        EEGHEOGRCPz -> HEOGR          (won't match 10-20; caller drops EOG separately)
    """
    n = name
    if n.startswith("EEG"):
        n = n[3:]
    if n.endswith("CPz"):
        n = n[:-3]
    return n


def _ap_group(canonical: str) -> Optional[str]:
    """Return one of {'frontal','central','parietal','occipital'} or None.

    Matches by 10-20 prefix in this precedence:
      Fp/AF/F*    -> frontal   (Fp1, AF7, F3, FC1 ...)
      C/T*        -> central   (C3, FC5, T7, FT8 ...)   -- T grouped with C
      P/CP/PO*    -> parietal  (P3, CP5, PO7 ...)
      O*          -> occipital (O1, Oz ...)
    Channels that don't match any of these prefixes return None and are
    dropped from the regional aggregation. Expects an already-canonical
    label; callers route raw names through `_canonical_label` first.
    """
    n = canonical.upper()
    if n.startswith(("FP", "AF", "FC", "F")):
        return "frontal"
    if n.startswith(("CP",)):
        return "parietal"
    if n.startswith(("PO",)):
        return "parietal"
    if n.startswith(("C", "T", "FT")):
        return "central"
    if n.startswith("P"):
        return "parietal"
    if n.startswith("O"):
        return "occipital"
    return None


def _hemisphere(canonical: str) -> Optional[str]:
    """'L' if the canonical name's trailing number is odd, 'R' if even, None for midline."""
    digits = ""
    for ch in reversed(canonical):
        if ch.isdigit():
            digits = ch + digits
        else:
            break
    if not digits:
        return None  # midline (e.g. Cz, FCz, Oz) -- excluded from regional means
    return "L" if int(digits) % 2 == 1 else "R"


def assign_region(channel_name: str) -> Optional[str]:
    """Return e.g. 'frontal_L' or None if this channel is midline / off-cap.

    Accepts either raw FLS names (`EEGFp1CPz`) or canonical names (`Fp1`).
    """
    canon = _canonical_label(channel_name)
    ap = _ap_group(canon)
    if ap is None:
        return None
    side = _hemisphere(canon)
    if side is None:
        return None
    region = f"{ap}_{side}"
    return region if region in REGION_ORDER else None


def build_region_index(channel_names: Sequence[str]) -> Dict[str, List[int]]:
    """Return {region_name: [channel_indices]} for the eight standard regions.

    Channels with no region (EOG, midline, unknown prefix) are silently
    excluded. Empty regions yield an empty list -- callers should treat
    them as a zero-power region rather than a hard error so that one
    channel-naming oddity doesn't kill a whole trial.
    """
    out: Dict[str, List[int]] = {r: [] for r in REGION_ORDER}
    for i, ch in enumerate(channel_names):
        r = assign_region(ch)
        if r is not None:
            out[r].append(i)
    return out


# ---------- bad-channel handling ---------------------------------------------

def soft_bad_channels(
    data: np.ndarray,
    channel_names: Sequence[str],
    *,
    var_low_ratio: float = 0.01,
    var_high_ratio: float = 100.0,
) -> List[str]:
    """Return a list of channel names flagged as bad on this trial.

    A channel is flagged if its temporal variance is < `var_low_ratio` or
    > `var_high_ratio` times the median variance of all channels. The
    suspect-channel list from the README is also auto-flagged when its
    variance falls outside a slightly tighter band ([0.1, 10] x median),
    since the README says they're often (but not always) bad.

    Parameters
    ----------
    data : (n_samples, n_channels) array
    channel_names : per-column channel labels matching `data`
    """
    if data.ndim != 2:
        raise ValueError(f"expected 2-D data, got shape {data.shape}")
    var = np.var(data, axis=0)
    finite = var[np.isfinite(var) & (var > 0)]
    if finite.size == 0:
        return list(channel_names)  # everything is bad
    med = float(np.median(finite))
    bad: List[str] = []
    for i, ch in enumerate(channel_names):
        v = float(var[i])
        if not np.isfinite(v) or v <= 0:
            bad.append(ch); continue
        ratio = v / med if med > 0 else float("inf")
        if ratio < var_low_ratio or ratio > var_high_ratio:
            bad.append(ch); continue
        if ch in SUSPECT_CHANNELS and (ratio < 0.1 or ratio > 10.0):
            bad.append(ch)
    return bad


# ---------- bandpower --------------------------------------------------------

def _welch_psd(
    data: np.ndarray,
    sfreq: float,
    n_per_seg: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD per channel. Returns (freqs, psd) where psd has shape (n_channels, n_freqs)."""
    from scipy.signal import welch

    if n_per_seg is None:
        # 2 s window @ 500 Hz default; pad if the trial is shorter.
        n_per_seg = min(int(2.0 * sfreq), data.shape[0])
    f, p = welch(
        data, fs=sfreq, nperseg=n_per_seg, noverlap=n_per_seg // 2, axis=0,
    )
    # p shape: (n_freqs, n_channels) -> transpose to (n_channels, n_freqs)
    return f.astype(np.float64), p.T.astype(np.float64)


def integrate_bands(
    freqs: np.ndarray,
    psd: np.ndarray,
    bands: Sequence[Tuple[str, float, float]] = BANDS,
) -> np.ndarray:
    """Return (n_channels, n_bands) integrated power via the trapezoidal rule."""
    from scipy.integrate import trapezoid

    out = np.zeros((psd.shape[0], len(bands)), dtype=np.float64)
    for k, (_name, lo, hi) in enumerate(bands):
        mask = (freqs >= lo) & (freqs < hi)
        if not mask.any():
            out[:, k] = 0.0
            continue
        out[:, k] = trapezoid(psd[:, mask], freqs[mask], axis=1)
    return out


def regional_log_power(
    data: np.ndarray,
    channel_names: Sequence[str],
    sfreq: float,
    bands: Sequence[Tuple[str, float, float]] = BANDS,
    region_order: Sequence[str] = REGION_ORDER,
    relative: bool = True,
) -> np.ndarray:
    """Return a (n_regions * n_bands,) feature vector of log10 band power.

    With `relative=True` (default), each channel's bandpower is divided
    by that channel's total bandpower across the 5 bands BEFORE regional
    averaging. The resulting values are interpretable as "fraction of
    this channel's power that lives in this band", and they're invariant
    to per-channel and per-subject multiplicative gain factors (skin/
    skull conductance, electrode coupling, day-to-day reference drift).
    Without this normalization, PC1 of the 40-d feature vector dominated
    inter-subject variance at >95% — the cosine RDM became a sign-of-PC1
    indicator and hid any tier-relevant structure on PC2..PC40.

    With `relative=False`, the legacy behaviour: log10 of absolute
    integrated power per region/band. Kept for backward compatibility
    and to allow direct comparison against the original feature set.

    Empty regions produce log10(eps) so the column count stays fixed.
    """
    freqs, psd = _welch_psd(data, sfreq=sfreq)
    bp = integrate_bands(freqs, psd, bands)            # (n_channels, n_bands)
    if relative:
        # Per-channel normalization: divide by the channel's total band
        # power so each row sums to 1. Channels with zero or negative
        # total (rare; only on a hard-zeroed bad channel after the
        # interpolation fallback) are left at zero rather than NaN-ing.
        ch_total = bp.sum(axis=1, keepdims=True)
        safe = np.where(ch_total > 1e-12, ch_total, 1.0)
        bp = np.where(ch_total > 1e-12, bp / safe, 0.0)
    region_idx = build_region_index(channel_names)
    out = np.zeros((len(region_order), len(bands)), dtype=np.float64)
    for r_i, region in enumerate(region_order):
        idxs = region_idx.get(region, [])
        if not idxs:
            out[r_i, :] = 0.0
            continue
        out[r_i, :] = bp[idxs, :].mean(axis=0)
    out = np.log10(out + 1e-12)
    return out.flatten()


def eeg_feature_column_names(
    region_order: Sequence[str] = REGION_ORDER,
    bands: Sequence[Tuple[str, float, float]] = BANDS,
) -> List[str]:
    """Deterministic column names for the regional-bandpower feature vector."""
    return [f"eeg_{r}_{b[0]}" for r in region_order for b in bands]


# ---------- per-trial pipeline -----------------------------------------------

def _load_edf_with_mne(edf_path: Path) -> Tuple[np.ndarray, float, List[str]]:
    """Load one EDF with MNE, return (data (n_samples, n_channels), sfreq, channel_names).

    No filtering yet -- we filter after dropping EOG and marking bad channels.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    return raw.get_data().T.astype(np.float64), float(raw.info["sfreq"]), list(raw.ch_names)


def _preprocess_trial(
    edf_path: Path,
) -> Tuple[np.ndarray, float, List[str], Dict[str, object]]:
    """Load + drop EOG + canonicalize channel names + interpolate bads + filter.

    Returns (data, sfreq, ch_names, info). The returned `ch_names` are
    canonical 10-20 labels (e.g. `Fp1`), not the raw `EEGFp1CPz` form,
    so downstream regional aggregation can use them directly.
    """
    import mne

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    info_dict: Dict[str, object] = {"orig_n_channels": len(raw.ch_names)}

    # Drop EOG channels by name pattern (catches the four canonical EOG
    # leads in EOG_CHANNELS and any concatenated variants the dataset
    # emits, e.g. `EEGHEOGLHEOGRCPz`). Done BEFORE renaming so the regex
    # sees the raw `EEG...CPz` form.
    drop = [c for c in raw.ch_names if _is_eog_channel(c)]
    if drop:
        raw.drop_channels(drop)
    info_dict["dropped_eog"] = drop

    # Rename remaining channels to canonical 10-20 labels so set_montage
    # and the region/hemisphere routing both find them. Skip any rename
    # that would collide with another canonical label (defensive — the
    # FLS dataset is all-distinct after stripping `EEG...CPz`, but a
    # collision would silently break interpolation).
    canon_map = {ch: _canonical_label(ch) for ch in raw.ch_names}
    if len(set(canon_map.values())) == len(canon_map):
        raw.rename_channels(canon_map)
    else:
        log.warning("canonical channel names collide for %s; skipping rename "
                    "(set_montage will fail)", edf_path.name)
    info_dict["canonical_channels"] = list(raw.ch_names)

    # Soft bad-channel detection on the time-domain data, then ask MNE to
    # interpolate from spatial neighbours. If MNE can't interpolate
    # (no montage / digitization), fall back to zeroing the bad channels.
    data = raw.get_data().T  # (n_samples, n_channels)
    bads = soft_bad_channels(data, raw.ch_names)
    info_dict["bad_channels"] = bads
    if bads:
        raw.info["bads"] = bads
        try:
            # standard_1005 includes the AntNeuro WaveGuard 5/10 supplementary
            # positions (FCC1h, CCP3h, TPP9h, ...) that 1020 is missing, so
            # most FLS channels get a position rather than NaN, and
            # interpolate_bads can use spatial neighbours instead of falling
            # back to zeroing.
            raw.set_montage("standard_1005", on_missing="ignore", match_case=False)
            raw.interpolate_bads(reset_bads=True, verbose=False)
        except Exception as e:
            log.warning("interpolate_bads failed (%s) for %s; zeroing bad channels",
                        e, edf_path.name)
            data = raw.get_data().T
            bad_idx = [raw.ch_names.index(b) for b in bads if b in raw.ch_names]
            data[:, bad_idx] = 0.0
            raw._data = data.T

    # Bandpass + notch.
    raw.filter(l_freq=BANDPASS_LOW_HZ, h_freq=BANDPASS_HIGH_HZ, verbose=False)
    raw.notch_filter(freqs=NOTCH_FREQ_HZ, verbose=False)

    return (
        raw.get_data().T.astype(np.float64),
        float(raw.info["sfreq"]),
        list(raw.ch_names),
        info_dict,
    )


def trial_feature_vector(edf_path: Path) -> np.ndarray:
    """Run the per-trial pipeline and return the 40-d regional log-power vector."""
    data, sfreq, ch_names, _info = _preprocess_trial(edf_path)
    return regional_log_power(data, ch_names, sfreq=sfreq)


def build_fls_eeg_feature_frame(
    repo_root: Path,
    *,
    scores: pd.DataFrame | None = None,
    eeg_dir: Path | None = None,
) -> pd.DataFrame:
    """Build a (n_trials, 40 + metadata) DataFrame for the FLS EEG modality.

    Trials whose EDF is missing or fails to parse are skipped with a
    warning; the row count of the returned frame may be less than the
    number of rows in PerformanceScores.csv.
    """
    from skill_manifold.features_fls_gaze import (
        fls_eeg_dir,
        fls_performance_csv,
        load_fls_performance_scores,
    )

    if scores is None:
        scores = load_fls_performance_scores(fls_performance_csv(repo_root))
    if eeg_dir is None:
        eeg_dir = fls_eeg_dir(repo_root)

    feat_cols = eeg_feature_column_names()
    rows = []
    for _, r in scores.iterrows():
        tid = r["trial_id"]
        path = Path(eeg_dir) / f"{tid}.edf"
        if not path.exists():
            log.warning("missing FLS EEG EDF: %s", path)
            continue
        try:
            vec = trial_feature_vector(path)
        except Exception as e:
            log.warning("skip %s (%s: %s)", tid, type(e).__name__, e)
            continue
        if vec.shape[0] != len(feat_cols):
            log.warning("trial %s feature dim %d != expected %d; skipping",
                        tid, vec.shape[0], len(feat_cols))
            continue

        record = {
            "trial_id": tid,
            "subject_id": int(r["subject_id"]),
            "task_id": int(r["task_id"]),
            "try_num": int(r["try_num"]),
            "age": float(r["age"]),
            "dominant_hand": str(r["dominant_hand"]),
            "dominant_eye": str(r.get("dominant_eye", "")),
            "gender": str(r.get("gender", "")),
            "performance": float(r["performance"]),
            "perf_min": float(r.get("perf_min", np.nan)),
            "perf_max": float(r.get("perf_max", np.nan)),
        }
        record.update(dict(zip(feat_cols, vec)))
        rows.append(record)

    return pd.DataFrame(rows).sort_values("trial_id").reset_index(drop=True)
