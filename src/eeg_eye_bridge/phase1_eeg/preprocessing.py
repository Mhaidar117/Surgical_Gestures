"""
Simulator-only EEG preprocessing: load, filter, channel selection, sliding windows.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Load ``src/data/eeg_processor.py`` without importing ``data`` package (avoids cv2, etc.).
_SRC = Path(__file__).resolve().parents[2]
_EEG_PROC_PATH = _SRC / "data" / "eeg_processor.py"
_spec = importlib.util.spec_from_file_location("phase1_eeg_processor", _EEG_PROC_PATH)
_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_mod)
EEGProcessor = _mod.EEGProcessor
MNE_AVAILABLE = _mod.MNE_AVAILABLE


def _load_edf_pyedflib_numpy(edf_path: Path) -> Tuple[np.ndarray, float]:
    """Load EDF with pyedflib (same layout as ``EEGProcessor`` when MNE is absent)."""
    import pyedflib

    f = pyedflib.EdfReader(str(edf_path))
    try:
        n_channels = f.signals_in_file
        n_samples = f.getNSamples()[0]
        data = np.zeros((n_samples, n_channels), dtype=np.float64)
        for i in range(n_channels):
            data[:, i] = f.readSignal(i)
        sfreq = float(f.getSampleFrequency(0))
        return data.astype(np.float32), sfreq
    finally:
        f.close()


def load_eeg_preprocessed(
    edf_path: Path,
    processor: EEGProcessor,
    apply_filtering: bool = True,
    apply_ica: bool = False,
) -> Tuple[np.ndarray, float]:
    """
    Load EDF as (n_samples, n_channels), update ``processor.sampling_rate``.

    If MNE is available, keep EEG channels only via ``pick_types``; otherwise use
    ``EEGProcessor.load_edf`` (all channels).

    If MNE rejects the file (corrupt header, etc.), tries ``pyedflib`` when installed
    and uses all signals (no EEG-only pick).
    """
    edf_path = Path(edf_path)
    if MNE_AVAILABLE:
        import mne

        raw = None
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except (ValueError, OSError):
            raw = None

        if raw is not None:
            picks = mne.pick_types(raw.info, eeg=True, exclude=[])
            if len(picks) > 0:
                raw.pick(picks)
            data = raw.get_data().T
            sfreq = float(raw.info["sfreq"])
            processor.sampling_rate = sfreq
            if apply_filtering:
                data = processor.apply_notch_filter(data)
                data = processor.apply_bandpass_filter(data)
            if apply_ica:
                data = processor.apply_ica_denoising(data)
            return data.astype(np.float32), sfreq

        try:
            data, sfreq = _load_edf_pyedflib_numpy(edf_path)
        except ImportError as e:
            raise ValueError(
                f"MNE could not read EDF ({edf_path}); install pyedflib for an alternate reader"
            ) from e
        except Exception as e:
            raise ValueError(f"Bad or unreadable EDF file: {edf_path}") from e
        processor.sampling_rate = sfreq
        if apply_filtering:
            data = processor.apply_notch_filter(data)
            data = processor.apply_bandpass_filter(data)
        if apply_ica:
            data = processor.apply_ica_denoising(data)
        return data.astype(np.float32), sfreq

    data = processor.process_eeg_file(
        edf_path, apply_filtering=apply_filtering, apply_ica=apply_ica
    )
    return data.astype(np.float32), float(processor.sampling_rate)


def sliding_windows(
    eeg: np.ndarray,
    sfreq: float,
    window_sec: float,
    hop_sec: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Non-overlapping or overlapping sliding windows over a full trial.

    Args:
        eeg: (n_samples, n_channels)
        sfreq: sampling rate (Hz)
        window_sec: window length in seconds
        hop_sec: hop in seconds

    Returns:
        windows: (n_windows, n_channels, n_samples_in_window) float32
        window_times: (n_windows, 2) float64, ``[start_sec, end_sec]`` per window
    """
    n_samples, n_ch = eeg.shape
    win = int(round(window_sec * sfreq))
    hop = int(round(hop_sec * sfreq))
    if win <= 0:
        raise ValueError("window_sec too small for sampling rate")
    if hop <= 0:
        hop = win

    x = eeg.astype(np.float32, copy=False)
    if n_samples < win:
        pad = win - n_samples
        x = np.pad(x, ((0, pad), (0, 0)), mode="edge")
        n_samples = x.shape[0]

    starts: List[int] = []
    s = 0
    while s + win <= n_samples:
        starts.append(s)
        s += hop
    if not starts:
        starts = [0]
        x = np.pad(x, ((0, win - n_samples), (0, 0)), mode="edge")

    chunks = [x[s : s + win] for s in starts]
    w_arr = np.stack(chunks, axis=0)
    w_arr = np.transpose(w_arr, (0, 2, 1))
    times = np.array([(st / sfreq, (st + win) / sfreq) for st in starts], dtype=np.float64)
    return w_arr.astype(np.float32), times


def synthetic_trial(
    n_samples: int = 5000,
    n_channels: int = 32,
    sfreq: float = 500.0,
    seed: int = 0,
) -> Tuple[np.ndarray, float]:
    """Deterministic pseudo-EEG for tests when EDF files are absent."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples, dtype=np.float64) / sfreq
    x = np.zeros((n_samples, n_channels), dtype=np.float32)
    for c in range(n_channels):
        f = 8.0 + c * 0.3
        x[:, c] = (0.5 * np.sin(2 * np.pi * f * t) + 0.1 * rng.standard_normal(n_samples)).astype(
            np.float32
        )
    return x, sfreq


def build_processor(
    low_freq: float = 1.0,
    high_freq: float = 40.0,
    notch_freq: Optional[float] = 50.0,
    sampling_rate: float = 500.0,
) -> EEGProcessor:
    return EEGProcessor(
        sampling_rate=sampling_rate,
        low_freq=low_freq,
        high_freq=high_freq,
        notch_freq=notch_freq,
        ica_components=None,
    )
