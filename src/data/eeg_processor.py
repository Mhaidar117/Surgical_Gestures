"""
EEG processing pipeline: loading, filtering, epoching, and RDM computation.
"""
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import pickle
from scipy import signal
from scipy.stats import spearmanr
import warnings

try:
    import mne
    from mne.preprocessing import ICA
    MNE_AVAILABLE = True
except ImportError:
    try:
        import pyedflib
        PYEDFLIB_AVAILABLE = True
        MNE_AVAILABLE = False
    except ImportError:
        MNE_AVAILABLE = False
        PYEDFLIB_AVAILABLE = False
        warnings.warn("Neither mne nor pyedflib available. EEG processing will be limited.")


class EEGProcessor:
    """
    Processes EEG data: loading, filtering, epoching, and RDM computation.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        low_freq: float = 1.0,
        high_freq: float = 40.0,
        notch_freq: Optional[float] = 50.0,
        ica_components: Optional[int] = None
    ):
        """
        Args:
            sampling_rate: EEG sampling frequency (Hz)
            low_freq: Low cutoff frequency for bandpass filter (Hz)
            high_freq: High cutoff frequency for bandpass filter (Hz)
            notch_freq: Notch filter frequency (Hz), None to disable
            ica_components: Number of ICA components to use, None to disable ICA
        """
        self.sampling_rate = sampling_rate
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.notch_freq = notch_freq
        self.ica_components = ica_components
    
    def load_edf(self, edf_path: Union[str, Path]) -> np.ndarray:
        """
        Load EEG data from EDF file.
        
        Args:
            edf_path: Path to EDF file
        
        Returns:
            EEG data array of shape (n_samples, n_channels)
        """
        if MNE_AVAILABLE:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            data = raw.get_data().T  # (n_samples, n_channels)
            self.sampling_rate = raw.info['sfreq']
            return data
        elif PYEDFLIB_AVAILABLE:
            f = pyedflib.EdfReader(str(edf_path))
            n_channels = f.signals_in_file
            n_samples = f.getNSamples()[0]
            data = np.zeros((n_samples, n_channels))
            for i in range(n_channels):
                data[:, i] = f.readSignal(i)
            f.close()
            return data
        else:
            raise ImportError("Need mne or pyedflib to load EDF files")
    
    def apply_bandpass_filter(
        self,
        eeg_data: np.ndarray,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply bandpass filter to EEG data.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            low_freq: Low cutoff frequency, None uses self.low_freq
            high_freq: High cutoff frequency, None uses self.high_freq
        
        Returns:
            Filtered EEG data
        """
        low_freq = low_freq or self.low_freq
        high_freq = high_freq or self.high_freq
        
        # Design bandpass filter
        nyquist = self.sampling_rate / 2.0
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, eeg_data, axis=0)
        
        return filtered
    
    def apply_notch_filter(
        self,
        eeg_data: np.ndarray,
        notch_freq: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply notch filter to remove line noise.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            notch_freq: Notch frequency, None uses self.notch_freq
        
        Returns:
            Filtered EEG data
        """
        if notch_freq is None:
            notch_freq = self.notch_freq
        
        if notch_freq is None:
            return eeg_data
        
        # Design notch filter
        nyquist = self.sampling_rate / 2.0
        freq = notch_freq / nyquist
        b, a = signal.iirnotch(freq, 30)  # Q=30
        filtered = signal.filtfilt(b, a, eeg_data, axis=0)
        
        return filtered
    
    def apply_ica_denoising(
        self,
        eeg_data: np.ndarray,
        n_components: Optional[int] = None
    ) -> np.ndarray:
        """
        Apply ICA for denoising (if MNE is available).
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            n_components: Number of ICA components, None uses self.ica_components
        
        Returns:
            Denoised EEG data
        """
        if not MNE_AVAILABLE or n_components is None:
            return eeg_data
        
        n_components = n_components or self.ica_components
        if n_components is None:
            return eeg_data
        
        # Create MNE RawArray
        info = mne.create_info(
            ch_names=[f'EEG{i:03d}' for i in range(eeg_data.shape[1])],
            sfreq=self.sampling_rate,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(eeg_data.T, info, verbose=False)
        
        # Fit ICA
        ica = ICA(n_components=n_components, random_state=42, verbose=False)
        ica.fit(raw)
        
        # Apply ICA
        raw_ica = ica.apply(raw, verbose=False)
        denoised = raw_ica.get_data().T
        
        return denoised
    
    def extract_epochs(
        self,
        eeg_data: np.ndarray,
        event_times: np.ndarray,
        epoch_duration: float = 1.0,
        baseline_duration: float = 0.1
    ) -> np.ndarray:
        """
        Extract epochs around event times.
        
        Args:
            eeg_data: EEG data of shape (n_samples, n_channels)
            event_times: Event times in samples
            epoch_duration: Duration of each epoch in seconds
            baseline_duration: Baseline duration for correction in seconds
        
        Returns:
            Epochs array of shape (n_events, n_samples_per_epoch, n_channels)
        """
        n_samples_per_epoch = int(epoch_duration * self.sampling_rate)
        n_baseline_samples = int(baseline_duration * self.sampling_rate)
        
        epochs = []
        for event_time in event_times:
            start_idx = int(event_time) - n_baseline_samples
            end_idx = start_idx + n_samples_per_epoch
            
            if start_idx >= 0 and end_idx <= len(eeg_data):
                epoch = eeg_data[start_idx:end_idx]
                
                # Baseline correction
                if n_baseline_samples > 0:
                    baseline = epoch[:n_baseline_samples].mean(axis=0, keepdims=True)
                    epoch = epoch - baseline
                
                epochs.append(epoch)
        
        return np.array(epochs) if epochs else np.zeros((0, n_samples_per_epoch, eeg_data.shape[1]))
    
    def compute_rdm(
        self,
        eeg_data: np.ndarray,
        method: str = 'pearson',
        trial_average: bool = False
    ) -> np.ndarray:
        """
        Compute Representational Dissimilarity Matrix (RDM) from EEG data.
        
        Args:
            eeg_data: EEG data of shape (n_stimuli, n_features) or (n_trials, n_stimuli, n_features)
            method: Distance method ('pearson', 'euclidean', 'cosine')
            trial_average: If True and eeg_data is 3D, average across trials first
        
        Returns:
            RDM matrix of shape (n_stimuli, n_stimuli)
        """
        if len(eeg_data.shape) == 3 and trial_average:
            eeg_data = eeg_data.mean(axis=0)
        
        if len(eeg_data.shape) == 3:
            # Flatten trials and stimuli: (n_trials, n_stimuli, n_features) -> (n_trials * n_stimuli, n_features)
            n_trials, n_stimuli, n_features = eeg_data.shape
            eeg_data = eeg_data.reshape(n_trials * n_stimuli, n_features)
        
        n_stimuli = eeg_data.shape[0]
        rdm = np.zeros((n_stimuli, n_stimuli))
        
        if method == 'pearson':
            for i in range(n_stimuli):
                for j in range(n_stimuli):
                    if i == j:
                        rdm[i, j] = 0.0
                    else:
                        # 1 - Pearson correlation = dissimilarity
                        corr, _ = spearmanr(eeg_data[i], eeg_data[j])
                        rdm[i, j] = 1.0 - corr if not np.isnan(corr) else 1.0
        
        elif method == 'euclidean':
            for i in range(n_stimuli):
                for j in range(n_stimuli):
                    rdm[i, j] = np.linalg.norm(eeg_data[i] - eeg_data[j])
        
        elif method == 'cosine':
            for i in range(n_stimuli):
                for j in range(n_stimuli):
                    if i == j:
                        rdm[i, j] = 0.0
                    else:
                        dot_product = np.dot(eeg_data[i], eeg_data[j])
                        norm_i = np.linalg.norm(eeg_data[i])
                        norm_j = np.linalg.norm(eeg_data[j])
                        if norm_i > 0 and norm_j > 0:
                            cosine_sim = dot_product / (norm_i * norm_j)
                            rdm[i, j] = 1.0 - cosine_sim
                        else:
                            rdm[i, j] = 1.0
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return rdm
    
    def process_eeg_file(
        self,
        edf_path: Union[str, Path],
        apply_filtering: bool = True,
        apply_ica: bool = False
    ) -> np.ndarray:
        """
        Complete EEG processing pipeline.
        
        Args:
            edf_path: Path to EDF file
            apply_filtering: Whether to apply bandpass and notch filters
            apply_ica: Whether to apply ICA denoising
        
        Returns:
            Processed EEG data of shape (n_samples, n_channels)
        """
        # Load
        eeg_data = self.load_edf(edf_path)
        
        # Filtering
        if apply_filtering:
            eeg_data = self.apply_notch_filter(eeg_data)
            eeg_data = self.apply_bandpass_filter(eeg_data)
        
        # ICA
        if apply_ica and MNE_AVAILABLE:
            eeg_data = self.apply_ica_denoising(eeg_data)
        
        return eeg_data


def compute_eeg_rdm_batch(
    eeg_epochs: np.ndarray,
    window_size_ms: float = 100.0,
    method: str = 'pearson',
    trial_average: bool = True
) -> np.ndarray:
    """
    Compute RDM for a batch of EEG epochs.
    
    Args:
        eeg_epochs: EEG epochs of shape (n_trials, n_stimuli, n_samples, n_channels) or (n_stimuli, n_samples, n_channels)
        window_size_ms: Window size in milliseconds for feature extraction
        method: RDM computation method
        trial_average: Whether to average across trials
    
    Returns:
        RDM matrix of shape (n_stimuli, n_stimuli)
    """
    processor = EEGProcessor()
    
    # Extract features from each epoch (average over time window)
    if len(eeg_epochs.shape) == 4:
        # (n_trials, n_stimuli, n_samples, n_channels)
        n_trials, n_stimuli, n_samples, n_channels = eeg_epochs.shape
        window_samples = int(window_size_ms * processor.sampling_rate / 1000.0)
        
        # Average over time window for each stimulus
        features = eeg_epochs[:, :, :window_samples, :].mean(axis=2)  # (n_trials, n_stimuli, n_channels)
        
        if trial_average:
            features = features.mean(axis=0)  # (n_stimuli, n_channels)
        else:
            features = features.reshape(n_trials * n_stimuli, n_channels)
    
    elif len(eeg_epochs.shape) == 3:
        # (n_stimuli, n_samples, n_channels)
        n_stimuli, n_samples, n_channels = eeg_epochs.shape
        window_samples = int(window_size_ms * processor.sampling_rate / 1000.0)
        features = eeg_epochs[:, :window_samples, :].mean(axis=1)  # (n_stimuli, n_channels)
    
    else:
        raise ValueError(f"Unexpected eeg_epochs shape: {eeg_epochs.shape}")
    
    # Compute RDM
    rdm = processor.compute_rdm(features, method=method, trial_average=False)
    
    return rdm

