"""
Temporal alignment utilities for synchronizing video frames, kinematics, and EEG data.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import interpolate


class SyncManager:
    """
    Manages temporal synchronization between video frames, kinematics, and EEG epochs.
    """
    
    def __init__(self, video_fps: float = 30.0, kinematics_fps: Optional[float] = None, eeg_fs: Optional[float] = None):
        """
        Args:
            video_fps: Video frame rate (Hz)
            kinematics_fps: Kinematics sampling rate (Hz), if None assumes same as video
            eeg_fs: EEG sampling frequency (Hz), if None assumes 1000 Hz
        """
        self.video_fps = video_fps
        self.kinematics_fps = kinematics_fps or video_fps
        self.eeg_fs = eeg_fs or 1000.0
    
    def frame_to_time(self, frame_idx: int) -> float:
        """Convert frame index to time in seconds."""
        return frame_idx / self.video_fps
    
    def time_to_frame(self, time: float) -> int:
        """Convert time in seconds to frame index."""
        return int(time * self.video_fps)
    
    def frame_to_kinematics_idx(self, frame_idx: int) -> int:
        """Convert frame index to kinematics sample index."""
        time = self.frame_to_time(frame_idx)
        return int(time * self.kinematics_fps)
    
    def kinematics_idx_to_frame(self, kin_idx: int) -> int:
        """Convert kinematics sample index to frame index."""
        time = kin_idx / self.kinematics_fps
        return self.time_to_frame(time)
    
    def frame_to_eeg_sample(self, frame_idx: int) -> int:
        """Convert frame index to EEG sample index."""
        time = self.frame_to_time(frame_idx)
        return int(time * self.eeg_fs)
    
    def eeg_sample_to_frame(self, eeg_sample: int) -> int:
        """Convert EEG sample index to frame index."""
        time = eeg_sample / self.eeg_fs
        return self.time_to_frame(time)
    
    def resample_kinematics_to_frames(
        self, 
        kinematics: np.ndarray, 
        frame_indices: np.ndarray,
        method: str = 'linear'
    ) -> np.ndarray:
        """
        Resample kinematics to match frame indices.
        
        Args:
            kinematics: Array of shape (N_kin, D) where N_kin is number of kinematics samples
            frame_indices: Array of frame indices to resample to
            method: Interpolation method ('linear', 'cubic', 'nearest')
        
        Returns:
            Resampled kinematics of shape (len(frame_indices), D)
        """
        if len(kinematics) == 0:
            return np.zeros((len(frame_indices), kinematics.shape[1] if len(kinematics.shape) > 1 else 1))
        
        # Create time points for kinematics
        kin_times = np.arange(len(kinematics)) / self.kinematics_fps
        
        # Create time points for target frames
        frame_times = frame_indices / self.video_fps
        
        # Handle case where kinematics is 1D
        if len(kinematics.shape) == 1:
            kinematics = kinematics.reshape(-1, 1)
        
        # Interpolate each dimension
        resampled = np.zeros((len(frame_times), kinematics.shape[1]))
        for dim in range(kinematics.shape[1]):
            if method == 'linear':
                f = interpolate.interp1d(kin_times, kinematics[:, dim], kind='linear', 
                                        bounds_error=False, fill_value='extrapolate')
            elif method == 'cubic':
                f = interpolate.interp1d(kin_times, kinematics[:, dim], kind='cubic',
                                        bounds_error=False, fill_value='extrapolate')
            else:  # nearest
                f = interpolate.interp1d(kin_times, kinematics[:, dim], kind='nearest',
                                        bounds_error=False, fill_value='extrapolate')
            resampled[:, dim] = f(frame_times)
        
        return resampled
    
    def get_eeg_window_for_frames(
        self,
        frame_indices: np.ndarray,
        eeg_data: np.ndarray,
        window_size_ms: float = 100.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract EEG windows aligned to frame indices.
        
        Args:
            frame_indices: Array of frame indices
            eeg_data: EEG data of shape (N_eeg, C) where N_eeg is number of samples
            window_size_ms: Window size in milliseconds
        
        Returns:
            Tuple of (eeg_windows, window_centers) where:
            - eeg_windows: Array of shape (len(frame_indices), window_samples, C)
            - window_centers: Array of EEG sample indices corresponding to frame centers
        """
        window_samples = int(window_size_ms * self.eeg_fs / 1000.0)
        half_window = window_samples // 2
        
        eeg_windows = []
        window_centers = []
        
        for frame_idx in frame_indices:
            center_sample = self.frame_to_eeg_sample(frame_idx)
            start_sample = max(0, center_sample - half_window)
            end_sample = min(len(eeg_data), center_sample + half_window)
            
            # Extract window
            window = eeg_data[start_sample:end_sample]
            
            # Pad if necessary
            if len(window) < window_samples:
                pad_before = half_window - (center_sample - start_sample)
                pad_after = half_window - (end_sample - center_sample)
                window = np.pad(window, ((pad_before, pad_after), (0, 0)), mode='edge')
            
            eeg_windows.append(window[:window_samples])
            window_centers.append(center_sample)
        
        return np.array(eeg_windows), np.array(window_centers)
    
    def create_timestamp_map(
        self,
        video_start_time: float = 0.0,
        num_frames: int = 0,
        kinematics_start_time: Optional[float] = None,
        eeg_start_time: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create a mapping of timestamps for all modalities.
        
        Returns:
            Dictionary with keys 'video', 'kinematics', 'eeg' containing timestamp arrays
        """
        video_times = np.arange(num_frames) / self.video_fps + video_start_time
        
        if kinematics_start_time is None:
            kinematics_start_time = video_start_time
        num_kinematics = int((video_times[-1] - kinematics_start_time) * self.kinematics_fps) + 1
        kinematics_times = np.arange(num_kinematics) / self.kinematics_fps + kinematics_start_time
        
        if eeg_start_time is None:
            eeg_start_time = video_start_time
        num_eeg = int((video_times[-1] - eeg_start_time) * self.eeg_fs) + 1
        eeg_times = np.arange(num_eeg) / self.eeg_fs + eeg_start_time
        
        return {
            'video': video_times,
            'kinematics': kinematics_times,
            'eeg': eeg_times
        }

