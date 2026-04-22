"""
Offline script to precompute and cache EEG RDMs for training efficiency.
"""
import os
import sys
import argparse
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data.eeg_processor import EEGProcessor, compute_eeg_rdm_batch
from data.sync_manager import SyncManager


def find_eeg_files(data_root: Path, task: str) -> dict:
    """Find all EEG files for a given task."""
    eeg_dir = data_root / 'EEG'
    if not eeg_dir.exists():
        return {}
    
    eeg_files = {}
    for eeg_file in eeg_dir.glob('*.edf'):
        # Try to match with trial IDs
        trial_id = eeg_file.stem
        eeg_files[trial_id] = str(eeg_file)
    
    return eeg_files


def load_trial_metadata(data_root: Path, task: str) -> dict:
    """Load trial metadata."""
    meta_file = data_root / 'Gestures' / task / f'meta_file_{task}.txt'
    metadata = {}
    
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    trial_id = parts[0]
                    surgeon_id = parts[1] if len(parts) > 1 else 'Unknown'
                    skill_level = parts[2] if len(parts) > 2 else 'N'
                    metadata[trial_id] = {
                        'surgeon_id': surgeon_id,
                        'skill_level': skill_level
                    }
    
    return metadata


def extract_epochs_for_frames(
    eeg_data: np.ndarray,
    frame_indices: np.ndarray,
    sync_manager: SyncManager,
    window_size_ms: float = 100.0
) -> np.ndarray:
    """Extract EEG epochs aligned to video frames."""
    window_samples = int(window_size_ms * sync_manager.eeg_fs / 1000.0)
    half_window = window_samples // 2
    
    epochs = []
    for frame_idx in frame_indices:
        center_sample = sync_manager.frame_to_eeg_sample(frame_idx)
        start_sample = max(0, center_sample - half_window)
        end_sample = min(len(eeg_data), center_sample + half_window)
        
        epoch = eeg_data[start_sample:end_sample]
        
        # Pad if necessary
        if len(epoch) < window_samples:
            pad_before = half_window - (center_sample - start_sample)
            pad_after = half_window - (end_sample - center_sample)
            epoch = np.pad(epoch, ((pad_before, pad_after), (0, 0)), mode='edge')
        
        epochs.append(epoch[:window_samples])
    
    return np.array(epochs) if epochs else np.zeros((0, window_samples, eeg_data.shape[1]))


def precompute_rdms_for_trial(
    eeg_path: str,
    trial_id: str,
    transcription_path: str,
    sync_manager: SyncManager,
    eeg_processor: EEGProcessor,
    cache_dir: Path,
    window_size_ms: float = 100.0,
    rdm_method: str = 'pearson'
) -> dict:
    """
    Precompute RDMs for a single trial.
    
    Returns:
        Dictionary with RDM data and metadata
    """
    # Load and process EEG
    eeg_data = eeg_processor.process_eeg_file(eeg_path, apply_filtering=True, apply_ica=False)
    
    # Load transcription to get gesture segments
    gesture_segments = []
    if os.path.exists(transcription_path):
        with open(transcription_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    start_frame = int(parts[0])
                    end_frame = int(parts[1])
                    gesture = parts[2]
                    gesture_segments.append({
                        'start_frame': start_frame,
                        'end_frame': end_frame,
                        'gesture': gesture
                    })
    
    # Extract epochs for each gesture segment
    all_epochs = []
    all_gestures = []
    all_frame_indices = []
    
    for segment in gesture_segments:
        start_frame = segment['start_frame']
        end_frame = segment['end_frame']
        
        # Sample frames (every 5 frames for efficiency)
        frame_indices = np.arange(start_frame, end_frame + 1, 5)
        
        # Extract EEG epochs
        epochs = extract_epochs_for_frames(
            eeg_data, frame_indices, sync_manager, window_size_ms
        )
        
        if len(epochs) > 0:
            all_epochs.append(epochs)
            all_gestures.extend([segment['gesture']] * len(epochs))
            all_frame_indices.extend(frame_indices.tolist())
    
    if len(all_epochs) == 0:
        return None
    
    # Concatenate all epochs
    all_epochs = np.concatenate(all_epochs, axis=0)  # (n_stimuli, n_samples, n_channels)
    
    # Compute RDM
    # Average over time to get features per stimulus
    features = all_epochs.mean(axis=1)  # (n_stimuli, n_channels)
    rdm = eeg_processor.compute_rdm(features, method=rdm_method, trial_average=False)
    
    # Save to cache
    cache_file = cache_dir / f"{trial_id}_rdm.pkl"
    cache_data = {
        'rdm': rdm,
        'features': features,
        'gestures': all_gestures,
        'frame_indices': all_frame_indices,
        'trial_id': trial_id,
        'window_size_ms': window_size_ms,
        'method': rdm_method
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)
    
    return cache_data


def main():
    parser = argparse.ArgumentParser(description='Precompute EEG RDMs for offline use')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing Gestures/ and EEG/ folders')
    parser.add_argument('--task', type=str, default='Knot_Tying',
                       choices=['Knot_Tying', 'Needle_Passing', 'Suturing'],
                       help='Task name')
    parser.add_argument('--cache_dir', type=str, default='cache/eeg_rdms',
                       help='Directory to save cached RDMs')
    parser.add_argument('--window_size_ms', type=float, default=100.0,
                       help='EEG window size in milliseconds')
    parser.add_argument('--rdm_method', type=str, default='pearson',
                       choices=['pearson', 'euclidean', 'cosine'],
                       help='RDM computation method')
    parser.add_argument('--sampling_rate', type=float, default=1000.0,
                       help='EEG sampling rate (Hz)')
    parser.add_argument('--low_freq', type=float, default=1.0,
                       help='Low cutoff frequency for bandpass filter')
    parser.add_argument('--high_freq', type=float, default=40.0,
                       help='High cutoff frequency for bandpass filter')
    
    args = parser.parse_args()
    
    data_root = Path(args.data_root)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    eeg_processor = EEGProcessor(
        sampling_rate=args.sampling_rate,
        low_freq=args.low_freq,
        high_freq=args.high_freq
    )
    sync_manager = SyncManager(video_fps=30.0, eeg_fs=args.sampling_rate)
    
    # Find EEG files
    eeg_files = find_eeg_files(data_root, args.task)
    print(f"Found {len(eeg_files)} EEG files")
    
    # Load metadata
    metadata = load_trial_metadata(data_root, args.task)
    
    # Get transcription directory
    transcription_dir = data_root / 'Gestures' / args.task / 'transcriptions'
    
    # Process each EEG file
    processed = 0
    failed = 0
    
    for trial_id, eeg_path in tqdm(eeg_files.items(), desc='Processing EEG files'):
        transcription_path = transcription_dir / f"{trial_id}.txt"
        
        if not transcription_path.exists():
            print(f"Warning: No transcription found for {trial_id}, skipping")
            failed += 1
            continue
        
        try:
            result = precompute_rdms_for_trial(
                eeg_path=eeg_path,
                trial_id=trial_id,
                transcription_path=str(transcription_path),
                sync_manager=sync_manager,
                eeg_processor=eeg_processor,
                cache_dir=cache_dir,
                window_size_ms=args.window_size_ms,
                rdm_method=args.rdm_method
            )
            
            if result is not None:
                processed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"Error processing {trial_id}: {e}")
            failed += 1
    
    print(f"\nCompleted: {processed} successful, {failed} failed")
    print(f"RDMs cached in: {cache_dir}")


if __name__ == '__main__':
    main()

