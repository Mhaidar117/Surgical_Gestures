"""
Unified dataset class for JIGSAWS data with RGB, flow, kinematics, and EEG support.
Maintains backward compatibility with existing blob format.
"""
import os
import pickle
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from .sync_manager import SyncManager
from .transforms_vit import get_vit_transforms, sample_temporal_windows


class JIGSAWSViTDataset(Dataset):
    """
    Unified dataset for JIGSAWS surgical gesture data.
    Supports both new format (RGB + RAFT + metadata) and legacy format (Farnebäck flow).
    """
    
    def __init__(
        self,
        data_root: str,
        task: str = 'Knot_Tying',
        split: Optional[str] = None,
        mode: str = 'train',
        task_type: str = 'gesture',  # 'gesture', 'skill', 'kinematics'
        use_rgb: bool = True,
        use_flow: bool = True,
        use_eeg: bool = False,
        legacy_blobs_path: Optional[str] = None,
        video_fps: float = 30.0,
        transforms: Optional[callable] = None,
        cache_dir: Optional[str] = None,
        arm: str = 'PSM2'
    ):
        """
        Args:
            data_root: Root directory containing Gestures/ folder
            task: Task name ('Knot_Tying', 'Needle_Passing', 'Suturing')
            split: Split name (e.g., 'fold_1') for LOUO, None uses all data
            mode: 'train' or 'val'
            task_type: Type of task ('gesture', 'skill', 'kinematics')
            use_rgb: Whether to load RGB frames
            use_flow: Whether to load optical flow
            use_eeg: Whether to load EEG data
            legacy_blobs_path: Path to legacy blob folder (for backward compatibility)
            video_fps: Video frame rate
            transforms: Optional transform pipeline
            cache_dir: Directory for caching features
        """
        self.data_root = Path(data_root)
        self.task = task
        self.split = split
        self.mode = mode
        self.task_type = task_type
        self.use_rgb = use_rgb
        self.use_flow = use_flow
        self.use_eeg = use_eeg
        self.video_fps = video_fps
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.arm = arm
        
        # Setup paths
        self.task_dir = self.data_root / 'Gestures' / task
        self.video_dir = self.task_dir / 'video'
        self.kinematics_dir = self.task_dir / 'kinematics' / 'AllGestures'
        self.transcriptions_dir = self.task_dir / 'transcriptions'
        self.meta_file = self.task_dir / f'meta_file_{task}.txt'
        self.eeg_dir = self.data_root / 'EEG' if use_eeg else None
        
        # Load metadata
        self.metadata = self._load_metadata()
        
        # Setup sync manager
        self.sync_manager = SyncManager(video_fps=video_fps)
        
        # Setup transforms
        if transforms is None:
            self.transforms = get_vit_transforms(mode=mode)
        else:
            self.transforms = transforms
        
        # Load data samples
        if legacy_blobs_path and os.path.exists(legacy_blobs_path):
            # Legacy mode: load from existing blobs
            self.samples = self._load_legacy_blobs(legacy_blobs_path)
        else:
            # New mode: load from raw data
            self.samples = self._load_samples_from_raw()
    
    def _load_metadata(self) -> Dict:
        """Load metadata from meta file."""
        metadata = {}
        if self.meta_file.exists():
            with open(self.meta_file, 'r') as f:
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
    
    def _load_legacy_blobs(self, blobs_path: str) -> List[Dict]:
        """Load samples from legacy blob format."""
        samples = []
        blob_files = sorted([f for f in os.listdir(blobs_path) if f.endswith('.p')])
        
        for blob_file in blob_files:
            blob_path = os.path.join(blobs_path, blob_file)
            try:
                # Parse blob filename: blob_N_video_TRIAL_gesture_GESTURE.p
                parts = blob_file.replace('.p', '').split('_')
                if len(parts) >= 5:
                    trial_id = '_'.join(parts[2:-1])  # Handle multi-part trial IDs
                    gesture = parts[-1]
                    
                    # Extract metadata
                    surgeon_id = trial_id.split('_')[2][0] if len(trial_id.split('_')) > 2 else 'Unknown'
                    
                    samples.append({
                        'blob_path': blob_path,
                        'trial_id': trial_id,
                        'gesture': gesture,
                        'surgeon_id': surgeon_id,
                        'legacy': True
                    })
            except Exception as e:
                print(f"Warning: Could not parse blob file {blob_file}: {e}")
                continue
        
        return samples
    
    def _load_samples_from_raw(self) -> List[Dict]:
        """Load samples from raw video/kinematics/transcription files."""
        samples = []
        
        # Get all transcription files
        transcription_files = sorted([f for f in os.listdir(self.transcriptions_dir) 
                                     if f.endswith('.txt')])
        
        for trans_file in transcription_files:
            trial_id = trans_file.replace('.txt', '')
            
            # Check if trial is in split (if split is specified)
            if self.split is not None:
                # This will be handled by split_loader, for now include all
                pass
            
            # Load transcription
            trans_path = self.transcriptions_dir / trans_file
            with open(trans_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        start_frame = int(parts[0])
                        end_frame = int(parts[1])
                        gesture = parts[2]
                        
                        # Get metadata
                        surgeon_id = self.metadata.get(trial_id, {}).get('surgeon_id', 'Unknown')
                        skill_level = self.metadata.get(trial_id, {}).get('skill_level', 'N')
                        
                        # Map skill level to numeric
                        skill_dict = {'N': 0, 'I': 1, 'E': 2}
                        skill_label = skill_dict.get(skill_level, 0)
                        
                        # Find video file
                        video_file = None
                        for vf in os.listdir(self.video_dir):
                            if trial_id in vf and vf.endswith('.avi'):
                                video_file = vf
                                break
                        
                        if video_file is None:
                            continue
                        
                        video_path = self.video_dir / video_file
                        kinematics_path = self.kinematics_dir / f"{trial_id}.txt"
                        
                        # EEG path (if available)
                        eeg_path = None
                        if self.use_eeg and self.eeg_dir:
                            # Look for matching EEG file (format: trial_id.edf or similar)
                            for ef in os.listdir(self.eeg_dir):
                                if trial_id in ef and ef.endswith('.edf'):
                                    eeg_path = self.eeg_dir / ef
                                    break
                        
                        samples.append({
                            'trial_id': trial_id,
                            'video_path': str(video_path),
                            'kinematics_path': str(kinematics_path),
                            'eeg_path': str(eeg_path) if eeg_path else None,
                            'start_frame': start_frame,
                            'end_frame': end_frame,
                            'gesture': gesture,
                            'surgeon_id': surgeon_id,
                            'skill_label': skill_label,
                            'legacy': False
                        })
        
        return samples
    
    def _load_video_frames(self, video_path: str, start_frame: int, end_frame: int) -> np.ndarray:
        """Load RGB frames from video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Seek to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for i in range(start_frame, min(end_frame + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                break
        
        cap.release()
        return np.array(frames) if frames else np.zeros((1, 224, 224, 3), dtype=np.uint8)
    
    def _load_legacy_blob_data(self, blob_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load data from legacy blob format."""
        with open(blob_path, 'rb') as f:
            flow_tensor, kin_tensor = pickle.load(f)
        
        # Convert legacy flow format (50 channels) to (T, 2, H, W)
        if flow_tensor.shape[0] == 50:
            T = 25
            flow_tensor = flow_tensor.view(T, 2, 240, 320)
        
        return flow_tensor, kin_tensor
    
    def _load_kinematics(self, kinematics_path: str, start_frame: int, end_frame: int) -> np.ndarray:
        """Load kinematics data."""
        if not os.path.exists(kinematics_path):
            return np.zeros((end_frame - start_frame + 1, 76))
        
        kinematics_list = []
        with open(kinematics_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    try:
                        kinematics_list.append([float(v) for v in parts])
                    except:
                        continue
        
        if not kinematics_list:
            return np.zeros((end_frame - start_frame + 1, 76))
        
        kinematics = np.array(kinematics_list)
        
        # Resample to match frames
        frame_indices = np.arange(start_frame, min(end_frame + 1, len(kinematics)))
        if len(frame_indices) == 0:
            frame_indices = np.arange(start_frame, start_frame + 1)
        
        resampled = self.sync_manager.resample_kinematics_to_frames(
            kinematics, frame_indices, method='linear'
        )
        
        return resampled
    def _extract_arm_kinematics(self, kinematics_76: np.ndarray) -> np.ndarray:
        """
        Extract raw kinematics for the specified arm (PSM1 or PSM2).

        JIGSAWS 76-dim format:
        - Columns 1-38: Master manipulators (MTM Left + MTM Right)
        - Columns 39-57 (0-indexed 38-56): Slave left (PSM1) - 19 dims
        - Columns 58-76 (0-indexed 57-75): Slave right (PSM2) - 19 dims

        Each arm's 19 columns:
        - 0-2: position (x, y, z)
        - 3-11: rotation matrix (9D, row-major)
        - 12-14: translational velocity
        - 15-17: rotational velocity
        - 18: gripper angle

        Output: raw 19-dim kinematics for the specified arm
        """
        if self.arm == 'PSM2':
            # Extract PSM2 columns (58-76, 0-indexed: 57-76)
            arm_kinematics = kinematics_76[:, 57:76]  # Shape: (T, 19)
        elif self.arm == 'PSM1':
            # Extract PSM1 columns (39-57, 0-indexed: 38-57)
            arm_kinematics = kinematics_76[:, 38:57]  # Shape: (T, 19)
        else:
            raise ValueError(f"Unsupported arm: {self.arm}. Use 'PSM1' or 'PSM2'.")

        return arm_kinematics
    
    def _get_gesture_label(self, gesture: str) -> int:
        """Convert gesture string to label index."""
        gesture_map = {
            'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4,
            'G6': 5, 'G7': 6, 'G8': 7, 'G9': 8, 'G10': 9,
            'G11': 10, 'G12': 11, 'G13': 12, 'G14': 13, 'G15': 14
        }
        return gesture_map.get(gesture, 0)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample."""
        sample = self.samples[idx]
        
        if sample.get('legacy', False):
            # Legacy blob format
            flow_tensor, kin_tensor = self._load_legacy_blob_data(sample['blob_path'])
            
            # Convert to RGB if needed (create dummy RGB from flow)
            if self.use_rgb:
                # For legacy data, we don't have RGB, so we'll use zeros
                # In practice, you'd want to load the actual video
                rgb_frames = torch.zeros((flow_tensor.shape[0], 3, 240, 320))
            else:
                rgb_frames = None
            
            # Apply transforms
            if self.use_rgb and rgb_frames is not None:
                rgb_frames = self.transforms(rgb_frames)
            
            # Sample temporal windows based on task
            if self.task_type == 'gesture':
                windows = sample_temporal_windows(rgb_frames if self.use_rgb else flow_tensor,
                                                 window_size=10, stride=5, task='gesture')
            elif self.task_type == 'skill':
                windows = sample_temporal_windows(rgb_frames if self.use_rgb else flow_tensor,
                                                 window_size=30, stride=15, task='skill')
            else:  # kinematics
                windows = sample_temporal_windows(rgb_frames if self.use_rgb else flow_tensor,
                                                 window_size=25, stride=2, task='kinematics')
            
            # Use first window for now (can be extended to return multiple)
            if len(windows) > 0:
                rgb_frames = windows[0] if self.use_rgb else None
                flow_tensor = windows[0] if not self.use_rgb and self.use_flow else None
            
            gesture_label = self._get_gesture_label(sample['gesture'])
            skill_label = 0  # Default for legacy data
            
        else:
            # New format: load from raw files
            start_frame = sample['start_frame']
            end_frame = sample['end_frame']
            
            # Load RGB frames
            if self.use_rgb:
                rgb_frames = self._load_video_frames(sample['video_path'], start_frame, end_frame)
                rgb_frames = self.transforms(rgb_frames)
            else:
                rgb_frames = None
            
            # Load flow (would be from precomputed RAFT, for now use None)
            flow_tensor = None  # TODO: Load from precomputed RAFT
            
            # Load kinematics
            kin_tensor = self._load_kinematics(sample['kinematics_path'], start_frame, end_frame)
            kin_tensor = self._extract_arm_kinematics(kin_tensor)
            kin_tensor = torch.from_numpy(kin_tensor).float()
            
            # Sample temporal windows
            if self.use_rgb and rgb_frames is not None:
                windows = sample_temporal_windows(rgb_frames, task=self.task_type)
                if len(windows) > 0:
                    rgb_frames = windows[0]
                    # Adjust kinematics to match window
                    kin_tensor = kin_tensor[:rgb_frames.shape[0]]
            
            gesture_label = self._get_gesture_label(sample['gesture'])
            skill_label = sample['skill_label']
        
        # Build output dictionary
        output = {
            'rgb': rgb_frames if self.use_rgb else None,
            'flow': flow_tensor if self.use_flow else None,
            'kinematics': kin_tensor,
            'gesture_label': torch.tensor(gesture_label, dtype=torch.long),
            'skill_label': torch.tensor(skill_label, dtype=torch.long),
            'trial_id': sample['trial_id'],
            'surgeon_id': sample['surgeon_id']
        }
        
        # Remove None values
        output = {k: v for k, v in output.items() if v is not None}
        
        return output

