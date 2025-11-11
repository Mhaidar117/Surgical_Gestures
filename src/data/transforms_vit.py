"""
Preprocessing transforms for ViT-based models.
Handles 224x224 normalization, temporal sampling, and augmentation.
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from typing import Tuple, Optional, List, Dict
from PIL import Image


class VideoToTensor:
    """Convert video frames (numpy array) to tensor."""
    
    def __call__(self, frames: np.ndarray) -> torch.Tensor:
        """
        Args:
            frames: Array of shape (T, H, W, C) or (H, W, C)
        
        Returns:
            Tensor of shape (T, C, H, W) or (C, H, W)
        """
        if len(frames.shape) == 3:
            # Single frame: (H, W, C) -> (C, H, W)
            return torch.from_numpy(frames).permute(2, 0, 1).float() / 255.0
        else:
            # Multiple frames: (T, H, W, C) -> (T, C, H, W)
            return torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0


class ResizeAndCenterCrop:
    """Resize to 256 then center crop to 224x224."""
    
    def __init__(self, target_size: int = 224, resize_size: int = 256):
        self.target_size = target_size
        self.resize_size = resize_size
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape (T, C, H, W) or (C, H, W)
        
        Returns:
            Resized and cropped tensor
        """
        is_single = len(frames.shape) == 3
        if is_single:
            frames = frames.unsqueeze(0)
        
        T, C, H, W = frames.shape
        
        # Resize to resize_size
        frames_resized = F.resize(frames.view(-1, C, H, W), (self.resize_size, self.resize_size))
        frames_resized = frames_resized.view(T, C, self.resize_size, self.resize_size)
        
        # Center crop to target_size
        top = (self.resize_size - self.target_size) // 2
        left = (self.resize_size - self.target_size) // 2
        frames_cropped = F.crop(frames_resized.view(-1, C, self.resize_size, self.resize_size),
                               top, left, self.target_size, self.target_size)
        frames_cropped = frames_cropped.view(T, C, self.target_size, self.target_size)
        
        if is_single:
            frames_cropped = frames_cropped.squeeze(0)
        
        return frames_cropped


class Normalize:
    """Normalize using ImageNet statistics."""
    
    def __init__(self, mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
                 std: Tuple[float, ...] = (0.229, 0.224, 0.225)):
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape (T, C, H, W) or (C, H, W)
        
        Returns:
            Normalized tensor
        """
        is_single = len(frames.shape) == 3
        if is_single:
            frames = frames.unsqueeze(0)
        
        frames = (frames - self.mean) / self.std
        
        if is_single:
            frames = frames.squeeze(0)
        
        return frames


class TemporalSampler:
    """Sample temporal windows from video sequences."""
    
    def __init__(self, window_size: int, stride: int = 1, mode: str = 'sliding'):
        """
        Args:
            window_size: Number of frames in window
            stride: Stride between windows
            mode: 'sliding' (overlapping) or 'non_overlapping'
        """
        self.window_size = window_size
        self.stride = stride
        self.mode = mode
    
    def __call__(self, frames: torch.Tensor, max_frames: Optional[int] = None) -> List[torch.Tensor]:
        """
        Args:
            frames: Tensor of shape (T, ...)
            max_frames: Maximum number of frames to sample (None = all)
        
        Returns:
            List of window tensors
        """
        T = frames.shape[0]
        if max_frames is not None:
            T = min(T, max_frames)
            frames = frames[:T]
        
        windows = []
        if self.mode == 'sliding':
            for start in range(0, T - self.window_size + 1, self.stride):
                windows.append(frames[start:start + self.window_size])
        else:  # non_overlapping
            for start in range(0, T, self.window_size):
                end = min(start + self.window_size, T)
                if end - start == self.window_size:
                    windows.append(frames[start:end])
        
        return windows


class TemporalAugmentation:
    """Temporal augmentation (temporal jitter, frame dropping)."""
    
    def __init__(self, drop_prob: float = 0.1, jitter_range: int = 2):
        self.drop_prob = drop_prob
        self.jitter_range = jitter_range
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Tensor of shape (T, ...)
        
        Returns:
            Augmented frames
        """
        T = frames.shape[0]
        
        # Random frame dropping
        if np.random.rand() < self.drop_prob and T > 1:
            drop_idx = np.random.randint(0, T)
            # Remove frame by shifting
            frames = torch.cat([frames[:drop_idx], frames[drop_idx+1:]], dim=0)
            T -= 1
        
        # Temporal jitter (small random shifts)
        if self.jitter_range > 0 and T > self.jitter_range:
            shift = np.random.randint(-self.jitter_range, self.jitter_range + 1)
            if shift > 0:
                frames = torch.cat([frames[shift:], frames[:shift]], dim=0)
            elif shift < 0:
                frames = torch.cat([frames[shift:], frames[:shift]], dim=0)
        
        return frames


def get_vit_transforms(
    mode: str = 'train',
    target_size: int = 224,
    normalize: bool = True
) -> transforms.Compose:
    """
    Get standard ViT preprocessing transforms.
    
    Args:
        mode: 'train' or 'val'
        target_size: Target image size (default 224)
        normalize: Whether to apply ImageNet normalization
    
    Returns:
        Compose transform
    """
    transform_list = [
        VideoToTensor(),
        ResizeAndCenterCrop(target_size=target_size)
    ]
    
    if normalize:
        transform_list.append(Normalize())
    
    if mode == 'train':
        # Add augmentation for training
        transform_list.append(TemporalAugmentation())
    
    return transforms.Compose(transform_list)


def sample_temporal_windows(
    frames: torch.Tensor,
    window_size: Optional[int] = None,
    stride: Optional[int] = None,
    task: str = 'gesture'
) -> List[torch.Tensor]:
    """
    Sample temporal windows based on task requirements.
    
    Args:
        frames: Tensor of shape (T, C, H, W)
        window_size: Window size (overridden by task if task is not 'custom')
        stride: Stride (overridden by task if task is not 'custom')
        task: 'gesture' (10 frames, stride 5), 'skill' (30 frames, stride 15), 
              'kinematics' (25 frames, stride 2), or 'custom' (uses provided window_size/stride)
    
    Returns:
        List of window tensors
    """
    if task == 'gesture':
        window_size, stride = 10, 5
    elif task == 'skill':
        window_size, stride = 30, 15
    elif task == 'kinematics':
        window_size, stride = 25, 2
    elif task == 'custom':
        # Use provided window_size and stride, or defaults
        if window_size is None:
            window_size = 10
        if stride is None:
            stride = 5
    else:
        # Default to gesture settings if unknown task
        window_size, stride = 10, 5
    
    sampler = TemporalSampler(window_size=window_size, stride=stride, mode='sliding')
    return sampler(frames)

