"""
Inference pipeline for kinematics prediction.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_vit_system import EEGInformedViTModel
from eval.postprocess import postprocess_kinematics
from safety.filters import validate_trajectory


def predict_kinematics(
    model: nn.Module,
    rgb_frames: torch.Tensor,
    autoregressive: bool = False,
    horizon: int = 50,
    postprocess: bool = True
) -> torch.Tensor:
    """
    Predict kinematics from RGB frames.
    
    Args:
        model: Trained model
        rgb_frames: RGB frames of shape (B, T, C, H, W) or (T, C, H, W)
        autoregressive: Whether to use autoregressive decoding
        horizon: Prediction horizon for autoregressive mode
        postprocess: Whether to apply post-processing
    
    Returns:
        Predicted kinematics of shape (B, T, D) or (T, D)
         
         #output needs to be xyz in cartesian space and angles in degrees. Mai wants it to look just like the input from JIGSAW
         
    """
    model.eval()
    device = next(model.parameters()).device
    
    is_batch = len(rgb_frames.shape) == 5
    if not is_batch:
        rgb_frames = rgb_frames.unsqueeze(0)
    
    rgb_frames = rgb_frames.to(device)
    
    with torch.no_grad():
        if autoregressive:
            # Autoregressive prediction
            outputs = model(rgb_frames, return_embeddings=True)
            memory = outputs['memory']
            
            # Start with learned start token or first frame embedding
            # Simplified - full implementation would iterate
            kinematics = outputs['kinematics']
        else:
            # Parallel prediction
            outputs = model(rgb_frames)
            kinematics = outputs['kinematics']
    
    if not is_batch:
        kinematics = kinematics.squeeze(0)
    
    # Post-process
    if postprocess:
        kinematics = postprocess_kinematics(kinematics)
    
    return kinematics


def predict_with_safety_check(
    model: nn.Module,
    rgb_frames: torch.Tensor,
    workspace_bounds: Optional[Dict] = None,
    max_velocity: float = 0.1,
    max_acceleration: float = 0.5
) -> Tuple[torch.Tensor, bool, str]:
    """
    Predict kinematics with safety validation.
    
    Args:
        model: Trained model
        rgb_frames: RGB frames
        workspace_bounds: Workspace bounds
        max_velocity: Maximum velocity
        max_acceleration: Maximum acceleration
    
    Returns:
        Tuple of (kinematics, is_safe, message)
    """
    # Predict
    kinematics = predict_kinematics(model, rgb_frames, postprocess=True)
    
    # Validate
    is_safe, message = validate_trajectory(
        kinematics,
        workspace_bounds,
        max_velocity,
        max_acceleration
    )
    
    return kinematics, is_safe, message

