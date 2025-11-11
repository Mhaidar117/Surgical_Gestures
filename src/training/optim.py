"""
Optimizer and scheduler setup for training.
"""
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, LambdaLR
from typing import Dict, Optional


def get_optimizer(
    model: torch.nn.Module,
    lr_base: float = 1e-4,
    lr_vit: float = 1e-5,
    lr_adapters: float = 5e-5,
    weight_decay: float = 0.05,
    use_adapters: bool = False
) -> optim.Optimizer:
    """
    Create optimizer with different learning rates for different components.
    
    Args:
        model: Model to optimize
        lr_base: Base learning rate for decoder/heads (can be float, int, or string)
        lr_vit: Learning rate for ViT backbone (can be float, int, or string)
        lr_adapters: Learning rate for adapters (can be float, int, or string)
        weight_decay: Weight decay
        use_adapters: Whether adapters are used
    
    Returns:
        Optimizer
    """
    # Convert learning rates to float (handles strings from YAML configs)
    lr_base = float(lr_base)
    lr_vit = float(lr_vit)
    lr_adapters = float(lr_adapters)
    weight_decay = float(weight_decay)
    
    # Separate parameters by component
    vit_params = []
    adapter_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'backbone' in name or 'visual' in name:
            if 'adapter' in name:
                adapter_params.append(param)
            else:
                vit_params.append(param)
        else:
            other_params.append(param)
    
    # Create parameter groups
    param_groups = []
    
    if len(vit_params) > 0:
        param_groups.append({
            'params': vit_params,
            'lr': lr_vit,
            'weight_decay': weight_decay
        })
    
    if len(adapter_params) > 0:
        param_groups.append({
            'params': adapter_params,
            'lr': lr_adapters,
            'weight_decay': weight_decay
        })
    
    if len(other_params) > 0:
        param_groups.append({
            'params': other_params,
            'lr': lr_base,
            'weight_decay': weight_decay
        })
    
    # Ensure we have at least one parameter group
    if len(param_groups) == 0:
        # Fallback: use all parameters with base learning rate
        param_groups = [{
            'params': [p for p in model.parameters() if p.requires_grad],
            'lr': lr_base,
            'weight_decay': weight_decay
        }]
    
    # Create optimizer
    optimizer = optim.AdamW(param_groups)
    
    # Verify all learning rates are floats (not lists/tuples)
    for i, group in enumerate(optimizer.param_groups):
        if not isinstance(group['lr'], (int, float)):
            raise ValueError(f"Parameter group {i} has invalid learning rate type: {type(group['lr'])}. Expected float, got {group['lr']}")
        # Ensure it's a float
        group['lr'] = float(group['lr'])
    
    return optimizer


def get_scheduler(
    optimizer: optim.Optimizer,
    num_epochs: int,
    warmup_epochs: int = 0,
    scheduler_type: str = 'cosine'
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: Optimizer
        num_epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        scheduler_type: 'cosine' or 'linear'
    
    Returns:
        Learning rate scheduler
    """
    if warmup_epochs > 0:
        # Warmup scheduler
        # Use LambdaLR for warmup to avoid issues with multiple parameter groups
        # LambdaLR's lr_lambda receives epoch number, returns multiplier for base LR
        def warmup_lambda(epoch):
            # Linear warmup from 0.1 to 1.0 over warmup_epochs
            return 0.1 + 0.9 * (epoch / max(warmup_epochs, 1))
        
        warmup_scheduler = LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda
        )
        
        # Main scheduler
        if scheduler_type == 'cosine':
            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs - warmup_epochs,
                eta_min=1e-6
            )
        else:  # linear
            main_scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0 - (epoch / (num_epochs - warmup_epochs))
            )
        
        # Sequential scheduler
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_epochs]
        )
    else:
        # No warmup
        if scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=num_epochs,
                eta_min=1e-6
            )
        else:  # linear
            scheduler = LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0 - (epoch / num_epochs)
            )
    
    return scheduler

