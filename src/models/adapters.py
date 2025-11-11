"""
Lightweight adapter layers for per-subject personalization in ViT blocks.
"""
import torch
import torch.nn as nn
from typing import Optional


class AdapterLayer(nn.Module):
    """
    Lightweight adapter layer inserted into ViT blocks.
    Implements bottleneck architecture: down-project -> activation -> up-project.
    """
    
    def __init__(
        self,
        d_model: int,
        adapter_dim: int = 64,
        activation: str = 'gelu',
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension (e.g., 384 for ViT-S)
            adapter_dim: Adapter bottleneck dimension
            activation: Activation function ('gelu' or 'relu')
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.adapter_dim = adapter_dim
        
        # Down-projection
        self.down_proj = nn.Linear(d_model, adapter_dim)
        
        # Activation
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Up-projection
        self.up_proj = nn.Linear(adapter_dim, d_model)
        
        # Initialize with small values to minimize impact on pretrained weights
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.normal_(self.up_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.zeros_(self.up_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (..., d_model)
        
        Returns:
            Adapter output of shape (..., d_model)
        """
        # Residual connection is handled externally
        out = self.down_proj(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.up_proj(out)
        return out


class ViTBlockWithAdapter(nn.Module):
    """
    ViT block with adapter layer inserted after attention and MLP.
    """
    
    def __init__(
        self,
        original_block: nn.Module,
        adapter_dim: int = 64,
        adapter_after_attn: bool = True,
        adapter_after_mlp: bool = True
    ):
        """
        Args:
            original_block: Original ViT block from timm
            adapter_dim: Adapter bottleneck dimension
            adapter_after_attn: Whether to insert adapter after attention
            adapter_after_mlp: Whether to insert adapter after MLP
        """
        super().__init__()
        self.original_block = original_block
        
        # Get model dimension from block
        # For timm ViT blocks, norm1 is typically LayerNorm with d_model features
        d_model = original_block.norm1.normalized_shape[0]
        
        self.adapter_after_attn = adapter_after_attn
        self.adapter_after_mlp = adapter_after_mlp
        
        if adapter_after_attn:
            self.adapter_attn = AdapterLayer(d_model, adapter_dim)
        else:
            self.adapter_attn = None
        
        if adapter_after_mlp:
            self.adapter_mlp = AdapterLayer(d_model, adapter_dim)
        else:
            self.adapter_mlp = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adapters.
        
        Args:
            x: Input tensor of shape (B, N, d_model) where N is number of patches + 1 (CLS)
        
        Returns:
            Output tensor of same shape
        """
        # Attention block
        x_norm = self.original_block.norm1(x)
        attn_out = self.original_block.attn(x_norm)
        x = x + attn_out
        
        # Adapter after attention
        if self.adapter_after_attn and self.adapter_attn is not None:
            x = x + self.adapter_attn(x)
        
        # MLP block
        x_norm = self.original_block.norm2(x)
        mlp_out = self.original_block.mlp(x_norm)
        x = x + mlp_out
        
        # Adapter after MLP
        if self.adapter_after_mlp and self.adapter_mlp is not None:
            x = x + self.adapter_mlp(x)
        
        return x


def insert_adapters_into_vit(
    vit_model: nn.Module,
    adapter_dim: int = 64,
    adapter_layers: Optional[list] = None,
    adapter_after_attn: bool = True,
    adapter_after_mlp: bool = True
) -> nn.Module:
    """
    Insert adapter layers into ViT model.
    
    Args:
        vit_model: ViT model from timm
        adapter_dim: Adapter bottleneck dimension
        adapter_layers: List of layer indices to add adapters to, None = all layers
        adapter_after_attn: Whether to insert adapter after attention
        adapter_after_mlp: Whether to insert adapter after MLP
    
    Returns:
        Modified ViT model with adapters
    """
    # Get transformer blocks
    if hasattr(vit_model, 'blocks'):
        blocks = vit_model.blocks
    elif hasattr(vit_model, 'transformer') and hasattr(vit_model.transformer, 'blocks'):
        blocks = vit_model.transformer.blocks
    else:
        raise ValueError("Could not find transformer blocks in ViT model")
    
    # Determine which layers to modify
    if adapter_layers is None:
        adapter_layers = list(range(len(blocks)))
    
    # Replace blocks with adapter versions
    for i in adapter_layers:
        if i < len(blocks):
            original_block = blocks[i]
            blocks[i] = ViTBlockWithAdapter(
                original_block,
                adapter_dim=adapter_dim,
                adapter_after_attn=adapter_after_attn,
                adapter_after_mlp=adapter_after_mlp
            )
    
    return vit_model


def freeze_vit_except_adapters(vit_model: nn.Module, adapter_layers: Optional[list] = None):
    """
    Freeze all ViT parameters except adapters.
    
    Args:
        vit_model: ViT model with adapters
        adapter_layers: List of layer indices with adapters, None = all layers
    """
    # Freeze all parameters first
    for param in vit_model.parameters():
        param.requires_grad = False
    
    # Get transformer blocks
    if hasattr(vit_model, 'blocks'):
        blocks = vit_model.blocks
    elif hasattr(vit_model, 'transformer') and hasattr(vit_model.transformer, 'blocks'):
        blocks = vit_model.transformer.blocks
    else:
        raise ValueError("Could not find transformer blocks in ViT model")
    
    # Unfreeze adapters
    if adapter_layers is None:
        adapter_layers = list(range(len(blocks)))
    
    for i in adapter_layers:
        if i < len(blocks):
            block = blocks[i]
            if hasattr(block, 'adapter_attn') and block.adapter_attn is not None:
                for param in block.adapter_attn.parameters():
                    param.requires_grad = True
            if hasattr(block, 'adapter_mlp') and block.adapter_mlp is not None:
                for param in block.adapter_mlp.parameters():
                    param.requires_grad = True

