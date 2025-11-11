"""
ViT encoder module with layer extraction and adapter support.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import hashlib
import pickle

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    raise ImportError("timm is required for ViT models. Install with: pip install timm")

from .adapters import insert_adapters_into_vit, freeze_vit_except_adapters


class ViTFrameEncoder(nn.Module):
    """
    ViT encoder for processing video frames.
    Supports layer extraction for RSA, adapters for personalization, and feature caching.
    """
    
    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        pretrained: bool = True,
        freeze_until: int = 6,
        cache_dir: Optional[str] = None,
        use_adapters: bool = False,
        adapter_dim: int = 64,
        adapter_layers: Optional[List[int]] = None,
        return_layers: Optional[List[str]] = None
    ):
        """
        Args:
            model_name: timm model name
            pretrained: Whether to use pretrained weights
            freeze_until: Freeze blocks until this index (0-based)
            cache_dir: Directory for caching features
            use_adapters: Whether to use adapter layers
            adapter_dim: Adapter bottleneck dimension
            adapter_layers: List of layer indices to add adapters to, None = all layers
            return_layers: List of layer names to return activations for ('early', 'mid', 'late')
        """
        super().__init__()
        
        if not TIMM_AVAILABLE:
            raise ImportError("timm is required")
        
        self.model_name = model_name
        self.freeze_until = freeze_until
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_adapters = use_adapters
        self.return_layers = return_layers or ['early', 'mid', 'late']
        
        # Create ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''  # Return all tokens, not just CLS
        )
        
        # Get number of blocks
        if hasattr(self.backbone, 'blocks'):
            self.num_blocks = len(self.backbone.blocks)
        elif hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'blocks'):
            self.num_blocks = len(self.backbone.transformer.blocks)
        else:
            raise ValueError("Could not determine number of blocks")
        
        # Insert adapters if requested
        if use_adapters:
            self.backbone = insert_adapters_into_vit(
                self.backbone,
                adapter_dim=adapter_dim,
                adapter_layers=adapter_layers
            )
        
        # Freeze early layers
        self._freeze_stages()
        
        # Determine which layers to extract
        self.layer_indices = self._get_layer_indices()
        
        # Get embedding dimension
        if hasattr(self.backbone, 'embed_dim'):
            self.embed_dim = self.backbone.embed_dim
        else:
            # Try to infer from first block
            if hasattr(self.backbone, 'blocks'):
                first_block = self.backbone.blocks[0]
            else:
                first_block = self.backbone.transformer.blocks[0]
            self.embed_dim = first_block.norm1.normalized_shape[0]
    
    def _get_layer_indices(self) -> Dict[str, int]:
        """Map layer names to block indices."""
        indices = {}
        if 'early' in self.return_layers:
            indices['early'] = self.num_blocks // 4
        if 'mid' in self.return_layers:
            indices['mid'] = self.num_blocks // 2
        if 'late' in self.return_layers:
            indices['late'] = self.num_blocks - 1
        return indices
    
    def _freeze_stages(self):
        """Freeze early layers."""
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
        elif hasattr(self.backbone, 'transformer') and hasattr(self.backbone.transformer, 'blocks'):
            blocks = self.backbone.transformer.blocks
        else:
            return
        
        for i, block in enumerate(blocks):
            for param in block.parameters():
                param.requires_grad = i >= self.freeze_until
        
        # Freeze patch embedding and positional encoding
        if hasattr(self.backbone, 'patch_embed'):
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
        if hasattr(self.backbone, 'pos_embed'):
            self.backbone.pos_embed.requires_grad = False
    
    def _maybe_load_cache(
        self,
        video_ids: Optional[List[str]],
        frame_indices: Optional[List[int]],
        device: torch.device
    ) -> Optional[torch.Tensor]:
        """Load cached features if available."""
        if self.cache_dir is None or video_ids is None or frame_indices is None:
            return None
        
        try:
            # Create cache key
            cache_key = f"{video_ids[0]}_{frame_indices[0]}_{frame_indices[-1]}"
            cache_file = self.cache_dir / f"{cache_key}.pt"
            
            if cache_file.exists():
                return torch.load(cache_file, map_location=device)
        except Exception:
            pass
        
        return None
    
    def _maybe_write_cache(
        self,
        video_ids: Optional[List[str]],
        frame_indices: Optional[List[int]],
        features: torch.Tensor
    ):
        """Write features to cache."""
        if self.cache_dir is None or video_ids is None or frame_indices is None:
            return
        
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_key = f"{video_ids[0]}_{frame_indices[0]}_{frame_indices[-1]}"
            cache_file = self.cache_dir / f"{cache_key}.pt"
            torch.save(features, cache_file)
        except Exception:
            pass
    
    def forward(
        self,
        frames: torch.Tensor,
        video_ids: Optional[List[str]] = None,
        frame_indices: Optional[List[int]] = None,
        return_layers: Optional[List[str]] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward pass through ViT encoder.
        
        Args:
            frames: Input frames of shape (B, T, C, H, W) or (B*T, C, H, W)
            video_ids: Optional list of video IDs for caching
            frame_indices: Optional list of frame indices for caching
            return_layers: Override default return_layers
        
        Returns:
            Tuple of (emb_layers: Dict mapping layer names to activations,
                     pooled_emb: CLS token embeddings of shape (B, T, D))
        """
        return_layers = return_layers or self.return_layers
        
        # Handle input shape
        original_shape = frames.shape
        if len(original_shape) == 5:
            # (B, T, C, H, W) -> (B*T, C, H, W)
            B, T = original_shape[:2]
            frames = frames.view(-1, *original_shape[2:])
            reshape_back = True
        else:
            # Assume (B*T, C, H, W)
            B = frames.shape[0]
            T = 1
            reshape_back = False
        
        # Check cache
        cached = self._maybe_load_cache(video_ids, frame_indices, frames.device)
        
        if cached is not None:
            features = cached
        else:
            # Forward through backbone
            # timm models return features before global pooling
            features = self.backbone.forward_features(frames)
            
            # Cache features
            self._maybe_write_cache(video_ids, frame_indices, features)
        
        # Extract CLS token (first token)
        # timm ViT outputs (B*T, N_patches+1, D) where first token is CLS
        cls_tokens = features[:, 0, :]  # (B*T, D)
        
        # Reshape back to (B, T, D)
        if reshape_back:
            cls_tokens = cls_tokens.view(B, T, -1)
        
        # Extract layer activations for RSA
        emb_layers = {}
        
        # For layer extraction, we need to hook into intermediate blocks
        # This is a simplified version - in practice, you'd use forward hooks
        # For now, we'll return the final features as a placeholder
        # Full implementation would require modifying forward pass to capture intermediate activations
        
        # Get block outputs for specified layers
        if hasattr(self.backbone, 'blocks'):
            blocks = self.backbone.blocks
        else:
            blocks = self.backbone.transformer.blocks
        
        # Store activations during forward pass
        # This requires hook registration - simplified here
        layer_activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                layer_activations[name] = output
            return hook
        
        # Register hooks for specified layers
        hooks = []
        for layer_name, layer_idx in self.layer_indices.items():
            if layer_name in return_layers and layer_idx < len(blocks):
                hook = blocks[layer_idx].register_forward_hook(hook_fn(layer_name))
                hooks.append(hook)
        
        # Re-run forward to capture activations
        if len(hooks) > 0:
            _ = self.backbone.forward_features(frames)
            for layer_name in return_layers:
                if layer_name in layer_activations:
                    act = layer_activations[layer_name]
                    # Extract CLS token
                    if len(act.shape) == 3:
                        cls_act = act[:, 0, :]  # (B*T, D)
                        if reshape_back:
                            cls_act = cls_act.view(B, T, -1)
                        emb_layers[layer_name] = cls_act
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # If no layers were captured, use final features as fallback
        if len(emb_layers) == 0:
            for layer_name in return_layers:
                emb_layers[layer_name] = cls_tokens
        
        return emb_layers, cls_tokens


class ViTFlowEncoder(nn.Module):
    """
    ViT encoder for optical flow (2-channel input).
    Initialized from RGB ViT via channel averaging.
    """
    
    def __init__(
        self,
        rgb_encoder: ViTFrameEncoder,
        adapter_dim: int = 64
    ):
        """
        Args:
            rgb_encoder: Pre-trained RGB ViT encoder
            adapter_dim: Adapter dimension (not used, kept for compatibility)
        """
        super().__init__()
        
        # Create new model with 2-channel input
        self.backbone = timm.create_model(
            rgb_encoder.model_name.replace('_patch16_224', '_patch16_224'),
            pretrained=False,
            num_classes=0,
            global_pool='',
            in_chans=2  # 2 channels for flow
        )
        
        # Initialize patch embedding from RGB encoder
        if hasattr(rgb_encoder.backbone, 'patch_embed'):
            rgb_patch_embed = rgb_encoder.backbone.patch_embed
            flow_patch_embed = self.backbone.patch_embed
            
            # Average RGB channels to get 2-channel flow embedding
            with torch.no_grad():
                rgb_weight = rgb_patch_embed.proj.weight.data  # (D, 3, kernel, kernel)
                # Average across RGB channels and split into 2 flow channels
                avg_weight = rgb_weight.mean(dim=1, keepdim=True)  # (D, 1, kernel, kernel)
                flow_patch_embed.proj.weight.data = avg_weight.repeat(1, 2, 1, 1) / 2.0
        
        # Copy other weights from RGB encoder
        self._copy_encoder_weights(rgb_encoder.backbone, self.backbone)
        
        self.embed_dim = rgb_encoder.embed_dim
    
    def _copy_encoder_weights(self, src, dst):
        """Copy weights from RGB encoder to flow encoder."""
        # Copy blocks
        if hasattr(src, 'blocks') and hasattr(dst, 'blocks'):
            for src_block, dst_block in zip(src.blocks, dst.blocks):
                for src_param, dst_param in zip(src_block.parameters(), dst_block.parameters()):
                    if src_param.shape == dst_param.shape:
                        dst_param.data.copy_(src_param.data)
        
        # Copy positional embedding
        if hasattr(src, 'pos_embed') and hasattr(dst, 'pos_embed'):
            if src.pos_embed.shape == dst.pos_embed.shape:
                dst.pos_embed.data.copy_(src.pos_embed.data)
    
    def forward(
        self,
        flow: torch.Tensor,
        video_ids: Optional[List[str]] = None,
        frame_indices: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Forward pass through flow encoder.
        
        Args:
            flow: Flow tensor of shape (B, T, 2, H, W) or (B*T, 2, H, W)
            video_ids: Optional (for compatibility)
            frame_indices: Optional (for compatibility)
        
        Returns:
            CLS token embeddings of shape (B, T, D)
        """
        # Handle input shape
        if len(flow.shape) == 5:
            B, T = flow.shape[:2]
            flow = flow.view(-1, *flow.shape[2:])
            reshape_back = True
        else:
            B = flow.shape[0]
            T = 1
            reshape_back = False
        
        # Forward through backbone
        features = self.backbone.forward_features(flow)
        cls_tokens = features[:, 0, :]  # (B*T, D)
        
        if reshape_back:
            cls_tokens = cls_tokens.view(B, T, -1)
        
        return cls_tokens

