"""
Temporal transformer for aggregating frame-level embeddings over time.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class TemporalAggregator(nn.Module):
    """
    Temporal transformer encoder for aggregating frame embeddings.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_len: int = 64,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            activation: Activation function
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        
        # Learned positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            seq: Input sequence of shape (B, T, D)
            mask: Optional attention mask of shape (B, T) where True indicates padding
        
        Returns:
            Tuple of (memory: contextualized sequence (B, T, D),
                     pooled: pooled representation (B, D))
        """
        B, T, D = seq.shape
        
        # Add positional encoding
        if T <= self.max_len:
            positions = self.pos_embed[:, :T]
        else:
            # If sequence is longer than max_len, interpolate positional encoding
            positions = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        seq = seq + positions
        
        # Create key padding mask (True = ignore)
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = mask  # (B, T), True = padding
        
        # Forward through encoder
        memory = self.encoder(seq, src_key_padding_mask=key_padding_mask)
        memory = self.norm(memory)
        
        # Pooled representation: mean pooling (ignoring padding)
        if mask is not None:
            # Set padding positions to 0 before mean
            memory_masked = memory.clone()
            memory_masked[mask] = 0.0
            lengths = (~mask).sum(dim=1, keepdim=True).float()  # (B, 1)
            pooled = memory_masked.sum(dim=1) / lengths.clamp(min=1.0)
        else:
            pooled = memory.mean(dim=1)  # (B, D)
        
        return memory, pooled


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over temporal dimension.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.attention = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            seq: Input sequence of shape (B, T, D)
            mask: Optional attention mask
        
        Returns:
            Pooled representation of shape (B, D)
        """
        B, T, D = seq.shape
        
        # Expand query to batch size
        query = self.query.expand(B, -1, -1)
        
        # Apply attention
        attn_output, _ = self.attention(
            query, seq, seq,
            key_padding_mask=mask
        )
        
        # Layer norm and squeeze
        pooled = self.norm(attn_output.squeeze(1))  # (B, D)
        
        return pooled


class TemporalAggregatorWithPooling(nn.Module):
    """
    Temporal aggregator with multiple pooling options.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        max_len: int = 64,
        use_attention_pooling: bool = True
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_attention_pooling: Whether to use attention pooling in addition to mean
        """
        super().__init__()
        
        self.aggregator = TemporalAggregator(
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_len=max_len
        )
        
        self.use_attention_pooling = use_attention_pooling
        if use_attention_pooling:
            self.attention_pool = AttentionPooling(d_model, dropout)
    
    def forward(
        self,
        seq: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            seq: Input sequence of shape (B, T, D)
            mask: Optional attention mask
        
        Returns:
            Tuple of (memory: contextualized sequence (B, T, D),
                     mean_pooled: mean pooled (B, D),
                     attn_pooled: attention pooled (B, D) or None)
        """
        memory, mean_pooled = self.aggregator(seq, mask)
        
        attn_pooled = None
        if self.use_attention_pooling:
            attn_pooled = self.attention_pool(memory, mask)
        
        return memory, mean_pooled, attn_pooled

