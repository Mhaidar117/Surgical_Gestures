"""
Autoregressive transformer decoder for kinematics prediction.
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class CrossAttnDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention and cross-attention.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            tgt: Target sequence (B, T_tgt, D)
            memory: Encoder memory (B, T_src, D)
            tgt_mask: Causal mask for self-attention (T_tgt, T_tgt)
            memory_mask: Mask for cross-attention (T_tgt, T_src)
            tgt_key_padding_mask: Padding mask for tgt (B, T_tgt)
            memory_key_padding_mask: Padding mask for memory (B, T_src)
        
        Returns:
            Output tensor (B, T_tgt, D)
        """
        # Self-attention
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        tgt = self.norm1(tgt + self.dropout(attn_output))
        
        # Cross-attention
        attn_output, _ = self.cross_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )
        tgt = self.norm2(tgt + self.dropout(attn_output))
        
        # Feedforward
        ff_output = self.ff(tgt)
        tgt = self.norm3(tgt + ff_output)
        
        return tgt


def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Generate causal mask for autoregressive decoding.
    
    Args:
        seq_len: Sequence length
        device: Device
    
    Returns:
        Causal mask of shape (seq_len, seq_len) where True = mask out
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


class KinematicsDecoder(nn.Module):
    """
    Autoregressive transformer decoder for kinematics prediction.
    """
    
    def __init__(
        self,
        num_layers: int = 6,
        d_model: int = 384,
        n_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
        d_out: int = 10,
        d_kin_input: int = 76,
        use_learned_start_token: bool = True
    ):
        """
        Args:
            num_layers: Number of decoder layers
            d_model: Model dimension
            n_heads: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            d_out: Output dimension (e.g., 10 for pos3 + rot6D + jaw1)
            d_kin_input: Input kinematics dimension
            use_learned_start_token: Whether to use learned start token
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_out = d_out
        self.d_kin_input = d_kin_input
        self.use_learned_start_token = use_learned_start_token
        
        # Input projection for kinematics
        self.kin_embed = nn.Linear(d_kin_input, d_model)
        
        # Learned start token
        if use_learned_start_token:
            self.start_token = nn.Parameter(torch.randn(1, 1, d_model))
            nn.init.normal_(self.start_token, std=0.02)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 1000, d_model))  # Max length 1000
        nn.init.normal_(self.pos_embed, std=0.02)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            CrossAttnDecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_out)
        )
    
    def embed_kinematics(self, kinematics: torch.Tensor) -> torch.Tensor:
        """
        Embed kinematics input.
        
        Args:
            kinematics: Kinematics tensor of shape (B, T, d_kin_input)
        
        Returns:
            Embedded tensor of shape (B, T, d_model)
        """
        return self.kin_embed(kinematics)
    
    def forward(
        self,
        targets: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            targets: Target kinematics of shape (B, T, d_kin_input) or embedded (B, T, d_model)
            memory: Encoder memory of shape (B, T_src, d_model)
            tgt_mask: Causal mask (T, T)
            memory_key_padding_mask: Padding mask for memory (B, T_src)
        
        Returns:
            Predicted kinematics of shape (B, T, d_out)
        """
        # Embed targets if needed
        if targets.shape[-1] == self.d_kin_input:
            tgt = self.embed_kinematics(targets)
        else:
            tgt = targets
        
        B, T, D = tgt.shape
        
        # Add positional encoding
        if T <= self.pos_embed.shape[1]:
            positions = self.pos_embed[:, :T]
        else:
            # Interpolate if longer
            positions = nn.functional.interpolate(
                self.pos_embed.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        
        tgt = tgt + positions
        
        # Generate causal mask if not provided
        if tgt_mask is None and self.training:
            tgt_mask = generate_causal_mask(T, tgt.device)
        
        # Forward through decoder layers
        for layer in self.layers:
            tgt = layer(
                tgt, memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        # Output projection
        output = self.head(tgt)
        
        return output
    
    def autoregressive_step(
        self,
        prev_output: torch.Tensor,
        memory: torch.Tensor,
        step: int,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Single autoregressive step.
        
        Args:
            prev_output: Previous outputs of shape (B, T, d_out) or embedded (B, T, d_model)
            memory: Encoder memory (B, T_src, D)
            step: Current step index
            memory_key_padding_mask: Padding mask for memory
        
        Returns:
            Next step prediction of shape (B, d_out)
        """
        # Generate causal mask for current step
        T = prev_output.shape[1]
        tgt_mask = generate_causal_mask(T, prev_output.device)
        
        # Forward pass
        output = self.forward(
            prev_output, memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Return last timestep
        return output[:, -1, :]

