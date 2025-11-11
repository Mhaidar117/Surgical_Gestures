"""
Kinematics decoder module with gesture and skill classification heads.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .decoder_autoreg import KinematicsDecoder, generate_causal_mask
from .temporal_transformer import TemporalAggregatorWithPooling


class GestureHead(nn.Module):
    """
    Gesture classification head with temporal attention pooling.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_gestures: int = 15,
        dropout: float = 0.1,
        label_smoothing: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            num_gestures: Number of gesture classes
            dropout: Dropout probability
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.label_smoothing = label_smoothing
        
        # Temporal attention pooling (over first 10 steps for gesture)
        self.attention_pool = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout, batch_first=True
        )
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        nn.init.normal_(self.query, std=0.02)
        
        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_gestures)
        )
    
    def forward(
        self,
        memory: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            memory: Encoder memory of shape (B, T, D)
            mask: Optional attention mask
        
        Returns:
            Gesture logits of shape (B, T, num_gestures) for per-frame or (B, num_gestures) for pooled
        """
        B, T, D = memory.shape
        
        # Use first 10 steps for gesture classification
        T_gesture = min(10, T)
        memory_gesture = memory[:, :T_gesture, :]
        
        # Attention pooling
        query = self.query.expand(B, -1, -1)
        pooled, _ = self.attention_pool(query, memory_gesture, memory_gesture)
        pooled = pooled.squeeze(1)  # (B, D)
        
        # Classification
        logits = self.head(pooled)  # (B, num_gestures)
        
        return logits


class SkillHead(nn.Module):
    """
    Skill classification head (trial-level).
    """
    
    def __init__(
        self,
        d_model: int = 384,
        num_skills: int = 3,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            num_skills: Number of skill classes (Novice, Intermediate, Expert)
            hidden_dim: Hidden dimension for MLP
            dropout: Dropout probability
        """
        super().__init__()
        
        # Two-layer MLP
        self.head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_skills)
        )
    
    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pooled: Pooled representation of shape (B, D)
        
        Returns:
            Skill logits of shape (B, num_skills)
        """
        return self.head(pooled)


class KinematicsModule(nn.Module):
    """
    Complete kinematics module with decoder, gesture, and skill heads.
    """
    
    def __init__(
        self,
        d_model: int = 384,
        d_kin_input: int = 76,
        d_kin_output: int = 10,
        num_gestures: int = 15,
        num_skills: int = 3,
        decoder_layers: int = 6,
        decoder_heads: int = 6,
        dim_feedforward: int = 1536,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Model dimension
            d_kin_input: Input kinematics dimension
            d_kin_output: Output kinematics dimension (e.g., 10 for pos3 + rot6D + jaw1)
            num_gestures: Number of gesture classes
            num_skills: Number of skill classes
            decoder_layers: Number of decoder layers
            decoder_heads: Number of attention heads in decoder
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        # Deterministic projection: MLP from embeddings to latent
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Autoregressive decoder
        self.decoder = KinematicsDecoder(
            num_layers=decoder_layers,
            d_model=d_model,
            n_heads=decoder_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            d_out=d_kin_output,
            d_kin_input=d_kin_input
        )
        
        # Gesture head
        self.gesture_head = GestureHead(
            d_model=d_model,
            num_gestures=num_gestures,
            dropout=dropout
        )
        
        # Skill head
        self.skill_head = SkillHead(
            d_model=d_model,
            num_skills=num_skills,
            dropout=dropout
        )
    
    def decode_kinematics(
        self,
        pooled_emb: torch.Tensor,
        memory: torch.Tensor,
        target_kinematics: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 1.0,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decode kinematics from embeddings.
        
        Args:
            pooled_emb: Pooled embeddings of shape (B, T, D) or (B, D)
            memory: Encoder memory of shape (B, T_src, D)
            target_kinematics: Ground truth kinematics for teacher forcing (B, T, d_kin_input)
            teacher_forcing_prob: Probability of using ground truth
            memory_key_padding_mask: Padding mask for memory
        
        Returns:
            Tuple of (k_hat: predicted kinematics (B, T, d_out),
                     gesture_logits: gesture logits (B, num_gestures),
                     skill_logits: skill logits (B, num_skills))
        """
        # Project embeddings
        if len(pooled_emb.shape) == 2:
            # (B, D) -> expand to (B, T, D) where T is from memory
            B, D = pooled_emb.shape
            T = memory.shape[1]
            pooled_emb = pooled_emb.unsqueeze(1).expand(B, T, D)
        
        projected = self.projection(pooled_emb)  # (B, T, D)
        
        # Prepare decoder inputs
        if target_kinematics is not None and self.training:
            # Teacher forcing
            use_gt = torch.rand(1, device=pooled_emb.device).item() < teacher_forcing_prob
            if use_gt:
                decoder_input = target_kinematics
            else:
                # Use previous predictions (autoregressive)
                # Start with learned start token or first projected embedding
                decoder_input = self.decoder.embed_kinematics(target_kinematics[:, :1, :])
                # This is simplified - full autoregressive would require iterative decoding
                decoder_input = projected
        else:
            # Inference: use projected embeddings as initial state
            if target_kinematics is not None:
                decoder_input = target_kinematics
            else:
                # Use projected embeddings as decoder input
                # In practice, you'd start with a learned start token
                decoder_input = projected
        
        # Decode kinematics
        k_hat = self.decoder(
            decoder_input,
            memory,
            memory_key_padding_mask=memory_key_padding_mask
        )
        
        # Gesture classification
        gesture_logits = self.gesture_head(memory)
        
        # Skill classification (use pooled representation)
        if len(pooled_emb.shape) == 3:
            pooled = pooled_emb.mean(dim=1)  # (B, D)
        else:
            pooled = pooled_emb
        skill_logits = self.skill_head(pooled)
        
        return k_hat, gesture_logits, skill_logits
    
    def forward(
        self,
        pooled_emb: torch.Tensor,
        memory: torch.Tensor,
        target_kinematics: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 1.0,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            Dictionary with 'kinematics', 'gesture_logits', 'skill_logits'
        """
        k_hat, gesture_logits, skill_logits = self.decode_kinematics(
            pooled_emb, memory, target_kinematics,
            teacher_forcing_prob, memory_key_padding_mask
        )
        
        return {
            'kinematics': k_hat,
            'gesture_logits': gesture_logits,
            'skill_logits': skill_logits
        }

