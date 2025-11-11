"""
Brain RDM module for EEG alignment during training.
Handles loading precomputed RDMs, tau-lag alignment, and minibatch sampling.
"""
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from scipy.stats import spearmanr


def compute_model_rdm(
    features: torch.Tensor,
    method: str = 'pearson'
) -> torch.Tensor:
    """
    Compute Representational Dissimilarity Matrix (RDM) from model features.
    
    Args:
        features: Model features of shape (M, D) where M is number of stimuli
        method: Distance method ('pearson', 'euclidean', 'cosine')
    
    Returns:
        RDM matrix of shape (M, M)
    """
    features_np = features.detach().cpu().numpy()
    M = features_np.shape[0]
    rdm = np.zeros((M, M))
    
    if method == 'pearson':
        for i in range(M):
            for j in range(M):
                if i == j:
                    rdm[i, j] = 0.0
                else:
                    corr, _ = spearmanr(features_np[i], features_np[j])
                    rdm[i, j] = 1.0 - corr if not np.isnan(corr) else 1.0
    
    elif method == 'euclidean':
        for i in range(M):
            for j in range(M):
                rdm[i, j] = np.linalg.norm(features_np[i] - features_np[j])
    
    elif method == 'cosine':
        for i in range(M):
            for j in range(M):
                if i == j:
                    rdm[i, j] = 0.0
                else:
                    dot_product = np.dot(features_np[i], features_np[j])
                    norm_i = np.linalg.norm(features_np[i])
                    norm_j = np.linalg.norm(features_np[j])
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot_product / (norm_i * norm_j)
                        rdm[i, j] = 1.0 - cosine_sim
                    else:
                        rdm[i, j] = 1.0
    
    return torch.from_numpy(rdm).float()


def flatten_upper_tri(rdm: torch.Tensor) -> torch.Tensor:
    """
    Flatten upper triangle of RDM matrix.
    
    Args:
        rdm: RDM matrix of shape (M, M)
    
    Returns:
        Flattened vector of shape (M*(M-1)//2,)
    """
    # Get upper triangle indices (excluding diagonal)
    M = rdm.shape[0]
    triu_indices = torch.triu_indices(M, M, offset=1)
    return rdm[triu_indices[0], triu_indices[1]]


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute Spearman correlation between two vectors.
    
    Args:
        x: First vector
        y: Second vector
    
    Returns:
        Spearman correlation coefficient
    """
    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    
    if len(x_np) != len(y_np):
        raise ValueError(f"Vectors must have same length: {len(x_np)} vs {len(y_np)}")
    
    corr, _ = spearmanr(x_np, y_np)
    return corr if not np.isnan(corr) else 0.0


class BrainRDM:
    """
    Brain RDM module for loading and managing EEG RDMs.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        tau_range: List[int] = None,
        rdm_method: str = 'pearson'
    ):
        """
        Args:
            cache_dir: Directory containing precomputed EEG RDMs
            tau_range: List of tau values (ms) for temporal alignment
            rdm_method: RDM computation method
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.tau_range = tau_range or [0, 50, 100, 150, 200, 250, 300]
        self.rdm_method = rdm_method
        
        # Cache for loaded RDMs
        self.rdm_cache: Dict[str, Dict] = {}
    
    def load_eeg_rdm(
        self,
        trial_id: str,
        tau: int = 0,
        window_size_ms: float = 100.0
    ) -> Optional[torch.Tensor]:
        """
        Load precomputed EEG RDM for a trial.
        
        Args:
            trial_id: Trial identifier
            tau: Temporal lag in milliseconds
            window_size_ms: Window size used for RDM computation
        
        Returns:
            RDM matrix of shape (M, M) or None if not found
        """
        if self.cache_dir is None:
            return None
        
        # Check cache first
        cache_key = f"{trial_id}_tau{tau}_w{window_size_ms}"
        if cache_key in self.rdm_cache:
            return self.rdm_cache[cache_key].get('rdm')
        
        # Try to load from file
        rdm_file = self.cache_dir / f"{trial_id}_rdm.pkl"
        if not rdm_file.exists():
            return None
        
        try:
            with open(rdm_file, 'rb') as f:
                rdm_data = pickle.load(f)
            
            # Store in cache
            self.rdm_cache[cache_key] = rdm_data
            
            # Return RDM (tau alignment would be handled during RDM computation)
            return torch.from_numpy(rdm_data['rdm']).float()
        except Exception as e:
            print(f"Error loading RDM for {trial_id}: {e}")
            return None
    
    def get_eeg_rdm(
        self,
        batch_meta: Dict,
        indices: Optional[List[int]] = None,
        tau: int = 0,
        mode: str = 'precomputed'
    ) -> Optional[torch.Tensor]:
        """
        Get EEG RDM for a batch.
        
        Args:
            batch_meta: Dictionary with batch metadata (trial_ids, frame_indices, etc.)
            indices: Optional list of indices to sample from batch
            tau: Temporal lag in milliseconds
            mode: 'precomputed' or 'online'
        
        Returns:
            RDM matrix of shape (M, M) where M is number of stimuli
        """
        if mode != 'precomputed':
            # Online computation not implemented yet
            return None
        
        trial_ids = batch_meta.get('trial_ids', [])
        if not trial_ids:
            return None
        
        # Sample indices if provided
        if indices is not None:
            trial_ids = [trial_ids[i] for i in indices]
        
        # Load RDMs for each trial
        rdms = []
        for trial_id in trial_ids:
            rdm = self.load_eeg_rdm(trial_id, tau=tau)
            if rdm is not None:
                rdms.append(rdm)
        
        if len(rdms) == 0:
            return None
        
        # For now, return first RDM
        # In practice, you'd combine RDMs from multiple trials or stimuli
        return rdms[0]
    
    def find_best_tau(
        self,
        model_rdm: torch.Tensor,
        trial_ids: List[str],
        tau_range: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        """
        Find best tau value that maximizes correlation between model and EEG RDMs.
        
        Args:
            model_rdm: Model RDM of shape (M, M)
            trial_ids: List of trial IDs
            tau_range: Range of tau values to search, None uses self.tau_range
        
        Returns:
            Tuple of (best_tau, best_correlation)
        """
        tau_range = tau_range or self.tau_range
        
        best_tau = 0
        best_corr = -1.0
        
        model_rdm_flat = flatten_upper_tri(model_rdm)
        
        for tau in tau_range:
            # Load EEG RDM for this tau
            eeg_rdm = self.get_eeg_rdm(
                {'trial_ids': trial_ids},
                tau=tau
            )
            
            if eeg_rdm is None:
                continue
            
            # Compute correlation
            eeg_rdm_flat = flatten_upper_tri(eeg_rdm)
            
            if len(model_rdm_flat) == len(eeg_rdm_flat):
                corr = spearman_correlation(model_rdm_flat, eeg_rdm_flat)
                if corr > best_corr:
                    best_corr = corr
                    best_tau = tau
        
        return best_tau, best_corr


def sample_rdm_batch(
    features: torch.Tensor,
    batch_size: int = 32,
    random: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a subset of features for RDM computation.
    
    Args:
        features: Features of shape (N, D) where N is total number of stimuli
        batch_size: Number of stimuli to sample (M)
        random: Whether to sample randomly or sequentially
    
    Returns:
        Tuple of (sampled_features: (M, D), indices: (M,))
    """
    N = features.shape[0]
    M = min(batch_size, N)
    
    if random:
        indices = torch.randperm(N)[:M]
    else:
        indices = torch.arange(M)
    
    sampled = features[indices]
    
    return sampled, indices

