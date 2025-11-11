"""Additional modules."""
from .brain_rdm import BrainRDM, compute_model_rdm, flatten_upper_tri, spearman_correlation, sample_rdm_batch

__all__ = [
    'BrainRDM', 'compute_model_rdm', 'flatten_upper_tri',
    'spearman_correlation', 'sample_rdm_batch'
]

