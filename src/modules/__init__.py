"""Additional modules."""
from .brain_rdm import (
    BrainRDM,
    compute_model_rdm,
    flatten_upper_tri,
    spearman_correlation,
    sample_rdm_batch,
    load_eye_rdm,
    compute_task_centroid_rdm,
    eye_rsa_loss,
)

__all__ = [
    'BrainRDM', 'compute_model_rdm', 'flatten_upper_tri',
    'spearman_correlation', 'sample_rdm_batch',
    'load_eye_rdm', 'compute_task_centroid_rdm', 'eye_rsa_loss',
]

