"""
Multi-task JIGSAWS dataset for brain alignment training.
Combines all 3 tasks (Suturing, Needle Passing, Knot Tying) with task_label for task-centroid RSA.
"""
from pathlib import Path
from typing import Dict, List, Optional

from torch.utils.data import Dataset, ConcatDataset, Subset

from .jigsaws_vit_dataset import JIGSAWSViTDataset
from .split_loader import SplitLoader


# Task label mapping: Suturing=0, Needle Passing=1, Knot Tying=2
# Matches row/col order of target_rdm_3x3.npy from eye-tracking exploration
TASKS = ['Suturing', 'Needle_Passing', 'Knot_Tying']
TASK_TO_LABEL = {t: i for i, t in enumerate(TASKS)}


def filter_dataset_by_trials(dataset: JIGSAWSViTDataset, trial_ids: List[str]) -> Subset:
    """Filter dataset to only include samples from specific trials."""
    indices = []
    for idx in range(len(dataset)):
        sample_trial = dataset.samples[idx].get('trial_id', '')
        for tid in trial_ids:
            if tid in sample_trial or sample_trial in tid:
                indices.append(idx)
                break
    return Subset(dataset, indices)


class JIGSAWSMultiTaskDataset(Dataset):
    """
    Multi-task dataset combining Suturing, Needle Passing, and Knot Tying.
    Each sample includes task_label (0, 1, 2) for task-centroid RSA.
    """

    def __init__(
        self,
        data_root: str,
        split_name: str,
        mode: str = 'train',
        task_type: str = 'gesture',
        use_rgb: bool = True,
        use_flow: bool = True,
        arm: str = 'PSM2',
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing Gestures/ folder
            split_name: LOUO split name (e.g., 'fold_1')
            mode: 'train', 'val', or 'test'
            task_type: 'gesture', 'skill', or 'kinematics'
            use_rgb: Whether to load RGB frames
            use_flow: Whether to load optical flow
            arm: 'PSM1' or 'PSM2'
            **kwargs: Additional args passed to JIGSAWSViTDataset
        """
        self.data_root = Path(data_root)
        self.split_name = split_name
        self.mode = mode
        self.task_type = task_type
        self.arm = arm

        # Build per-task subsets
        self.task_datasets: List[Subset] = []
        self.task_labels: List[int] = []  # task_label for each sample
        self.cumulative_sizes: List[int] = [0]

        for task in TASKS:
            split_loader = SplitLoader(str(data_root), task, split_name)
            trials = (
                split_loader.get_train_trials()
                if mode == 'train'
                else split_loader.get_val_trials()
                if mode == 'val'
                else split_loader.get_test_trials()
            )

            full_dataset = JIGSAWSViTDataset(
                data_root=str(data_root),
                task=task,
                mode=mode,
                task_type=task_type,
                use_rgb=use_rgb,
                use_flow=use_flow,
                arm=arm,
                **kwargs
            )

            subset = filter_dataset_by_trials(full_dataset, trials)
            self.task_datasets.append(subset)
            self.task_labels.extend([TASK_TO_LABEL[task]] * len(subset))
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(subset))

        self._total_len = self.cumulative_sizes[-1]

    def _get_task_and_local_idx(self, idx: int) -> tuple:
        """Map flat index to (task_idx, local_idx)."""
        for task_idx in range(3):
            if idx < self.cumulative_sizes[task_idx + 1]:
                local_idx = idx - self.cumulative_sizes[task_idx]
                return task_idx, local_idx
        raise IndexError(f"Index {idx} out of range")

    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> Dict:
        task_idx, local_idx = self._get_task_and_local_idx(idx)
        subset = self.task_datasets[task_idx]
        sample = subset[local_idx]

        # Add task_label for brain alignment
        sample['task_label'] = self.task_labels[idx]
        return sample
