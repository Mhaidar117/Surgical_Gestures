"""
Multi-task JIGSAWS dataset for brain alignment training.
Combines all 3 tasks (Suturing, Needle Passing, Knot Tying) with task_label for task-centroid RSA.
"""
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset

from .jigsaws_vit_dataset import JIGSAWSViTDataset
from .split_loader import SplitLoader


# Task label mapping: Suturing=0, Needle Passing=1, Knot Tying=2
# Matches row/col order of target_rdm_3x3.npy from eye-tracking exploration
TASKS = ['Suturing', 'Needle_Passing', 'Knot_Tying']
TASK_TO_LABEL = {t: i for i, t in enumerate(TASKS)}


def filter_dataset_by_trials(
    dataset: JIGSAWSViTDataset,
    trial_ids: List[str],
    segment_filter: Optional[Dict[str, Dict]] = None,
) -> Subset:
    """Filter dataset to only include samples from specific trials.

    If ``segment_filter`` is provided, further restricts each trial's samples
    by frame range. See ``train_vit_system.filter_dataset_by_trials`` for the
    accepted bounds dict shape.
    """
    indices = []
    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        sample_trial = sample.get('trial_id', '')
        matched = None
        for tid in trial_ids:
            if tid in sample_trial or sample_trial in tid:
                matched = tid
                break
        if matched is None:
            continue
        if segment_filter:
            bounds = segment_filter.get(matched) or segment_filter.get(sample_trial)
            if bounds:
                sf = sample.get('start_frame')
                ef = sample.get('end_frame')
                if 'end_frame_max' in bounds and ef is not None and ef > bounds['end_frame_max']:
                    continue
                if 'start_frame_min' in bounds and sf is not None and sf < bounds['start_frame_min']:
                    continue
        indices.append(idx)
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
        split_family: str = 'louo',
        **kwargs
    ):
        """
        Args:
            data_root: Root directory containing Gestures/ folder
            split_name: Split name (e.g., 'fold_1')
            mode: 'train', 'val', or 'test'
            task_type: 'gesture', 'skill', or 'kinematics'
            use_rgb: Whether to load RGB frames
            use_flow: Whether to load optical flow
            arm: 'PSM1' or 'PSM2'
            split_family: Which splits file to read. See SplitLoader docs.
            **kwargs: Additional args passed to JIGSAWSViTDataset
        """
        self.data_root = Path(data_root)
        self.split_name = split_name
        self.mode = mode
        self.task_type = task_type
        self.arm = arm
        self.split_family = split_family

        # Build per-task subsets
        self.task_datasets: List[Subset] = []
        self.task_labels: List[int] = []  # task_label for each sample
        self.cumulative_sizes: List[int] = [0]

        for task in TASKS:
            split_loader = SplitLoader(str(data_root), task, split_name, split_family=split_family)
            trials = (
                split_loader.get_train_trials()
                if mode == 'train'
                else split_loader.get_val_trials()
                if mode == 'val'
                else split_loader.get_test_trials()
            )
            seg_filter = split_loader.get_segment_filter(mode)

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

            subset = filter_dataset_by_trials(full_dataset, trials, segment_filter=seg_filter)
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

        # Add task_label for brain alignment (as tensor, consistent with other labels)
        sample['task_label'] = torch.tensor(self.task_labels[idx], dtype=torch.long)
        return sample
