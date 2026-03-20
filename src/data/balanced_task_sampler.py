"""
Balanced batch sampler for multi-task training.
Ensures each batch contains samples from all 3 tasks for task-centroid RSA.
"""
import random
from typing import Iterator, List

from torch.utils.data import Sampler


class BalancedTaskBatchSampler(Sampler[List[int]]):
    """
    Batch sampler that ensures each batch has samples from all 3 tasks.
    Task labels: Suturing=0, Needle Passing=1, Knot Tying=2.
    """

    def __init__(
        self,
        task_labels: List[int],
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True
    ):
        """
        Args:
            task_labels: List of task labels (0, 1, 2) for each dataset index
            batch_size: Number of samples per batch
            drop_last: Whether to drop last incomplete batch
            shuffle: Whether to shuffle indices within batches
        """
        self.task_labels = task_labels
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle

        # Group indices by task
        self.task_indices: List[List[int]] = [[], [], []]
        for idx, tl in enumerate(task_labels):
            if 0 <= tl <= 2:
                self.task_indices[tl].append(idx)

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle each task's indices
        task_indices = [list(idx_list) for idx_list in self.task_indices]
        if self.shuffle:
            for lst in task_indices:
                random.shuffle(lst)

        # Samples per task per batch (ensure at least 1 per task)
        n_per_task = max(1, self.batch_size // 3)
        remainder = self.batch_size - 3 * n_per_task
        samples_per_task = [n_per_task, n_per_task, n_per_task]
        for i in range(remainder):
            samples_per_task[i] += 1

        # Round-robin: take samples_per_task[i] from each task per batch
        task_pos = [0, 0, 0]
        total = sum(len(lst) for lst in task_indices)

        while True:
            batch = []
            for task_id in range(3):
                for _ in range(samples_per_task[task_id]):
                    if task_pos[task_id] < len(task_indices[task_id]):
                        batch.append(task_indices[task_id][task_pos[task_id]])
                        task_pos[task_id] += 1

            if len(batch) == 0:
                break

            if len(batch) < self.batch_size and self.drop_last:
                break

            # Ensure we have at least one from each task (skip incomplete batches)
            if len(batch) < self.batch_size and self.drop_last:
                break

            if self.shuffle:
                random.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        n_samples = sum(len(lst) for lst in self.task_indices)
        if self.drop_last:
            return n_samples // self.batch_size
        return (n_samples + self.batch_size - 1) // self.batch_size
