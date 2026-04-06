"""
Map JIGSAWS batch fields to per-frame group ids for centroid RDMs.

``task`` grouping uses ``task_label`` (0,1,2) aligned with
``Suturing``, ``Needle_Passing``, ``Knot_Tying``.

``subskill`` grouping maps each ``gesture_label`` (0..14) to a family id
via config ``gesture_to_subskill_family``: list of 15 integers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

# Must match src/data/jigsaws_multitask_dataset.TASKS
JIGSAWS_TASK_ORDER = ("Suturing", "Needle_Passing", "Knot_Tying")


def expand_task_labels_to_frames(
    task_labels: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    """
    Repeat each sample's task label across T frames.

    Args:
        task_labels: (B,)
        time_steps: T

    Returns:
        (B * T,) on same device as task_labels
    """
    B = task_labels.shape[0]
    device = task_labels.device
    return task_labels.unsqueeze(1).expand(B, time_steps).reshape(B * time_steps)


def gesture_to_family_tensor(
    gesture_labels: torch.Tensor,
    mapping: Sequence[int],
) -> torch.Tensor:
    """
    Map gesture ids to subskill family ids.

    Args:
        gesture_labels: (N,) int64/int32
        mapping: length 15; mapping[g] = family id

    Returns:
        (N,) long tensor of family ids
    """
    if len(mapping) != 15:
        raise ValueError(
            f"gesture_to_subskill_family must have length 15, got {len(mapping)}"
        )
    g = gesture_labels.long().clamp(0, 14)
    table = torch.tensor(list(mapping), device=gesture_labels.device, dtype=torch.long)
    return table[g]


def expand_group_labels_for_bridge(
    batch: Dict,
    grouping: str,
    time_steps: int,
    *,
    gesture_to_subskill_family: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    Build per-frame group labels (B*T,) for centroid RDM computation.

    Args:
        batch: Must contain ``task_label``; for subskill also ``gesture_label``.
        grouping: ``task`` or ``subskill``.
        time_steps: T from ViT embeddings.
        gesture_to_subskill_family: Required when grouping == ``subskill``.

    Returns:
        Long tensor (B,) or (B*T,) — for task we return (B*T,) expanded
        to match flattened embeddings.
    """
    device = batch["task_label"].device
    if grouping == "task":
        return expand_task_labels_to_frames(batch["task_label"], time_steps)
    if grouping == "subskill":
        if gesture_to_subskill_family is None:
            raise ValueError(
                "bridge_grouping=subskill requires gesture_to_subskill_family in config"
            )
        gesture = batch["gesture_label"]
        fam = gesture_to_family_tensor(gesture, gesture_to_subskill_family)
        B = gesture.shape[0]
        return fam.unsqueeze(1).expand(B, time_steps).reshape(B * time_steps)
    raise ValueError(f"Unknown bridge_grouping: {grouping!r}")


def default_task_label_order_matches_jigsaws(unit_labels: List[str]) -> bool:
    """Return True if unit_labels match TASKS order (Suturing, Needle_Passing, Knot_Tying)."""
    if len(unit_labels) != 3:
        return False
    return [str(u) for u in unit_labels] == list(JIGSAWS_TASK_ORDER)


def permutation_aligning_labels_to_jigsaws(unit_labels: List[str]) -> List[int]:
    """
    Row/col permutation so that reordered labels match ``JIGSAWS_TASK_ORDER``.

    Returns:
        List ``perm`` of length 3 with ``unit_labels[perm[i]] == JIGSAWS_TASK_ORDER[i]``
        (string equality).

    Raises:
        ValueError: if labels cannot be matched one-to-one to the three task names.
    """
    if len(unit_labels) != 3:
        raise ValueError("Expected three unit_labels for task-family alignment")
    want = list(JIGSAWS_TASK_ORDER)
    have = [str(u) for u in unit_labels]
    perm: List[int] = []
    used = set()
    for w in want:
        matches = [i for i, h in enumerate(have) if h == w and i not in used]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot align unit_labels {have} to JIGSAWS order {want}: "
                f"ambiguous or missing match for {w!r}"
            )
        perm.append(matches[0])
        used.add(matches[0])
    return perm


def remap_group_labels(
    labels: torch.Tensor,
    order_map: Optional[Sequence[int]],
    num_groups: int,
) -> torch.Tensor:
    """
    Permute group ids so model row/col order matches target RDM order.

    Args:
        labels: (N,) long — group indices in ``0..num_groups-1`` in source order
        order_map: length ``num_groups``; ``order_map[i]`` = target index for source i.
            If None, identity.

    Returns:
        (N,) long tensor with remapped ids
    """
    if order_map is None:
        return labels.long()
    if len(order_map) != num_groups:
        raise ValueError(
            f"bridge.unit_label_order must have length {num_groups}, got {len(order_map)}"
        )
    table = torch.tensor(list(order_map), device=labels.device, dtype=torch.long)
    return table[labels.long().clamp(0, num_groups - 1)]
