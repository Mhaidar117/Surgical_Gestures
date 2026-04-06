"""Phase 4: ViT training integration for coarse bridge RDM targets."""

from .target_loader import (
    BridgeRDMTarget,
    align_bridge_target_to_jigsaws_task_family,
    load_bridge_target_from_manifest,
)
from .label_grouping import expand_group_labels_for_bridge

__all__ = [
    "BridgeRDMTarget",
    "align_bridge_target_to_jigsaws_task_family",
    "load_bridge_target_from_manifest",
    "expand_group_labels_for_bridge",
]
