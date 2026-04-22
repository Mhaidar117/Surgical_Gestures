"""
Main training script for ViT-based system with config-driven loop.

Usage:
    # Train with proper LOUO split
    python src/training/train_vit_system.py \
        --config src/configs/baseline.yaml \
        --data_root . \
        --task Knot_Tying \
        --split fold_1 \
        --output_dir checkpoints/knot_tying_fold1

    # Train on all data (no validation - not recommended)
    python src/training/train_vit_system.py \
        --config src/configs/baseline.yaml \
        --data_root . \
        --task Knot_Tying \
        --output_dir checkpoints/knot_tying_all
"""
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional, List
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.visual import ViTFrameEncoder
from models.temporal_transformer import TemporalAggregatorWithPooling
from models.kinematics import KinematicsModule
from models.losses import compute_total_loss, surgeon_ids_from_trial_ids
from modules.brain_rdm import (
    BrainRDM,
    compute_model_rdm,
    sample_rdm_batch,
    load_eye_rdm,
    compute_task_centroid_rdm,
    compute_centroid_rdm,
)
# Import optim - handle both normal import and importlib loading
try:
    # Try relative import first (works when imported as package)
    from .optim import get_optimizer, get_scheduler
except (ImportError, ValueError):
    # Fallback: load directly from file (works when loaded via importlib)
    import importlib.util
    optim_path = Path(__file__).parent / "optim.py"
    spec = importlib.util.spec_from_file_location("training.optim", optim_path)
    optim_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(optim_module)
    get_optimizer = optim_module.get_optimizer
    get_scheduler = optim_module.get_scheduler
from data import JIGSAWSViTDataset, JIGSAWSMultiTaskDataset, BalancedTaskBatchSampler
from data.split_loader import SplitLoader
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F


def pad_collate_fn(batch):
    """Collate samples with variable-length kinematics (and optionally rgb/flow) by zero-padding.

    Tensors keyed by 'kinematics', 'rgb', and 'flow' may have a variable leading
    time dimension T across samples in the same batch.  All others are either
    fixed-size tensors (scalars / feature vectors) or plain Python objects
    (strings).  Strategy:
      • Variable-T tensors  → pad to max(T) along dim 0 with zeros.
      • Fixed-size tensors  → torch.stack as usual.
      • None entries        → keep as None (whole key collapses to None).
      • Strings / other     → collect into a list.
    """
    # Keys whose first dimension is the time axis and may vary across samples.
    _variable_len_keys = {'kinematics', 'rgb', 'flow'}

    if not batch:
        return {}

    keys = batch[0].keys()
    out = {}

    for key in keys:
        values = [sample[key] for sample in batch]

        # All None → return None
        if all(v is None for v in values):
            out[key] = None
            continue

        # Mixed None / tensor → replace None with zero tensor matching others
        non_none = [v for v in values if v is not None]
        if any(v is None for v in values):
            ref = non_none[0]
            values = [v if v is not None else torch.zeros_like(ref) for v in values]

        if not isinstance(values[0], torch.Tensor):
            # Scalar ints/floats → convert to tensor so downstream code can call .device etc.
            if isinstance(values[0], (int, float)):
                out[key] = torch.tensor(values)
            else:
                # Strings and other non-numeric types — keep as list
                out[key] = values
            continue

        if key in _variable_len_keys:
            # Pad along dim 0 (time) to the longest sequence in the batch
            max_t = max(v.shape[0] for v in values)
            padded = []
            for v in values:
                pad_len = max_t - v.shape[0]
                if pad_len > 0:
                    # F.pad pads from the last dim; for a 2-D tensor [T, D] we
                    # need to pad (last_dim_end, last_dim_start, T_end, T_start)
                    pad_spec = [0, 0] * (v.dim() - 1) + [0, pad_len]
                    v = F.pad(v, pad_spec)
                padded.append(v)
            out[key] = torch.stack(padded, dim=0)
        else:
            out[key] = torch.stack(values, dim=0)

    return out


class EEGInformedViTModel(nn.Module):
    """Unified model combining all components."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        print("=" * 60)
        print("Initializing EEGInformedViTModel")
        print("=" * 60)
        
        # Visual encoder
        print(f"Creating visual encoder: {config.get('model_name', 'vit_small_patch16_224')}")
        self.visual = ViTFrameEncoder(
            model_name=config.get('model_name', 'vit_small_patch16_224'),
            pretrained=config.get('pretrained', True),
            freeze_until=config.get('freeze_until', 6),
            use_adapters=config.get('use_adapters', False),
            adapter_dim=config.get('adapter_dim', 64)
        )
        print(f"  - Pretrained: {config.get('pretrained', True)}")
        print(f"  - Freeze until layer: {config.get('freeze_until', 6)}")
        print(f"  - Use adapters: {config.get('use_adapters', False)}")
        
        # Optional flow encoder
        self.use_flow = config.get('use_flow', False)
        if self.use_flow:
            print("Creating flow encoder...")
            from models.visual import ViTFlowEncoder
            self.flow_encoder = ViTFlowEncoder(self.visual)
        else:
            print("Flow encoder: disabled")
        
        # Temporal aggregator
        print("Creating temporal aggregator...")
        self.temporal = TemporalAggregatorWithPooling(
            d_model=384,
            n_heads=6,
            num_layers=4
        )
        print(f"  - d_model: 384, n_heads: 6, num_layers: 4")
        
        # Kinematics module
        print("Creating kinematics module...")
        # d_kin_input should match the dataset output:
        # - 19 for single arm (PSM1 or PSM2): pos(3) + rot(9) + trans_vel(3) + rot_vel(3) + gripper(1)
        d_kin_input = config.get('d_kin_input', 19)
        d_kin_output = config.get('d_kin_output', 19)
        self.kinematics = KinematicsModule(
            d_model=384,
            d_kin_input=d_kin_input,
            d_kin_output=d_kin_output,
            num_gestures=15,
            num_skills=3
        )
        print(f"  - d_kin_input: {d_kin_input}, d_kin_output: {d_kin_output}")
        print(f"  - num_gestures: 15, num_skills: 3")
        
        # Brain RDM (tau / precomputed trial RDMs — only used for brain_mode == rsa)
        self.brain_mode = config.get('brain_mode', 'none')
        if self.brain_mode == 'rsa':
            print(f"Creating Brain RDM module (mode: {self.brain_mode})...")
            self.brain_rdm = BrainRDM(
                cache_dir=config.get('eeg_rdm_cache_dir'),
                tau_range=config.get('tau_range', [0, 50, 100, 150, 200, 250, 300])
            )
        else:
            self.brain_rdm = None
            if self.brain_mode == 'none':
                print("Brain RDM: disabled")
            else:
                print(
                    f"Brain RDM module: not instantiated (mode={self.brain_mode}; "
                    "eye/bridge use fixed target RDMs in train loop)"
                )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"  - Total: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"  - Trainable: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print("=" * 60)
    
    def forward(
        self,
        rgb: torch.Tensor,
        flow: Optional[torch.Tensor] = None,
        target_kinematics: Optional[torch.Tensor] = None,
        teacher_forcing_prob: float = 1.0,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Encode frames
        emb_layers, pooled_emb = self.visual(rgb)
        
        # Optional flow encoding
        if self.use_flow and flow is not None:
            flow_emb = self.flow_encoder(flow)
            # Concatenate or combine
            pooled_emb = (pooled_emb + flow_emb) / 2.0
        
        # Temporal aggregation
        memory, mean_pooled, attn_pooled = self.temporal(pooled_emb)
        
        # Decode kinematics
        outputs = self.kinematics(
            pooled_emb,
            memory,
            target_kinematics,
            teacher_forcing_prob
        )
        
        if return_embeddings:
            outputs['embeddings'] = emb_layers
            outputs['memory'] = memory
        
        return outputs


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Dict,
    epoch: int,
    target_rdm: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    component_losses = {}
    num_batches = 0
    brain_mode = config.get('brain_mode', 'none')
    use_embeddings = brain_mode in ('rsa', 'eye', 'bridge', 'kinematics_rsa')
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        try:
            rgb = batch['rgb'].contiguous().to(device)
            kinematics = batch['kinematics'].contiguous().to(device)
            gesture_labels = batch['gesture_label'].contiguous().to(device)
            skill_labels = batch['skill_label'].contiguous().to(device)
        except RuntimeError as e:
            print(f"\nError moving to device. Tensor shapes:")
            print(f"  rgb: {batch['rgb'].shape}")
            print(f"  kinematics: {batch['kinematics'].shape}")
            print(f"  gesture_labels: {batch['gesture_label'].shape}")
            print(f"  skill_labels: {batch['skill_label'].shape}")
            raise
        
        # Forward pass (return embeddings when brain alignment is enabled)
        teacher_forcing_prob = max(0.3, 1.0 - (epoch / config.get('teacher_forcing_decay_epochs', 40)))
        
        outputs = model(
            rgb,
            target_kinematics=kinematics,
            teacher_forcing_prob=teacher_forcing_prob,
            return_embeddings=use_embeddings
        )
        
        # Compute brain alignment if needed
        model_rdm = None
        eeg_rdm = None
        model_features = None
        eeg_patterns = None
        
        if brain_mode == 'eye' and target_rdm is not None and 'task_label' in batch:
            # Task-centroid RSA: use ViT embeddings and task labels
            emb = None
            features = outputs.get('embeddings', {})
            brain_layer = config.get('brain_layer', 'mid')
            if isinstance(features, dict):
                emb = features.get(brain_layer)
                if emb is None:
                    emb = features.get('mid')
                if emb is None:
                    emb = features.get('late')
                if emb is None:
                    emb = features.get('early')
                if emb is None and features:
                    emb = next(iter(features.values()))
            if emb is None and 'memory' in outputs:
                emb = outputs['memory']
            if emb is not None:
                B, T, D = emb.shape[0], emb.shape[1], emb.shape[2]
                embeddings_flat = emb.view(B * T, D)
                task_labels = batch['task_label'].to(device)
                task_labels_expanded = task_labels.unsqueeze(1).expand(-1, T).reshape(B * T)
                model_rdm = compute_task_centroid_rdm(embeddings_flat, task_labels_expanded)
                eeg_rdm = target_rdm.to(device)
                if not config.get('_brain_branch_fired', False):
                    print(
                        f"\n[brain] active branch fired (mode=eye, "
                        f"model_rdm shape={tuple(model_rdm.shape)}, "
                        f"target shape={tuple(eeg_rdm.shape)})",
                        flush=True,
                    )
                    config['_brain_branch_fired'] = True
        elif (
            brain_mode == 'bridge'
            and target_rdm is not None
            and config.get('_phase4_bridge') is not None
            and 'task_label' in batch
        ):
            from eeg_eye_bridge.phase4_vit.label_grouping import (
                expand_group_labels_for_bridge,
                remap_group_labels,
            )

            emb = None
            features = outputs.get('embeddings', {})
            brain_layer = config.get('brain_layer', 'mid')
            if isinstance(features, dict):
                emb = features.get(brain_layer)
                if emb is None:
                    emb = features.get('mid')
                if emb is None:
                    emb = features.get('late')
                if emb is None:
                    emb = features.get('early')
                if emb is None and features:
                    emb = next(iter(features.values()))
            if emb is None and 'memory' in outputs:
                emb = outputs['memory']
            if emb is not None:
                B, T, D = emb.shape[0], emb.shape[1], emb.shape[2]
                embeddings_flat = emb.view(B * T, D)
                bridge_meta = config['_phase4_bridge']
                num_groups = bridge_meta['num_groups']
                grouping = config.get('bridge_grouping', 'task')
                gesture_map = config.get('gesture_to_subskill_family')
                group_labels = expand_group_labels_for_bridge(
                    batch,
                    grouping,
                    T,
                    gesture_to_subskill_family=gesture_map,
                ).to(device)
                order_map = config.get('bridge_unit_label_order')
                if order_map is not None:
                    group_labels = remap_group_labels(
                        group_labels, order_map, num_groups
                    ).to(device)
                model_rdm = compute_centroid_rdm(
                    embeddings_flat, group_labels, num_groups
                )
                eeg_rdm = target_rdm.to(device)
                if model_rdm.shape != eeg_rdm.shape:
                    print(
                        f"WARNING: bridge model RDM shape {tuple(model_rdm.shape)} != "
                        f"target {tuple(eeg_rdm.shape)}; skipping brain term this batch"
                    )
                    model_rdm = None
                    eeg_rdm = None
                elif not config.get('_brain_branch_fired', False):
                    print(
                        f"\n[brain] active branch fired (mode=bridge, grouping={grouping}, "
                        f"model_rdm shape={tuple(model_rdm.shape)}, "
                        f"target shape={tuple(eeg_rdm.shape)}, "
                        f"num_groups_present={int(group_labels.unique().numel())})",
                        flush=True,
                    )
                    config['_brain_branch_fired'] = True
        elif brain_mode == 'kinematics_rsa' and 'memory' in outputs:
            # Stimulus-locked teacher signal: align per-sample video embedding
            # geometry with per-sample kinematic-trajectory geometry within the
            # same batch. Each sample = one gesture segment. Same subject, same
            # moment, two representations of the same act.
            from modules.brain_rdm import (
                pairwise_distance_rdm,
                kinematics_trajectory_features,
            )
            memory = outputs['memory']  # (B, T, D)
            model_emb = memory.mean(dim=1)  # (B, D)
            kin_feat = kinematics_trajectory_features(kinematics)  # (B, 2K)
            model_rdm = pairwise_distance_rdm(model_emb)
            eeg_rdm = pairwise_distance_rdm(kin_feat).detach()
            if not config.get('_brain_branch_fired', False):
                print(
                    f"\n[brain] active branch fired (mode=kinematics_rsa, "
                    f"B={model_emb.shape[0]}, D={model_emb.shape[1]}, "
                    f"kin_feat_dim={kin_feat.shape[-1]})",
                    flush=True,
                )
                config['_brain_branch_fired'] = True
        elif brain_mode == 'rsa' and 'embeddings' in outputs and model.brain_rdm is not None:
            # EEG RSA: sample features for RDM
            features = outputs['embeddings'].get('mid', outputs['memory'])
            if len(features.shape) == 3:
                features = features.view(-1, features.shape[-1])
            sampled_features, _ = sample_rdm_batch(features, batch_size=config.get('rdm_batch_size', 32))
            model_rdm = compute_model_rdm(sampled_features, method='pearson')
            batch_meta = {'trial_ids': batch.get('trial_id', [])}
            eeg_rdm = model.brain_rdm.get_eeg_rdm(batch_meta, tau=config.get('tau', 0))
        
        # Determine kinematics format based on output dimension
        d_kin_output = config.get('d_kin_output', 19)
        kin_format = '19d' if d_kin_output == 19 else '10d'

        # Surgeon-conditioned skill-contrastive inputs (only materialized if the
        # weight is >0; otherwise compute_total_loss ignores them).
        skill_contra_weight = (config.get('loss_weights') or {}).get('skill_contra', 0.0)
        skill_embeddings = None
        surgeon_ids = None
        if skill_contra_weight and 'memory' in outputs:
            skill_embeddings = outputs['memory'].mean(dim=1)
            trial_ids = batch.get('trial_id', [])
            if isinstance(trial_ids, (list, tuple)):
                surgeon_ids = surgeon_ids_from_trial_ids(list(trial_ids))

        # Compute loss
        loss, losses = compute_total_loss(
            outputs['kinematics'],
            kinematics,
            outputs['gesture_logits'],
            gesture_labels,
            outputs['skill_logits'],
            skill_labels,
            model_rdm=model_rdm,
            eeg_rdm=eeg_rdm,
            brain_mode=config.get('brain_mode', 'none'),
            loss_weights=config.get('loss_weights'),
            kinematics_format=kin_format,
            skill_embeddings=skill_embeddings,
            surgeon_ids=surgeon_ids,
            skill_contra_temperature=config.get('skill_contra_temperature', 0.07),
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        for k, v in losses.items():
            if k not in component_losses:
                component_losses[k] = 0.0
            component_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
        num_batches += 1
        
        # Update progress bar with current loss
        current_loss = loss.item()
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'avg_loss': f'{total_loss / num_batches:.4f}'
        })
    
    # Average losses
    avg_losses = {k: v / num_batches for k, v in component_losses.items()}
    avg_losses['total'] = total_loss / num_batches

    return avg_losses


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    config: Dict
) -> Dict[str, float]:
    """Validate for one epoch.

    Mirrors the kinematics_rsa + skill_contra branches from ``train_epoch`` so
    the reported val component losses include brain_rsa / skill_contra when the
    corresponding config weights are nonzero. The eye/bridge brain modes are
    not replicated here (they rely on task_label expansion and a preloaded
    target RDM; add if you need val-side tracking).
    """
    model.eval()
    total_loss = 0.0
    component_losses = {}
    num_batches = 0
    brain_mode = config.get('brain_mode', 'none')
    skill_contra_weight = (config.get('loss_weights') or {}).get('skill_contra', 0.0)
    return_embeddings = brain_mode in ('kinematics_rsa',) or skill_contra_weight > 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating", leave=False)

        for batch in pbar:
            # Move to device
            rgb = batch['rgb'].to(device)
            kinematics = batch['kinematics'].to(device)
            gesture_labels = batch['gesture_label'].to(device)
            skill_labels = batch['skill_label'].to(device)

            # Forward pass (no teacher forcing during validation)
            outputs = model(
                rgb,
                target_kinematics=kinematics,
                teacher_forcing_prob=0.0,
                return_embeddings=return_embeddings,
            )

            # Determine kinematics format based on output dimension
            d_kin_output = config.get('d_kin_output', 19)
            kin_format = '19d' if d_kin_output == 19 else '10d'

            model_rdm = None
            eeg_rdm = None
            skill_embeddings = None
            surgeon_ids = None

            if brain_mode == 'kinematics_rsa' and 'memory' in outputs:
                from modules.brain_rdm import (
                    pairwise_distance_rdm,
                    kinematics_trajectory_features,
                )
                model_emb = outputs['memory'].mean(dim=1)
                kin_feat = kinematics_trajectory_features(kinematics)
                model_rdm = pairwise_distance_rdm(model_emb)
                eeg_rdm = pairwise_distance_rdm(kin_feat).detach()

            if skill_contra_weight and 'memory' in outputs:
                skill_embeddings = outputs['memory'].mean(dim=1)
                trial_ids = batch.get('trial_id', [])
                if isinstance(trial_ids, (list, tuple)):
                    surgeon_ids = surgeon_ids_from_trial_ids(list(trial_ids))

            # Compute loss
            loss, losses = compute_total_loss(
                outputs['kinematics'],
                kinematics,
                outputs['gesture_logits'],
                gesture_labels,
                outputs['skill_logits'],
                skill_labels,
                model_rdm=model_rdm,
                eeg_rdm=eeg_rdm,
                brain_mode=brain_mode,
                loss_weights=config.get('loss_weights'),
                kinematics_format=kin_format,
                skill_embeddings=skill_embeddings,
                surgeon_ids=surgeon_ids,
                skill_contra_temperature=config.get('skill_contra_temperature', 0.07),
            )

            # Accumulate losses
            total_loss += loss.item()
            for k, v in losses.items():
                if k not in component_losses:
                    component_losses[k] = 0.0
                component_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
            num_batches += 1

            pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

    # Average losses
    avg_losses = {k: v / num_batches for k, v in component_losses.items()}
    avg_losses['total'] = total_loss / num_batches

    return avg_losses


def filter_dataset_by_trials(
    dataset: JIGSAWSViTDataset,
    trial_ids: List[str],
    segment_filter: Optional[Dict[str, Dict]] = None,
) -> Subset:
    """Filter dataset to only include samples from specific trials.

    If ``segment_filter`` is provided, further restricts each trial's samples
    by frame range. Each value is a dict with one of:
      - ``{'end_frame_max': int}``   -> keep segments with ``end_frame <= bound``
      - ``{'start_frame_min': int}`` -> keep segments with ``start_frame >= bound``

    Trials absent from ``segment_filter`` keep all their segments.
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


def main():
    parser = argparse.ArgumentParser(description='Train ViT system')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--task', type=str, default='Knot_Tying',
                       help='Task name (Knot_Tying, Needle_Passing, Suturing, or "all" for multi-task brain alignment)')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    parser.add_argument('--arm', type=str, default='PSM2', help='Arm to use (PSM1 or PSM2)')
    parser.add_argument('--split', type=str, default=None,
                        help='Split name (e.g., fold_1). If None, uses all data (no validation).')
    parser.add_argument('--split_family', type=str, default='louo',
                        choices=['louo', 'inter_trial_within_subject', 'intra_trial_half'],
                        help='Which splits file to read. louo = standard cross-subject; '
                             'inter_trial_within_subject = hold out one trial per surgeon; '
                             'intra_trial_half = within-subject temporal (early frames train, late frames test).')

    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("ViT Training System")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Task: {args.task}")
    print(f"Split: {args.split if args.split else 'None (using all data)'}")
    print(f"Split family: {args.split_family}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # Load config
    print("\nLoading configuration...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"  - Batch size: {config.get('batch_size', 16)}")
    print(f"  - Number of epochs: {config.get('num_epochs', 80)}")
    print(f"  - Learning rate (base): {config.get('lr_base', 1e-4)}")
    print(f"  - Learning rate (ViT): {config.get('lr_vit', 1e-5)}")
    print(f"  - Brain mode: {config.get('brain_mode', 'none')}")
    
    # Create model
    print("\nCreating model...")
    model = EEGInformedViTModel(config)
    
    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device('mps')
        print(f"\nUsing device: MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"\nUsing device: CUDA")
    else:
        device = torch.device('cpu')
        print(f"\nUsing device: CPU")
    
    print(f"Moving model to {device}...")
    model = model.to(device)
    print("Model moved to device successfully.")
    
    # Load target RDM for brain alignment (eye mode)
    target_rdm = None
    config['_phase4_bridge'] = None
    if config.get('brain_mode') == 'eye':
        eye_rdm_path = config.get('eye_rdm_path', 'Eye/Exploration/target_rdm_3x3.npy')
        rdm_path = Path(args.data_root) / eye_rdm_path if not Path(eye_rdm_path).is_absolute() else Path(eye_rdm_path)
        if rdm_path.exists():
            target_rdm = load_eye_rdm(str(rdm_path))
            print(f"\nLoaded eye-tracking target RDM from {rdm_path}")
        else:
            print(f"\nWARNING: Eye RDM not found at {rdm_path}, brain alignment disabled")
    elif config.get('brain_mode') == 'bridge':
        from eeg_eye_bridge.phase4_vit.target_loader import (
            load_bridge_target_from_manifest,
            align_bridge_target_to_jigsaws_task_family,
        )

        bridge_cfg = config.get('bridge') or {}
        manifest_rel = bridge_cfg.get(
            'manifest_path', 'cache/eeg_eye_bridge/phase3/rdm_manifest.json'
        )
        target_key = bridge_cfg.get('target_key')
        if not target_key:
            raise ValueError("bridge.target_key is required when brain_mode is 'bridge'")
        manifest_path = Path(manifest_rel)
        if not manifest_path.is_absolute():
            cand = Path(args.data_root) / manifest_rel
            manifest_path = cand if cand.is_file() else (Path.cwd() / manifest_rel)
        manifest_path = manifest_path.resolve()
        print(f"\nLoading Phase 3 bridge target from manifest: {manifest_path}")
        bt = load_bridge_target_from_manifest(manifest_path, target_key)
        if config.get('bridge_grouping', 'task') == 'task' and bt.num_groups == 3:
            try:
                bt = align_bridge_target_to_jigsaws_task_family(bt)
                print("  Aligned target RDM rows/cols to JIGSAWS task order.")
            except ValueError as e:
                print(
                    f"  WARNING: Could not auto-align unit_labels to JIGSAWS task order: {e}"
                )
        target_rdm = bt.matrix
        config['_phase4_bridge'] = {
            'num_groups': bt.num_groups,
            'unit_labels': list(bt.unit_labels),
            'name': bt.name,
            'rdm_type': bt.rdm_type,
            'unit_type': bt.unit_type,
        }
        print(
            f"  Bridge target {bt.name!r}: K={bt.num_groups}, unit_type={bt.unit_type!r}"
        )
        # Hard assertion: bridge grouping (task or subskill) must match a 3x3 target
        # so the model centroid RDM and the target RDM align. Fails LOUDLY at startup
        # instead of silently zero-padding centroids at batch time.
        bridge_grouping = config.get('bridge_grouping', 'task')
        if bridge_grouping in ('task', 'subskill'):
            if target_rdm.shape != (3, 3):
                raise ValueError(
                    f"bridge_grouping={bridge_grouping!r} requires a 3x3 target RDM, "
                    f"but target_key={target_key!r} produced shape {tuple(target_rdm.shape)}. "
                    "Pick a 3x3 manifest entry (e.g. eeg_latent_subskill_family, "
                    "eye_only_subskill_family, joint_eye_eeg_subskill_family) or change "
                    "bridge_grouping."
                )

    # Create dataset
    use_multi_task = args.task.lower() == 'all'
    if use_multi_task:
        if args.split is None:
            raise ValueError("--split is required when --task all (e.g., --split fold_1)")
        print(f"\nLoading multi-task dataset (all 3 tasks, split: {args.split})...")
        train_dataset = JIGSAWSMultiTaskDataset(
            data_root=args.data_root,
            split_name=args.split,
            mode='train',
            arm=args.arm,
            split_family=args.split_family,
        )
        val_dataset = JIGSAWSMultiTaskDataset(
            data_root=args.data_root,
            split_name=args.split,
            mode='val',
            arm=args.arm,
            split_family=args.split_family,
        )
        print(f"  - Train samples: {len(train_dataset)}")
        print(f"  - Val samples: {len(val_dataset)}")

        batch_size = config.get('batch_size', 16)
        # Force balanced-task sampling for any brain_mode so every batch contains all
        # three JIGSAWS tasks. For subskill grouping this is what gives enough family
        # coverage per batch to avoid empty-centroid zero-fills in compute_centroid_rdm.
        use_balanced_tasks = config.get('brain_mode') in ('eye', 'bridge')
        if use_balanced_tasks:
            task_labels = train_dataset.task_labels
            batch_sampler = BalancedTaskBatchSampler(
                task_labels=task_labels,
                batch_size=batch_size,
                drop_last=False,
                shuffle=True
            )
            train_dataloader = DataLoader(
                train_dataset,
                batch_sampler=batch_sampler,
                num_workers=0,
                collate_fn=pad_collate_fn
            )
        else:
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                collate_fn=pad_collate_fn
            )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=pad_collate_fn
        )
    else:
        print(f"\nLoading dataset (task: {args.task})...")
        full_dataset = JIGSAWSViTDataset(
            data_root=args.data_root,
            task=args.task,
            mode='train',
            arm=args.arm
        )
        print(f"  - Full dataset size: {len(full_dataset)} samples")

        # Apply split if specified
        val_dataloader = None
        if args.split is not None:
            print(f"\nApplying split: {args.split} (family={args.split_family})")
            split_loader = SplitLoader(
                args.data_root, args.task, args.split, split_family=args.split_family
            )

            train_trials = split_loader.get_train_trials()
            val_trials = split_loader.get_val_trials()
            train_seg_filter = split_loader.get_segment_filter('train')
            val_seg_filter = split_loader.get_segment_filter('val')

            print(f"  - Train trials ({len(train_trials)}): {train_trials}")
            print(f"  - Val trials ({len(val_trials)}): {val_trials}")
            if train_seg_filter:
                print(f"  - Train segment_filter: {len(train_seg_filter)} trials restricted")
            if val_seg_filter:
                print(f"  - Val segment_filter: {len(val_seg_filter)} trials restricted")

            train_dataset = filter_dataset_by_trials(
                full_dataset, train_trials, segment_filter=train_seg_filter
            )
            val_dataset = filter_dataset_by_trials(
                full_dataset, val_trials, segment_filter=val_seg_filter
            )

            print(f"  - Train samples: {len(train_dataset)}")
            print(f"  - Val samples: {len(val_dataset)}")

            val_dataloader = DataLoader(
                val_dataset,
                batch_size=config.get('batch_size', 16),
                shuffle=False,
                num_workers=0,
                collate_fn=pad_collate_fn
            )
        else:
            print("\n  WARNING: No split specified. Training on ALL data (no validation).")
            print("  Consider using --split fold_1 for proper train/val separation.")
            train_dataset = full_dataset

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', 16),
            shuffle=True,
            num_workers=0,  # Set to 0 for MPS compatibility
            collate_fn=pad_collate_fn
        )
    print(f"  - Batch size: {config.get('batch_size', 16)}")
    print(f"  - Number of training batches per epoch: {len(train_dataloader)}")
    
    # Create optimizer and scheduler
    print("\nSetting up optimizer and scheduler...")
    lr_base = float(config.get('lr_base', 1e-4))
    lr_vit = float(config.get('lr_vit', 1e-5))
    
    optimizer = get_optimizer(
        model,
        lr_base=lr_base,
        lr_vit=lr_vit,
        use_adapters=config.get('use_adapters', False)
    )
    print(f"  - Optimizer: Adam (lr_base={lr_base}, lr_vit={lr_vit})")
    
    num_epochs = config.get('num_epochs', 80)
    scheduler = get_scheduler(
        optimizer,
        num_epochs,
        warmup_epochs=config.get('warmup_epochs', 5)
    )
    print(f"  - Scheduler: CosineAnnealingLR with {config.get('warmup_epochs', 5)} warmup epochs")
    
    # Training loop
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    print(f"Checkpoints will be saved every {config.get('save_every', 10)} epochs")
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    # Track best validation loss for model selection
    best_val_loss = float('inf')
    best_epoch = -1

    # Main training loop with progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)

    for epoch in epoch_pbar:
        # Training
        train_losses = train_epoch(
            model, train_dataloader, optimizer, device, config, epoch,
            target_rdm=target_rdm
        )
        scheduler.step()

        # Validation (if split is specified)
        val_losses = None
        if val_dataloader is not None:
            val_losses = validate_epoch(model, val_dataloader, device, config)

        # Update epoch progress bar
        postfix = {
            'train': f"{train_losses['total']:.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        }
        if val_losses is not None:
            postfix['val'] = f"{val_losses['total']:.4f}"
        epoch_pbar.set_postfix(postfix)

        # Print detailed losses every epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  [TRAIN]")
        print(f"    Total Loss: {train_losses['total']:.6f}")
        print(f"    Kinematics Loss: {train_losses.get('kin', 0):.6f}")
        if 'kin_pos' in train_losses:
            print(f"      - Position: {train_losses['kin_pos']:.6f}")
        if 'kin_rot' in train_losses:
            print(f"      - Rotation: {train_losses['kin_rot']:.6f} ({train_losses.get('kin_rot_deg', 0):.2f}°)")
        if 'kin_jaw' in train_losses:
            print(f"      - Jaw: {train_losses['kin_jaw']:.6f}")
        print(f"    Gesture Loss: {train_losses.get('gesture', 0):.6f}")
        print(f"    Skill Loss: {train_losses.get('skill', 0):.6f}")
        if 'brain_rsa' in train_losses:
            rsa_corr = 1.0 - train_losses['brain_rsa']
            print(f"    Brain RSA Loss: {train_losses['brain_rsa']:.6f} (corr: {rsa_corr:.4f})")

        if val_losses is not None:
            print(f"  [VAL]")
            print(f"    Total Loss: {val_losses['total']:.6f}")
            print(f"    Kinematics Loss: {val_losses.get('kin', 0):.6f}")
            if 'kin_pos' in val_losses:
                print(f"      - Position: {val_losses['kin_pos']:.6f}")
            if 'kin_rot' in val_losses:
                print(f"      - Rotation: {val_losses['kin_rot']:.6f} ({val_losses.get('kin_rot_deg', 0):.2f}°)")
            if 'kin_jaw' in val_losses:
                print(f"      - Jaw: {val_losses['kin_jaw']:.6f}")
            print(f"    Gesture Loss: {val_losses.get('gesture', 0):.6f}")
            print(f"    Skill Loss: {val_losses.get('skill', 0):.6f}")

            # Track best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                best_epoch = epoch + 1
                # Save best model
                best_path = output_dir / 'best_model.pth'
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config,
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'split': args.split,
                    'task': args.task
                }
                torch.save(checkpoint, best_path)
                print(f"  * New best model saved (val_loss: {best_val_loss:.6f})")

        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

        # Save periodic checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'split': args.split,
                'task': args.task
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_path = output_dir / 'final_model.pth'
    checkpoint = {
        'epoch': num_epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'split': args.split,
        'task': args.task
    }
    torch.save(checkpoint, final_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Split: {args.split if args.split else 'None (all data)'}")
    print(f"Final model saved to: {final_path}")
    if best_epoch > 0:
        print(f"Best model (epoch {best_epoch}, val_loss: {best_val_loss:.6f}): {output_dir / 'best_model.pth'}")


if __name__ == '__main__':
    main()

