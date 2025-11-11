"""
Main training script for ViT-based system with config-driven loop.
"""
import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
from typing import Dict, Optional
import sys
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.visual import ViTFrameEncoder
from models.temporal_transformer import TemporalAggregatorWithPooling
from models.kinematics import KinematicsModule
from models.losses import compute_total_loss
from modules.brain_rdm import BrainRDM, compute_model_rdm, sample_rdm_batch
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
from data import JIGSAWSViTDataset
from torch.utils.data import DataLoader


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
        self.kinematics = KinematicsModule(
            d_model=384,
            d_kin_input=76,
            d_kin_output=config.get('d_kin_output', 10),
            num_gestures=15,
            num_skills=3
        )
        print(f"  - d_kin_input: 76, d_kin_output: {config.get('d_kin_output', 10)}")
        print(f"  - num_gestures: 15, num_skills: 3")
        
        # Brain RDM (training only)
        self.brain_mode = config.get('brain_mode', 'none')
        if self.brain_mode != 'none':
            print(f"Creating Brain RDM module (mode: {self.brain_mode})...")
            self.brain_rdm = BrainRDM(
                cache_dir=config.get('eeg_rdm_cache_dir'),
                tau_range=config.get('tau_range', [0, 50, 100, 150, 200, 250, 300])
            )
        else:
            print("Brain RDM: disabled")
        
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
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    component_losses = {}
    num_batches = 0
    
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
        
        # Forward pass
        teacher_forcing_prob = max(0.3, 1.0 - (epoch / config.get('teacher_forcing_decay_epochs', 40)))
        
        outputs = model(
            rgb,
            target_kinematics=kinematics,
            teacher_forcing_prob=teacher_forcing_prob
        )
        
        # Compute brain alignment if needed
        model_rdm = None
        eeg_rdm = None
        model_features = None
        eeg_patterns = None
        
        if config.get('brain_mode') != 'none' and 'embeddings' in outputs:
            # Sample features for RDM
            features = outputs['embeddings'].get('mid', outputs['memory'])
            if len(features.shape) == 3:
                features = features.view(-1, features.shape[-1])
            
            sampled_features, _ = sample_rdm_batch(features, batch_size=config.get('rdm_batch_size', 32))
            model_rdm = compute_model_rdm(sampled_features, method='pearson')
            
            # Get EEG RDM
            batch_meta = {
                'trial_ids': batch.get('trial_id', [])
            }
            eeg_rdm = model.brain_rdm.get_eeg_rdm(batch_meta, tau=config.get('tau', 0))
        
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
            loss_weights=config.get('loss_weights')
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


def main():
    parser = argparse.ArgumentParser(description='Train ViT system')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--data_root', type=str, required=True, help='Data root directory')
    parser.add_argument('--task', type=str, default='Knot_Tying', help='Task name')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Output directory')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("ViT Training System")
    print("=" * 60)
    print(f"Config file: {args.config}")
    print(f"Data root: {args.data_root}")
    print(f"Task: {args.task}")
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
    
    # Create dataset
    print(f"\nLoading dataset (task: {args.task})...")
    dataset = JIGSAWSViTDataset(
        data_root=args.data_root,
        task=args.task,
        mode='train'
    )
    print(f"  - Dataset size: {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config.get('batch_size', 16), 
        shuffle=True,
        num_workers=0  # Set to 0 for MPS compatibility
    )
    print(f"  - Batch size: {config.get('batch_size', 16)}")
    print(f"  - Number of batches per epoch: {len(dataloader)}")
    
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
    
    # Main training loop with progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress", position=0)
    
    for epoch in epoch_pbar:
        losses = train_epoch(model, dataloader, optimizer, device, config, epoch)
        scheduler.step()
        
        # Update epoch progress bar
        epoch_pbar.set_postfix({
            'loss': f"{losses['total']:.4f}",
            'kin': f"{losses.get('kin', 0):.4f}",
            'gest': f"{losses.get('gesture', 0):.4f}",
            'lr': f"{scheduler.get_last_lr()[0]:.6f}"
        })
        
        # Print detailed losses every epoch
        print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Total Loss: {losses['total']:.6f}")
        print(f"  Kinematics Loss: {losses.get('kin', 0):.6f}")
        if 'kin_pos' in losses:
            print(f"    - Position: {losses['kin_pos']:.6f}")
        if 'kin_rot' in losses:
            print(f"    - Rotation: {losses['kin_rot']:.6f} ({losses.get('kin_rot_deg', 0):.2f}°)")
        if 'kin_jaw' in losses:
            print(f"    - Jaw: {losses['kin_jaw']:.6f}")
        print(f"  Gesture Loss: {losses.get('gesture', 0):.6f}")
        print(f"  Skill Loss: {losses.get('skill', 0):.6f}")
        if 'brain_rsa' in losses:
            print(f"  Brain RSA Loss: {losses['brain_rsa']:.6f}")
        if 'brain_encoding' in losses:
            print(f"  Brain Encoding Loss: {losses['brain_encoding']:.6f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 10) == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'losses': losses
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final checkpoint saved to: {output_dir}")


if __name__ == '__main__':
    main()

