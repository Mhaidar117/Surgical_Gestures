# ViT-Based Surgical Gesture Recognition and Kinematics Prediction for dVRK

This repository implements a Vision Transformer (ViT) system for surgical gesture recognition and kinematics prediction from video. The system processes surgical video frames to predict robot kinematics that can be executed safely on a da Vinci Research Kit (dVRK) single-arm robot. To use please add a Gestures folder with each subdirectory containing the gesture folder from JIGSAW. 

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Data Format](#data-format)
- [Module Descriptions](#module-descriptions)
- [Installation](#installation)
- [Training Instructions](#training-instructions)
- [Evaluation Instructions](#evaluation-instructions)
- [dVRK Integration](#dvrk-integration)
- [Usage Examples](#usage-examples)

## Overview

The system consists of three main components:

1. **Visual Module**: ViT-S/16 encoder that processes RGB frames and optionally optical flow
2. **Kinematics Module**: Transformer decoder that predicts kinematics from visual embeddings, with gesture and skill classification heads
3. **Safety Module**: Validation and filtering for safe execution on dVRK

### Pipeline

```
Video frames → ViT encoder → per-frame embeddings → Transformer decoder → per-timestep kinematics (k̂ₜ)
                                                          ↳ gesture head (frame/segment)
                                                          ↳ skill head (trial)
                                                          ↓
                                                    Safety filters → dVRK execution
```

The system predicts:
- **Kinematics**: Position (3D), rotation (6D representation), and gripper angle per frame
- **Gestures**: Per-frame gesture classification (15 classes)
- **Skill**: Trial-level skill classification (Novice, Intermediate, Expert)

## Project Structure

```
Surgical_Gestures/
├── src/
│   ├── data/                    # Data loading and preprocessing
│   │   ├── jigsaws_vit_dataset.py    # Unified dataset class
│   │   ├── transforms_vit.py         # ViT preprocessing transforms
│   │   ├── sync_manager.py           # Temporal alignment utilities
│   │   └── split_loader.py           # LOUO split management
│   ├── models/                  # Model architectures
│   │   ├── visual.py                 # ViT encoder
│   │   ├── adapters.py               # Adapter layers for personalization
│   │   ├── temporal_transformer.py   # Temporal aggregation
│   │   ├── decoder_autoreg.py        # Autoregressive decoder
│   │   ├── kinematics.py              # Kinematics module with heads
│   │   └── losses.py                 # Loss functions
│   ├── training/                # Training infrastructure
│   │   ├── train_vit_system.py       # Main training script
│   │   └── optim.py                  # Optimizer/scheduler setup
│   ├── eval/                    # Evaluation utilities
│   │   ├── metrics.py                # Evaluation metrics
│   │   └── postprocess.py            # Trajectory smoothing
│   ├── safety/                  # Safety modules
│   │   ├── filters.py                # Safety validation
│   │   └── dvrk_interface.py         # dVRK simulator interface
│   ├── inference/               # Inference pipeline
│   │   └── predict.py                # Prediction utilities
│   └── configs/                 # Configuration files
│       └── baseline.yaml
├── scripts/                     # Preprocessing scripts
│   └── precompute_raft.py           # RAFT flow computation (optional)
├── Gestures/                    # JIGSAWS dataset
│   ├── Knot_Tying/
│   ├── Needle_Passing/
│   └── Suturing/
└── gesture_embedding/           # Legacy code (preserved)
```

## Data Format

### JIGSAWS Dataset Structure

The JIGSAWS dataset contains surgical gesture data organized by task:

```
Gestures/
├── Knot_Tying/
│   ├── video/                    # Video files (.avi)
│   │   ├── Knot_Tying_B001_capture1.avi
│   │   └── ...
│   ├── kinematics/
│   │   └── AllGestures/          # Kinematics files (.txt)
│   │       ├── Knot_Tying_B001.txt
│   │       └── ...
│   ├── transcriptions/            # Gesture annotations (.txt)
│   │   ├── Knot_Tying_B001.txt   # Format: start_frame end_frame gesture_id
│   │   └── ...
│   └── meta_file_Knot_Tying.txt  # Metadata: trial_id, skill_level, GRS scores
```

#### Video Files
- Format: `.avi` files at 30 Hz
- Naming: `{Task}_{TrialID}_capture{1|2}.avi`
- `capture1`: Left camera view
- `capture2`: Right camera view

#### Kinematics Files
- Format: `.txt` files with 76-dimensional vectors per line
- Structure:
  - Columns 1-3: Master left tooltip xyz
  - Columns 4-12: Master left tooltip rotation (9D)
  - Columns 13-15: Master left tooltip translational velocity
  - Columns 16-18: Master left tooltip rotational velocity
  - Column 19: Master left gripper angle
  - Columns 20-38: Master right (same structure)
  - Columns 39-57: Slave left (same structure)
  - Columns 58-76: Slave right (same structure)
- Sampling rate: 30 Hz (matches video)

#### Transcription Files
- Format: `.txt` files with gesture annotations
- Each line: `start_frame end_frame gesture_id`
- Gesture IDs for Knot_Tying: G1, G11, G12, G13, G14, G15
- Gesture IDs for Needle_Passing: G1, G2, G3, G4, G5, G6, G8, G11
- Gesture IDs for Suturing: G1, G2, G3, G4, G5, G6, G8, G9, G10, G11

#### Metadata Files
- Format: Tab-separated values
- Columns:
  1. Trial ID (e.g., `Knot_Tying_B001`)
  2. Skill level (E=Expert, I=Intermediate, N=Novice)
  3. GRS score
  4-9. Individual GRS element scores

## Module Descriptions

### Data Module (`src/data/`)

#### `jigsaws_vit_dataset.py`
Unified dataset class that handles:
- Loading RGB frames from video files
- Loading kinematics from text files
- Temporal alignment between modalities
- Backward compatibility with legacy blob format
- Frame sampling for different tasks (gesture: 10 frames, skill: 30 frames, kinematics: 25 frames)

**Key Features:**
- Supports both new format (RGB + RAFT + metadata) and legacy format (Farnebäck flow)
- Automatic temporal resampling for kinematics alignment
- Configurable window sizes and strides

#### `transforms_vit.py`
Preprocessing transforms for ViT:
- `VideoToTensor`: Convert numpy arrays to PyTorch tensors
- `ResizeAndCenterCrop`: Resize to 256 then center crop to 224×224
- `Normalize`: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- `TemporalSampler`: Sample temporal windows with configurable stride
- `TemporalAugmentation`: Frame dropping and temporal jitter for training

#### `sync_manager.py`
Temporal synchronization utilities:
- Frame-to-time conversions
- Kinematics resampling to match video frames
- Timestamp mapping across modalities

#### `split_loader.py`
LOUO (Leave-One-User-Out) split management:
- `generate_louo_splits()`: Generate 8-fold splits based on surgeon IDs
- `load_splits()`: Load split definitions from JSON/YAML
- `SplitLoader`: Convenience class for accessing splits

### Model Module (`src/models/`)

#### `visual.py`
ViT encoder for frame processing:
- `ViTFrameEncoder`: Main encoder class
  - Uses `timm` ViT-S/16 (384-dim, 6 heads, 12 blocks)
  - ImageNet-21k pretrained weights
  - Adapter support for personalization
  - Feature caching to avoid redundant forward passes
- `ViTFlowEncoder`: Optional flow encoder (2-channel input)
  - Initialized from RGB encoder via channel averaging

**Key Parameters:**
- `freeze_until`: Freeze early layers (default: 6)
- `use_adapters`: Enable adapter layers (default: False)
- `adapter_dim`: Adapter bottleneck dimension (default: 64)

#### `adapters.py`
Lightweight adapter layers for per-subject personalization:
- `AdapterLayer`: Bottleneck adapter (down-project -> activation -> up-project)
- `ViTBlockWithAdapter`: ViT block wrapper with adapters
- `insert_adapters_into_vit()`: Insert adapters into existing ViT
- `freeze_vit_except_adapters()`: Freeze base model, train only adapters

**Architecture:**
- Down-projection: `d_model -> adapter_dim` (default 64)
- Activation: GELU
- Up-projection: `adapter_dim -> d_model`
- Residual connection handled externally

#### `temporal_transformer.py`
Temporal aggregation over frame embeddings:
- `TemporalAggregator`: 4-layer transformer encoder
  - Learned positional encodings
  - Variable-length sequence support via attention masks
  - Mean pooling and attention pooling
- `AttentionPooling`: Learned attention-based pooling
- `TemporalAggregatorWithPooling`: Combined aggregator with multiple pooling options

**Architecture:**
- 4 transformer encoder layers
- `d_model=384`, `n_heads=6`, `dim_feedforward=1536`
- Max sequence length: 64 (interpolated for longer sequences)

#### `decoder_autoreg.py`
Autoregressive transformer decoder for kinematics:
- `CrossAttnDecoderLayer`: Decoder layer with self-attention and cross-attention
- `KinematicsDecoder`: 6-layer causal decoder
  - Self-attention with causal mask
  - Cross-attention to encoder memory
  - Learned start token
  - Positional encodings

**Architecture:**
- 6 decoder layers
- `d_model=384`, `n_heads=6`, `dim_feedforward=1536`
- Output dimension: 10 (pos3 + rot6D + jaw1) or 16 (with velocities)

#### `kinematics.py`
Complete kinematics module:
- `KinematicsModule`: Main module combining decoder and heads
  - Deterministic projection: MLP from embeddings to latent
  - Autoregressive decoder
  - Gesture head: Temporal attention pooling -> classification
  - Skill head: 2-layer MLP for trial-level classification
- `GestureHead`: Per-frame gesture classification (15 classes)
- `SkillHead`: Trial-level skill classification (3 classes: Novice, Intermediate, Expert)

**Output Format:**
- Kinematics: `(B, T, 10)` - [pos(3), rot6D(6), jaw(1)]
- Gesture logits: `(B, 15)`
- Skill logits: `(B, 3)`

#### `losses.py`
Comprehensive loss functions:
- `kinematics_loss()`: Position (SmoothL1), rotation (geodesic), gripper (MSE), jerk penalty
- `control_regularizer()`: Velocity/acceleration/joint limit penalties
- `compute_total_loss()`: Combined loss with configurable weights

**Loss Weights (default):**
- Kinematics: 1.0
- Gesture: 1.0
- Skill: 0.5
- Control regularizer: 0.01

### Training Module (`src/training/`)

#### `train_vit_system.py`
Main training script:
- `EEGInformedViTModel`: Unified model combining all components
- `train_epoch()`: Training loop with:
  - Teacher forcing with scheduled sampling
  - Gradient clipping
  - Loss accumulation and logging
  - Progress bars with tqdm

**Training Stages:**
1. **Stage 0**: Offline preprocessing (RAFT flow, optional)
2. **Stage 1**: Baseline training (40-80 epochs)
3. **Stage 2**: Fine-tuning (optional, 10-30 epochs)

#### `optim.py`
Optimizer and scheduler setup:
- `get_optimizer()`: AdamW with different LRs for:
  - ViT backbone: 1e-5
  - Adapters: 5e-5
  - Decoder/heads: 1e-4
- `get_scheduler()`: Cosine annealing with warmup

### Evaluation Module (`src/eval/`)

#### `metrics.py`
Evaluation metrics:
- `compute_kinematics_metrics()`: RMSE, end-effector error, rotation error
- `compute_gesture_metrics()`: Macro/micro F1, accuracy
- `compute_skill_metrics()`: Macro F1, accuracy

#### `postprocess.py`
Trajectory post-processing:
- `postprocess_kinematics()`: Complete pipeline
- `smooth_trajectory()`: Savitzky-Golay filter (window=5, order=2)
- `clip_velocities()`: Enforce velocity limits
- `project_rotation_svd()`: Project 6D rotation to valid SO(3) matrix

### Safety Module (`src/safety/`)

#### `filters.py`
Safety validation:
- `validate_trajectory()`: Complete safety check
- `check_workspace_bounds()`: Verify position within workspace
- `check_velocity_limits()`: Verify velocity constraints
- `check_acceleration_limits()`: Verify acceleration constraints

#### `dvrk_interface.py`
dVRK simulator interface:
- `DVRKInterface`: Interface class with safety checks
- `execute_trajectory()`: Execute with validation
- `check_torque()`: Verify torque limits

## Installation

### Requirements

```bash
# Core dependencies
pip install torch torchvision
pip install timm  # Vision Transformers
pip install numpy scipy
pip install scikit-learn
pip install pyyaml
pip install tqdm  # Progress bars
pip install opencv-python  # Video processing

# Optional: RAFT for optical flow
# Follow instructions at: https://github.com/princeton-vl/RAFT
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Surgical_Gestures
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download JIGSAWS dataset and place in `Gestures/` directory

4. (Optional) Download RAFT weights for optical flow preprocessing

## Training Instructions

### Stage 0: Preprocessing (Optional)

#### Precompute RAFT Optical Flow (Optional)

```bash
python scripts/precompute_raft.py \
    --video_dir Gestures/Knot_Tying/video \
    --output_dir cache/raft_flows \
    --raft_weights path/to/raft/weights.pth \
    --device mps
```

#### Generate LOUO Splits

```python
from src.data.split_loader import generate_louo_splits

generate_louo_splits(
    data_root='.',
    task='Knot_Tying',
    output_dir='data/splits'
)
```

### Stage 1: Baseline Training

Train the baseline model:

```bash
python gesture_embedding/main.py \
    --mode vit_train \
    --source_directory . \
    --transcriptions_path Knot_Tying \
    --weights_save_path src/configs/baseline.yaml \
    --weights_save_folder checkpoints/baseline
```

Or directly:

```bash
python src/training/train_vit_system.py \
    --config src/configs/baseline.yaml \
    --data_root . \
    --task Knot_Tying \
    --output_dir checkpoints/baseline
```

**Configuration (`baseline.yaml`):**
- `num_epochs: 80` - Training epochs
- `batch_size: 16` - Batch size
- `lr_base: 1e-4` - Learning rate for decoder/heads
- `lr_vit: 1e-5` - Learning rate for ViT
- `freeze_until: 6` - Freeze first 6 ViT blocks

**Expected Output:**
- Checkpoints saved every 10 epochs in `checkpoints/baseline/`
- Final checkpoint: `checkpoint_epoch_80.pth`
- Training progress displayed with tqdm progress bars

**Training Multiple Seeds:**

For reproducibility and statistical power, train with multiple seeds:

```bash
for seed in 42 123 456 789 999; do
    python src/training/train_vit_system.py \
        --config src/configs/baseline.yaml \
        --data_root . \
        --task Knot_Tying \
        --output_dir checkpoints/baseline_seed${seed} \
        --seed ${seed}
done
```

### Training Tips

1. **Monitor Training:**
   - Watch for kinematic RMSE decreasing
   - Gesture F1 should increase
   - Skill classification accuracy should improve
   - Use tqdm progress bars to track training progress

2. **Hyperparameter Tuning:**
   - Adjust learning rates if training is unstable
   - Increase batch size if memory allows
   - Adjust `freeze_until` to control ViT fine-tuning

3. **Memory Management:**
   - Reduce `batch_size` if OOM errors occur
   - Use gradient accumulation for effective larger batches
   - Consider using CPU if MPS/CUDA issues occur

## Evaluation Instructions

### Single Model Evaluation

Evaluate a trained model on test set:

```bash
python gesture_embedding/main.py \
    --mode vit_eval \
    --source_directory . \
    --transcriptions_path Knot_Tying \
    --weights_save_path checkpoints/baseline/checkpoint_epoch_80.pth
```

Or directly:

```python
from src.training.train_vit_system import EEGInformedViTModel
from src.eval.metrics import *
from src.data import JIGSAWSViTDataset
from torch.utils.data import DataLoader
import torch

# Load model
checkpoint = torch.load('checkpoints/baseline/checkpoint_epoch_80.pth')
model = EEGInformedViTModel(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Create test dataset
dataset = JIGSAWSViTDataset(
    data_root='.',
    task='Knot_Tying',
    split='fold_1',  # Or use test split
    mode='val'
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Evaluate
all_metrics = {}
with torch.no_grad():
    for batch in dataloader:
        outputs = model(batch['rgb'])
        # Compute metrics...
```

### LOUO Cross-Validation

Evaluate across all LOUO folds:

```python
from src.data.split_loader import SplitLoader
from src.eval.metrics import *
import numpy as np

split_loader = SplitLoader(data_root='.', task='Knot_Tying')
all_folds_metrics = []

for fold_name in split_loader.get_all_folds():
    split_loader.split_name = fold_name
    
    # Load model
    checkpoint = torch.load(f'checkpoints/baseline/checkpoint_epoch_80.pth')
    model = EEGInformedViTModel(checkpoint['config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset for this fold
    test_trials = split_loader.get_test_trials()
    dataset = JIGSAWSViTDataset(
        data_root='.',
        task='Knot_Tying',
        split=fold_name,
        mode='val'
    )
    
    # Evaluate and accumulate metrics
    fold_metrics = evaluate_fold(model, dataset)
    all_folds_metrics.append(fold_metrics)

# Average across folds
avg_metrics = {k: np.mean([m[k] for m in all_folds_metrics]) 
               for k in all_folds_metrics[0].keys()}
print("LOUO Average Metrics:", avg_metrics)
```

### Metrics Reported

- **Kinematics:**
  - Position RMSE (mm)
  - End-effector error (mm)
  - Rotation RMSE
  - Rotation geodesic error (degrees)

- **Gesture Classification:**
  - Macro F1 score
  - Micro F1 score
  - Accuracy
  - Per-gesture precision/recall

- **Skill Classification:**
  - Macro F1 score
  - Accuracy

## dVRK Integration

### Safety Validation

Before executing on dVRK, all trajectories must pass safety checks:

```python
from src.safety.filters import validate_trajectory

# Validate trajectory
is_safe, message = validate_trajectory(
    kinematics,
    workspace_bounds={'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.3, 0.3)},
    max_velocity=0.1,
    max_acceleration=0.5
)

if not is_safe:
    print(f"Trajectory unsafe: {message}")
    # Apply corrections or reject trajectory
```

### dVRK Execution

Execute validated trajectories on dVRK:

```python
from src.safety.dvrk_interface import DVRKInterface
from src.eval.postprocess import postprocess_kinematics

# Post-process kinematics for smoothness
kinematics_smooth = postprocess_kinematics(kinematics)

# Validate
is_safe, message = validate_trajectory(kinematics_smooth, ...)

if is_safe:
    # Execute on dVRK
    dvrk = DVRKInterface()
    dvrk.connect()
    success, msg = dvrk.execute_trajectory(kinematics_smooth)
    dvrk.disconnect()
    
    if success:
        print("Trajectory executed successfully")
    else:
        print(f"Execution failed: {msg}")
```

### Safety Features

The safety module enforces:
- **Workspace bounds**: Position limits to prevent collisions
- **Velocity limits**: Maximum velocity constraints
- **Acceleration limits**: Maximum acceleration constraints
- **Torque limits**: Joint torque constraints
- **Collision detection**: Basic collision checking

## Usage Examples

### Basic Training

```python
from src.training.train_vit_system import EEGInformedViTModel, train_epoch
from src.data import JIGSAWSViTDataset
from torch.utils.data import DataLoader
import yaml
import torch

# Load config
with open('src/configs/baseline.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create model
model = EEGInformedViTModel(config)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = model.to(device)

# Create dataset
dataset = JIGSAWSViTDataset(
    data_root='.',
    task='Knot_Tying',
    mode='train'
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train
from src.training.optim import get_optimizer, get_scheduler
optimizer = get_optimizer(model, lr_base=1e-4, lr_vit=1e-5)
scheduler = get_scheduler(optimizer, num_epochs=80, warmup_epochs=5)

for epoch in range(80):
    losses = train_epoch(model, dataloader, optimizer, device, config, epoch)
    scheduler.step()
    print(f"Epoch {epoch}: {losses}")
```

### Inference

```python
from src.inference.predict import predict_kinematics
import torch
import cv2
import numpy as np

# Load model
checkpoint = torch.load('checkpoints/baseline/checkpoint_epoch_80.pth')
model = EEGInformedViTModel(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load video
cap = cv2.VideoCapture('test_video.avi')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# Predict
frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0
kinematics = predict_kinematics(model, frames_tensor, postprocess=True)

print(f"Predicted kinematics shape: {kinematics.shape}")
```

### Complete dVRK Pipeline

```python
from src.inference.predict import predict_kinematics
from src.eval.postprocess import postprocess_kinematics
from src.safety.filters import validate_trajectory
from src.safety.dvrk_interface import DVRKInterface
import torch
import cv2
import numpy as np

# 1. Load model
checkpoint = torch.load('checkpoints/baseline/checkpoint_epoch_80.pth')
model = EEGInformedViTModel(checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 2. Load and process video
cap = cv2.VideoCapture('surgical_video.avi')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float() / 255.0

# 3. Predict kinematics
with torch.no_grad():
    kinematics = predict_kinematics(model, frames_tensor, postprocess=True)

# 4. Post-process for smoothness
kinematics_smooth = postprocess_kinematics(kinematics)

# 5. Validate safety
is_safe, message = validate_trajectory(
    kinematics_smooth,
    workspace_bounds={'x': (-0.5, 0.5), 'y': (-0.5, 0.5), 'z': (-0.3, 0.3)},
    max_velocity=0.1,
    max_acceleration=0.5
)

# 6. Execute on dVRK if safe
if is_safe:
    dvrk = DVRKInterface()
    dvrk.connect()
    success, msg = dvrk.execute_trajectory(kinematics_smooth)
    dvrk.disconnect()
    
    if success:
        print("Trajectory executed successfully on dVRK")
    else:
        print(f"Execution failed: {msg}")
else:
    print(f"Trajectory rejected: {message}")
```

## Citation

If you use this code, please cite:

```bibtex
@article{jigsaws,
  title={JIGSAWS: A Dataset for Surgical Skill Assessment},
  author={Gao, Yixin and Vedula, S Swaroop and Reiley, Carol E and Ahmidi, Narges and Varadarajan, Balakrishnan and Lin, Henry C and Tao, Lingling and Zappella, Luca and Béjar, Benjamin and Yuh, David D and others},
  journal={arXiv preprint arXiv:1506.04112},
  year={2015}
}
```

## License

See LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub.
