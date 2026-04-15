# CLAUDE.md — Project Context for Surgical Gestures

**Repository docs:** Only this file and [`README.md`](README.md) are versioned as project documentation. Optional long-form notes (embedding workflows, legacy README text, Phase 1 schema detail) may live in a local `docs/` directory, which is gitignored.

## Project Overview

ViT-based surgical gesture recognition and kinematics prediction for the da Vinci Research Kit (dVRK), with a multi-phase **EEG–Eye Bridge** pipeline that derives neuroscientific regularizers from EEG and eye-tracking data. The model watches surgical video and simultaneously predicts robot kinematics, gesture labels, and surgeon skill level.

**Dataset:** JIGSAWS (JHU-ISI Gesture and Skill Assessment Working Set) — three tasks: Suturing, Needle Passing, Knot Tying. Video at 30 Hz, 76-dim kinematics (4 arms × 19D), 15 gesture classes, 3 skill levels (E/I/N), 8 surgeons.

**Evaluation protocol:** 8-fold Leave-One-User-Out (LOUO) cross-validation.

## Quick Reference — How to Run Things

```bash
# Always set PYTHONPATH first
export PYTHONPATH=src

# Generate cross-validation splits
python generate_splits.py

# Full EEG–Eye bridge pipeline (Phase 1 → 2 → 3 → optional Phase 4 training)
python scripts/eeg_eye_bridge/run_full_pipeline.py
python scripts/eeg_eye_bridge/run_full_pipeline.py --phase1-synthetic --skip-train  # smoke test

# 8-fold LOUO baseline training
./run_8fold_louo.sh                          # all tasks, baseline config
./run_8fold_louo.sh Suturing 1 4             # single task, folds 1-4
./run_8fold_louo.sh all 1 8 brain_eye.yaml   # custom config

# 8-fold LOUO with brain alignment
./run_8fold_louo_brain.sh

# Run all EEG–eye bridge tests
python tests/eeg_eye_bridge/phase1/test_phase1_eeg.py
python -m pytest tests/eeg_eye_bridge/phase2/test_phase2_eye_consistency.py -v
python -m pytest tests/eeg_eye_bridge/phase3/test_phase3_rdms.py -v
python tests/eeg_eye_bridge/phase4/test_phase4_vit_regularizer.py
python tests/eeg_eye_bridge/test_phase5_integration_coordinator.py
```

**Windows (PowerShell)** — same commands with `python` and `$env:PYTHONPATH = "src"` instead of `export`. Use Git Bash or WSL for `./run_8fold_louo.sh` / `./run_8fold_louo_brain.sh`, or invoke `python`/`bash` explicitly.

**Evaluation reports for LOUO aggregation:** [`src/eval/evaluate.py`](src/eval/evaluate.py) accepts optional `--output_dir eval_results` together with `--split fold_N` to write `<Task>_test_fold_<n>_results.txt` files compatible with [`aggregate_louo_results.py`](aggregate_louo_results.py). [`run_8fold_louo.sh`](run_8fold_louo.sh) passes `--output_dir` after each fold’s training.

## Architecture

### Model: `EEGInformedViTModel` (src/training/train_vit_system.py)

```
Video Frames
    │
    ▼
ViTFrameEncoder (ViT-S/16, 384-dim, pretrained ImageNet, adapters, freeze_until=6)
    │
    ▼
TemporalAggregatorWithPooling (4-layer transformer, d_model=384, 6 heads)
    │
    ├──▶ KinematicsModule → (B, T, 10): pos(3) + rot6D(6) + jaw(1)
    ├──▶ Gesture head → (B, 15)
    ├──▶ Skill head → (B, 3)
    └──▶ BrainRDM module → RSA loss (optional, weight=0.01)
```

### Loss Function (src/models/losses.py → `compute_total_loss`)

```
L = w_kin * L_kin + w_gesture * L_gesture + w_skill * L_skill + w_brain * L_brain + w_control * L_control
```

- **L_kin:** SmoothL1 (position) + geodesic SO(3) distance (rotation) + velocity + gripper
- **L_gesture / L_skill:** Cross-entropy
- **L_brain:** RSA loss = 1 − Pearson(flatten(model_RDM), flatten(target_RDM))
- **L_control:** Velocity/acceleration/joint limit regularizer for dVRK safety

### Brain Modes (set via `brain_mode` in YAML config)

| Mode | Config key | What it does |
|------|-----------|--------------|
| `none` | baseline.yaml | No brain alignment (Stage 1 warmup) |
| `eye` | brain_eye.yaml | Fixed 3×3 target RDM from eye-tracking |
| `bridge` | bridge_eeg_rdm.yaml | Phase 3 EEG/eye RDM from manifest |
| `rsa` | rsa.yaml | Direct EEG RDM with tau-lag alignment |
| `encoding` | encoding.yaml | Ridge regression encoding loss |

## EEG–Eye Bridge Pipeline (4 Phases)

### Phase 1: EEG Export
- **Script:** `scripts/eeg_eye_bridge/phase1/run_export.py`
- **Source:** `src/eeg_eye_bridge/phase1_eeg/`
- **Input:** Raw EDF files (or synthetic EEG via `--synthetic_only`)
- **Process:** Bandpass filter (1–40 Hz) → notch filter (50 Hz) → sliding windows → baseline encoder (64-dim) + predictive coding encoder (64-dim + prediction errors)
- **Output:** `cache/eeg_eye_bridge/phase1/` — per-trial pickles with embeddings, manifest.json

### Phase 2: Eye Consistency
- **Script:** `scripts/eeg_eye_bridge/phase2/run_phase2.py`
- **Source:** `src/eeg_eye_bridge/phase2_eye_latents/`
- **Input:** Phase 1 cache + eye-tracking CSVs from `Eye/EYE/`
- **Process:** Parse gaze/pupil data → clean (remove noise, interpolate blinks) → compute eye summary vectors → score EEG–eye consistency per trial
- **Output:** `cache/eeg_eye_bridge/phase2/` — eye_summaries/, consistency scores, selected representations

### Phase 3: RDM Construction
- **Script:** `scripts/eeg_eye_bridge/phase3/build_rdms.py`
- **Source:** `src/eeg_eye_bridge/phase3_rdm/`
- **Input:** Phase 1 + Phase 2 caches
- **Process:** Group trials by task/subskill → aggregate feature vectors → build 6+ candidate RDMs (eye-only, EEG-only, joint, performance-tier) using 1−Spearman metric → score by stability/transfer plausibility → rank and write manifest
- **Output:** `cache/eeg_eye_bridge/phase3/` — rdm_manifest.json, rdms/*.pkl

### Phase 4: ViT Training with Brain Regularizer
- **Script:** `src/training/train_vit_system.py`
- **Source:** `src/eeg_eye_bridge/phase4_vit/` (target loading, label grouping)
- **Process:** Load target RDM from Phase 3 manifest → during training, compute model centroid RDM from ViT embeddings grouped by task → RSA loss as soft regularizer
- **Config:** `src/configs/bridge_eeg_rdm.yaml`

## Directory Structure

```
Surgical_Gestures/
├── src/                                # All library code (add to PYTHONPATH)
│   ├── configs/                        # YAML training configs
│   ├── data/                           # Dataset, transforms, splits, sync
│   │   ├── jigsaws_vit_dataset.py      # Main dataset class
│   │   ├── balanced_task_sampler.py    # Equal tasks per batch
│   │   ├── eeg_processor.py            # EEG loading/filtering/epoching
│   │   ├── split_loader.py             # LOUO fold management
│   │   ├── sync_manager.py             # Multi-modal time alignment
│   │   └── transforms_vit.py           # ViT augmentations
│   ├── models/                         # Neural network components
│   │   ├── visual.py                   # ViTFrameEncoder, ViTFlowEncoder
│   │   ├── temporal_transformer.py     # Temporal aggregation
│   │   ├── kinematics.py              # Kinematics decoder head
│   │   ├── decoder_autoreg.py          # Autoregressive decoder
│   │   ├── losses.py                   # All loss functions + compute_total_loss
│   │   └── adapters.py                 # ViT adapter layers
│   ├── modules/
│   │   └── brain_rdm.py                # Differentiable RDM, centroid RDM, RSA loss
│   ├── eeg_eye_bridge/                 # Multi-phase bridge pipeline
│   │   ├── phase1_eeg/                 # EEG encoders (baseline, predictive coding)
│   │   ├── phase2_eye_latents/         # Eye loading, consistency, summarization
│   │   ├── phase3_rdm/                 # RDM core, builders, pipeline, task_bridge, schemas
│   │   ├── phase4_vit/                 # target_loader, label_grouping
│   │   └── integration/                # Cross-phase audit, adapters, schemas, synthetic
│   ├── training/
│   │   └── train_vit_system.py         # Main training loop + EEGInformedViTModel
│   ├── eval/                           # Evaluation (metrics, postprocess)
│   ├── inference/                      # Prediction, embedding extraction, visualization
│   └── safety/                         # dVRK workspace validation, filters
│
├── scripts/eeg_eye_bridge/             # Pipeline runner scripts
│   ├── run_full_pipeline.py            # Orchestrator: Phase 1 → 2 → 3 → 4
│   ├── phase1/run_export.py
│   ├── phase2/run_phase2.py
│   └── phase3/build_rdms.py
│
├── tests/eeg_eye_bridge/              # Per-phase tests (Phase 1–5)
│
├── data/                              # JIGSAWS video/kinematics/transcriptions
├── EEG/                               # Raw EEG recordings (EDF files)
├── Eye/                               # Eye-tracking data, target RDMs, metadata
│   ├── EYE/                           # Per-trial eye CSVs
│   ├── Exploration/                   # Precomputed eye target RDM (target_rdm_3x3.npy)
│   ├── Table1.csv                     # Task ID mappings
│   └── PerformanceScores.csv
├── Gestures/                          # Gesture transcription files
├── cache/eeg_eye_bridge/              # Pipeline intermediate outputs (phase1/, phase2/, phase3/)
├── checkpoints/                       # Trained model weights
├── reports/                           # Test and analysis reports (JSON + Markdown)
├── run_8fold_louo.sh                  # Baseline 8-fold LOUO training
├── run_8fold_louo_brain.sh            # Brain-aligned 8-fold LOUO training
├── generate_splits.py                 # Creates LOUO cross-validation splits
└── aggregate_louo_results.py          # Aggregates results across folds
```

## Key Concepts

**RDM (Representational Dissimilarity Matrix):** A K×K symmetric matrix where entry (i,j) measures dissimilarity between conditions i and j. Diagonal is always 0. The project uses 1−Spearman as the default metric. RDMs allow cross-modal comparison — e.g., does the ViT's internal representation of "Suturing vs Knot Tying" match how the brain/eyes differentiate them?

**RSA (Representational Similarity Analysis):** The technique of comparing RDMs across systems (model vs brain). The loss is 1−Pearson(model_rdm, target_rdm).

**Task Bridge:** Maps 27 simulator task IDs to JIGSAWS task families via transfer plausibility scores (needle_driving=0.95, needle_control=0.85, other=0.45). See `src/eeg_eye_bridge/phase3_rdm/task_bridge.py`.

**Centroid RDM:** During training, embeddings are grouped by condition (e.g., task), averaged to centroids, then pairwise distances give a model RDM. See `src/modules/brain_rdm.py → compute_task_centroid_rdm()`.

## Dependencies

Python 3.11+. Key packages: torch 2.7+, torchvision, timm, numpy (<2.0 for torch compat), scipy, pandas, scikit-learn, mne (EEG), opencv-python, pyyaml, tqdm, matplotlib, seaborn. See `requirements.txt`.

**NumPy constraint:** Keep numpy<2.0 to avoid PyTorch compatibility issues. If you see torch/numpy errors, run: `pip install 'numpy>=1.24,<2' --force-reinstall`

## Coding Conventions

- Python, readability over complexity
- `PYTHONPATH=src` is required for all imports
- YAML-driven config for training — model architecture, loss weights, brain mode all set in config
- Per-phase modularity: each EEG–eye bridge phase has its own source package, script, test, and cache directory
- Tests write JSON + Markdown reports to `reports/eeg_eye_bridge/`
- Phase 2 and 3 tests use pytest; Phase 1, 4, 5 use unittest-style scripts
- Kinematics use 19-dim single-arm format: pos(3) + rot(9 or 6D) + trans_vel(3) + rot_vel(3) + gripper(1)
- Differential learning rates: base=1e-4, ViT backbone=1e-5, adapters=5e-5
- Training uses teacher forcing with decay over 40 epochs
- 8 surgeons → 8-fold LOUO splits (fold_1 through fold_8)

## Common Tasks

**Add a new brain mode:** Create a YAML config under `src/configs/`, set `brain_mode` and relevant parameters, then update `train_vit_system.py` to handle the new mode in the brain loss computation section.

**Add a new RDM type:** Add a builder function in `src/eeg_eye_bridge/phase3_rdm/rdm_builders.py`, call it from `pipeline.py → build_all_candidate_rdms()`, and it will automatically appear in the manifest.

**Run a quick smoke test:** `python scripts/eeg_eye_bridge/run_full_pipeline.py --phase1-synthetic --skip-train`

**Train a single fold:** `python src/training/train_vit_system.py --config src/configs/baseline.yaml --task Suturing --split fold_1 --data_root . --output_dir checkpoints/test`
