# Project Current Status

**Last updated:** March 15, 2026
**Primary author:** Michael Haidar

---

### How to Run

**Baseline (single task):**
```bash
python3 src/training/train_vit_system.py --config src/configs/baseline.yaml --data_root . --task Knot_Tying --split fold_1 --output_dir checkpoints/kt_fold1
```

**Brain alignment (multi-task):**
```bash
python3 src/training/train_vit_system.py --config src/configs/brain_eye.yaml --data_root . --task all --split fold_1 --output_dir checkpoints/brain_fold1
```

**8-fold LOUO with brain:**
```bash
./run_8fold_louo_brain.sh
```

**Ablations:**
```bash
./scripts/run_ablations.sh all
```

---

## 1. Project Goal

Build a **brain-informed ViT pipeline** for surgical gesture recognition and kinematic prediction. The high-level architecture is:

```
Video frames → ViT encoder → per-frame embeddings → Transformer decoder → per-timestep kinematics (k̂ₜ)
                                      ↳ gesture head (frame-level, 15 classes)
                                      ↳ skill head (trial-level, Novice/Intermediate/Expert)

Brain alignment path (training only):
  Cognitive state RDMs (Eye / EEG) ↔ Model RDMs (selected ViT layers) → L_brain (RSA loss)
```

**Three high-level goals** (see `[full_scope_plan.md](full_scope_plan.md)` for full spec):

1. **Brain-aligned visual features**: Learn ViT features whose representational geometry matches surgeon cognitive state RDMs (eye tracking first, EEG later), making features more human-aligned and behaviorally meaningful.
2. **Executable kinematics on dVRK**: Produce deterministic per-timestep kinematics from visual embeddings, post-process for safety, and execute on a physical da Vinci Research Kit (single-arm).
3. **Smooth latent interpolation**: Structure the embedding space so linear/spherical interpolation between frames decodes into smooth, low-jerk kinematic trajectories.

**Current priority**: Eye-tracking brain alignment is **implemented**. Next: retrain Knot Tying to convergence, run full 8-fold LOUO, and complete ablation studies.

---

## 2. Dataset Overview

### 2a. JIGSAWS (`Gestures/`)

The primary video + kinematics dataset. Three surgical tasks, each with:

- **Video**: `.avi` files at 30 FPS (RGB, preprocessed to 224×224)
- **Kinematics**: Full 76-D state → parsed to 19-D per-arm (3D position, 9D rotation matrix, 3D translational velocity, 3D rotational velocity, 1D gripper). Interpolated to match video frame timestamps.
- **Gesture transcriptions**: 15 gesture classes, frame-level labels
- **Skill scores**: OSATS labels (Novice / Intermediate / Expert)
- **Evaluation protocol**: Leave-One-User-Out (LOUO), 8 surgeons → 8 folds


| Task           | Gesture Classes | Notes                                                |
| -------------- | --------------- | ---------------------------------------------------- |
| Suturing       | 15              | Clearest cluster separation in embedding space       |
| Needle Passing | 15              | Known label quality issues at 79.1% accuracy         |
| Knot Tying     | 15              | Training currently unstable; needs full training run |


**Data splits**: `data/splits/{task}_splits.json` and `.yaml`

### 2b. Eye Tracking (`Eye/EYE/`)

- ~1,570 CSV files, naming convention: `{participantID}_{taskID}_{trial}.csv`
- Participants 10–16 across 27 surgical tasks
- **Columns used**: `gaze_x` (col 0), `gaze_y` (col 1), `pupil_left` (col 17), `pupil_right` (col 18), `eye_movement_type` (col 19: 1=Fixation, 2=Saccade, 0/3=Noise)
- **JIGSAWS task ID mapping**: Suturing → ID 17, Needle Passing → ID 15, Knot Tying → ID 18
- Additional metadata: `Eye/PerformanceScores.csv`, `Eye/Table1.csv` (task ID ↔ name mapping)

### 2c. EEG (`EEG/EEG/`)

- ~1,637 EDF files, same naming convention and task coverage as eye tracking
- **Not yet processed** — raw EDF files only; pipeline exists but has not been run
- `cache/eeg_rdms/` is currently empty

---

## 3. What Is Complete

### 3a. ViT Model Architecture (`src/`)


| Component              | File                                 | Status  | Notes                                                                                                                                        |
| ---------------------- | ------------------------------------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Visual encoder         | `src/models/visual.py`               | Done    | ViT-Small, patch 16, 224×224, 384-D CLS tokens                                                                                               |
| Temporal transformer   | `src/models/temporal_transformer.py` | Done    | 4-layer encoder; mean + attention pooling for trial summary                                                                                  |
| Autoregressive decoder | `src/models/decoder_autoreg.py`      | Done    | 6-layer cross-attention; 10-D output (3D pos, 6D rot, 1D jaw)                                                                                |
| Loss functions         | `src/models/losses.py`               | Done    | Smooth L1 pos, geodesic rotation, MSE jaw, jerk penalty; `L_brain` active for `brain_mode: 'eye'`                                            |
| Brain RDM module       | `src/modules/brain_rdm.py`           | Done    | `load_eye_rdm()`, `compute_task_centroid_rdm()`, `eye_rsa_loss()` for task-centroid RSA; EEG path ready when `cache/eeg_rdms/` populated     |
| Adapters               | `src/models/adapters.py`             | Done    | Lightweight MLP adapters for per-subject personalization                                                                                     |


**Loss formulation**:

```
L_total = λ_kin·L_kin + λ_gest·L_gest + λ_skill·L_skill + λ_smooth·L_smooth + λ·L_brain

L_kin = L_pos (smooth L1) + L_rot (geodesic on SO(3)) + L_jaw (MSE)
Weights: λ_kin=1.0, λ_gest=1.0, λ_skill=0.5, λ_smooth=0.5, λ_brain=0.01 (when brain_mode: eye)
```

**Optimizer**: AdamW, differential LR — 1e-5 for ViT backbone, 1e-4 for decoder/heads; linear warmup 5 epochs → cosine annealing to 1e-6; gradient clip norm 1.0; teacher forcing with exponential decay for autoregressive decoder.

**Rotation representation**: 6D continuous rotation (Gram-Schmidt → SO(3)) to avoid Euler/quaternion discontinuities.

### 3b. Training & Evaluation Infrastructure

- **Main training script**: `src/training/train_vit_system.py` — full multi-task loop; `L_brain` integrated; `--task all` for brain alignment; `return_embeddings=True` when `brain_mode in ['rsa','eye']`
- **Data pipeline**: `src/data/jigsaws_vit_dataset.py`, `src/data/jigsaws_multitask_dataset.py` (multi-task with `task_label`), `src/data/balanced_task_sampler.py` (ensures all 3 tasks per batch), `src/data/transforms_vit.py`, `src/data/split_loader.py`
- **EEG processor**: `src/data/eeg_processor.py` — written, not yet run against data
- **Sync manager**: `src/data/sync_manager.py` — multi-modal timestamp alignment; written, not yet validated end-to-end
- **Evaluation**: `src/eval/evaluate.py`, `src/eval/metrics.py`, `src/eval/postprocess.py`
- **Embedding extraction/visualization**: `src/inference/extract_embeddings.py`, `src/inference/visualize_embeddings.py`
- **Shell scripts**: `run_8fold_louo.sh` (config override: `./run_8fold_louo.sh Knot_Tying 1 8 src/configs/custom.yaml`), `run_8fold_louo_brain.sh` (brain-aligned LOUO), `scripts/run_ablations.sh` (brain vs no-brain, λ sweep, adapters vs full fine-tune, ViT layers)

### 3c. Trained Checkpoints & Current Results

Results are from **single-fold LOUO evaluation** (one fold per task); single-fold results may inflate accuracy. Full 8-fold cross-validation is not yet complete.


| Task           | Fold | Epoch | Gesture Acc | Gesture F1 | Skill Acc | Rotation Error |
| -------------- | ---- | ----- | ----------- | ---------- | --------- | -------------- |
| Suturing       | 1    | 37    | **100%**    | **1.00**   | **100%**  | 3.96°          |
| Needle Passing | 1    | 7     | **79.1%**   | **0.79**   | **100%**  | 8.00°          |
| Knot Tying     | 1    | 2     | 24.3%       | 0.11       | 70.3%     | — (unstable)   |
| Knot Tying     | 2    | 10    | 36.5%       | 0.36       | 0%        | — (unstable)   |


**Knot Tying is undertrained.** Both folds stopped early (epoch 2 and epoch 10). The progress report (December 2025) previously achieved 100% on Knot Tying — this suggests the current checkpoint directory reflects interrupted/early training runs. Knot Tying needs to be retrained to convergence (~epoch 30–50) before running full LOUO.

**Pre-extracted embeddings**: `embeddings/*.pkl` (knot tying, needle passing, suturing — fold 1 each)
**Visualization outputs**: `plots/` — t-SNE and UMAP plots showing gesture cluster separation by task

### 3d. Eye Tracking RDM Analysis — COMPLETE

Full exploratory pipeline in `[Eye/Exploration/Eye_data_exploration.ipynb](Eye/Exploration/Eye_data_exploration.ipynb)`. Key outputs are saved and ready to use as training supervision.

**Pipeline summary:**

1. **Data loading & cleaning**: Load CSV, extract 5 columns, remove noise/blink rows (type 0/3), interpolate pupil=0 values (blinks) linearly
2. **Feature engineering**: Apply Gaussian smoothing (σ=2) to raw gaze/pupil signals; compute 500ms rolling window (15 samples @ 30 Hz):
  - `fixation_ratio` — proportion of fixations in window (focused attention)
  - `gaze_entropy` — √(std_x² + std_y²) (spatial dispersion / searching)
  - `pupil_variance` — std of average pupil diameter (cognitive load shifts)
3. **K-Means attempt (failed)**: 92.9% of detected states lasted <1 second ("flickering") — static clustering is inappropriate for temporal data
4. **HMM pivot (success)**: Switched to `GaussianHMM(n_components=5, covariance_type="diag", n_iter=100)` from `hmmlearn`. Achieves median run length >1.5 s; transition matrix diagonal values >0.9 (sticky/stable states). K=5 selected via model selection sweep.
5. **Universal HMM**: Single HMM trained on a multi-file corpus to establish a shared "vocabulary" of 5 cognitive states across all participants and tasks
6. **Task fingerprinting**: All ~1,570 eye tracking files processed → per-task normalized histogram over 5 cognitive states ("fingerprint"). 27 task fingerprints computed.
7. **Full 27×27 RDM**: Pairwise Euclidean distance between task fingerprints; covers all 27 surgical tasks. RDM range: [0.0000, 0.2409]
8. **3×3 JIGSAWS sub-RDM extracted and saved** (the key output):


| Output file                                  | Description                                                             |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| `Eye/Exploration/target_rdm_3x3.npy`         | **Normalized [0,1] RDM** — ready to use as `L_brain` supervision signal |
| `Eye/Exploration/target_rdm_3x3_raw.npy`     | Raw Euclidean distance matrix                                           |
| `Eye/Exploration/target_rdm_3x3_heatmap.png` | Visualization (rows/cols: Suturing, Needle Passing, Knot Tying)         |


**The 3×3 RDM encodes how dissimilar the cognitive states of surgeons are across the three JIGSAWS tasks.** This is the ground-truth representational geometry that the ViT's task-level embeddings should match.

### 3e. Robot Execution (`src/safety/`)

- **Relative motion replay**: `T_target = T_measured × (T_current⁻¹ × T_next)` — avoids need for absolute coordinate frame calibration (which JIGSAWS lacks)
- **Safety post-processing pipeline**: velocity limit 0.03 m/s, acceleration limit 0.15 m/s², Gaussian filtering (short-range), Savitzky-Golay filtering (3rd-order polynomial, broader smoothing), outlier rejection via z-score + interpolation, SO(3) projection via SVD
- **dVRK interface**: `src/safety/dvrk_interface.py` — `servo_cp` / `servo_jp` commands at 30 Hz
- **ROS2 pipeline**: Implemented by Mai Bui; tested in simulation

---

## 4. Eye Tracking Integration — IMPLEMENTED

### What Was Implemented

**Task-centroid RSA** (Option A) is now integrated:

1. **`src/modules/brain_rdm.py`** — Added:
   - `load_eye_rdm(path)` — loads `target_rdm_3x3.npy`, returns (3, 3) tensor
   - `compute_task_centroid_rdm(embeddings, task_labels)` — groups by task (0=Suturing, 1=Needle Passing, 2=Knot Tying), computes mean per task, returns pairwise Euclidean distance matrix
   - `eye_rsa_loss(model_rdm, target_rdm)` — differentiable Pearson-based RSA loss (1 − correlation) for backprop

2. **`src/data/jigsaws_multitask_dataset.py`** — New multi-task dataset combining all 3 tasks with `task_label` per sample; uses `SplitLoader` per task to merge train/val/test across tasks per fold.

3. **`src/data/balanced_task_sampler.py`** — New batch sampler ensuring each batch has samples from all 3 tasks (round-robin sampling).

4. **`src/training/train_vit_system.py`** — Integrated:
   - `--task all` for multi-task brain alignment
   - Load `target_rdm_3x3.npy` at startup when `brain_mode == 'eye'`
   - `return_embeddings=True` when `brain_mode in ['rsa','eye']`
   - Per-batch: flatten embeddings, expand task labels, compute 3×3 model RDM, apply `L_brain`
   - Log RSA correlation in epoch summary

5. **`src/models/losses.py`** — Added `brain_mode: 'eye'` branch using `eye_rsa_loss`.

6. **Configs**: `src/configs/brain_eye.yaml` (Stage 2), `src/configs/brain_eye_stage3.yaml` (Stage 3, reduced LR).

7. **Scripts**: `run_8fold_louo_brain.sh`, `scripts/run_ablations.sh` (brain, λ, finetune, layers).

### Remaining Next Steps

- **Retrain Knot Tying** to epoch 30–50 (current checkpoints undertrained)
- **Run full 8-fold LOUO** baseline: `run_8fold_louo.sh`; then brain-aligned: `run_8fold_louo_brain.sh`
- **Run ablation studies**: `./scripts/run_ablations.sh all`
- **Future**: Per-file fingerprint RDM (Option B) for finer-grained alignment

---

## 5. Future Work — EEG Integration Path

**Prerequisite**: Complete eye-tracking integration (Section 4) first. EEG data exists but requires preprocessing before it can be used.

### Step 1 — Preprocess EEG

- Run `src/data/eeg_processor.py` against all ~1,637 EDF files in `EEG/EEG/`
- Apply: bandpass filtering, epoch extraction aligned to trial timestamps, artifact rejection (ICA or threshold-based)
- EEG covers the same 27 tasks and participants as eye tracking — use `Eye/Table1.csv` for task ID mapping

### Step 2 — Precompute EEG RDMs

- Run `scripts/precompute_eeg_rdms.py`; outputs should go to `cache/eeg_rdms/` (currently empty)
- Window spec: sliding windows aligned to video frames; tau-lag range 0–300 ms
- The `BrainRDM` class in `src/modules/brain_rdm.py` is designed to load from `cache/eeg_rdms/` — it will work once populated

### Step 3 — Cross-Modal Validation

- Compute Spearman correlation between EEG-derived RDMs and the existing `target_rdm_3x3.npy` (eye-tracking RDM)
- High correlation = convergent validity; use whichever has higher SNR as the primary `L_brain` signal
- Optionally combine both: `L_brain = α·L_brain_eye + (1-α)·L_brain_eeg`

### Step 4 — Training with EEG Brain Loss

- Once `cache/eeg_rdms/` is populated, the `BrainRDM.load_eeg_rdm()` and `BrainRDM.get_eeg_rdm()` methods can be used directly
- EEG RDMs are per-trial (finer-grained than the 3×3 task-level eye RDM) — this enables frame-level RSA alignment with tau-lag search
- Re-run full LOUO with EEG `L_brain`; compare to eye-tracking alignment and baseline

### Step 5 — Per-Subject Adapter Training

- Use `src/models/adapters.py` for lightweight per-subject personalization
- Protocol: freeze ViT backbone, train adapters only during Stage 2 (brain alignment)
- EEG enables per-surgeon personalization (individual neural patterns); eye tracking provides a population-level average

---

## 6. Known Issues & Blockers


| Issue                                                       | Impact                                                  | Resolution                                         |
| ----------------------------------------------------------- | ------------------------------------------------------- | -------------------------------------------------- |
| Knot Tying undertrained (fold 1: epoch 2, fold 2: epoch 10) | Gesture F1 < 0.4 on KT; blocks reliable LOUO comparison | Retrain KT to epoch 30–50                          |
| Full 8-fold LOUO not complete                               | Single-fold results may inflate accuracy; no mean ± std | Run `run_8fold_louo.sh` and `run_8fold_louo_brain.sh` |
| `cache/eeg_rdms/` empty                                     | EEG path entirely blocked                               | Run `eeg_processor.py` + `precompute_eeg_rdms.py`  |
| `sync_manager.py` not validated end-to-end                  | Multi-modal timestamp alignment unverified              | Validate with small test clips before EEG training |


---

## 7. Key File Map


| File / Directory                                                                           | Purpose                                                     | Status                    |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------- |
| `[full_scope_plan.md](full_scope_plan.md)`                                                 | Full architectural spec and training stages                 | Reference                 |
| `[src/training/train_vit_system.py](src/training/train_vit_system.py)`                     | Main training script — `L_brain` integrated; `--task all`   | Done                      |
| `[src/modules/brain_rdm.py](src/modules/brain_rdm.py)`                                     | RSA loss; `load_eye_rdm`, `compute_task_centroid_rdm`, `eye_rsa_loss` | Done                      |
| `[src/data/jigsaws_multitask_dataset.py](src/data/jigsaws_multitask_dataset.py)`           | Multi-task dataset with `task_label` for brain alignment    | Done                      |
| `[src/data/balanced_task_sampler.py](src/data/balanced_task_sampler.py)`                   | Batch sampler ensuring all 3 tasks per batch                | Done                      |
| `[src/configs/brain_eye.yaml](src/configs/brain_eye.yaml)`                                 | Brain alignment config (Stage 2)                            | Done                      |
| `[src/configs/brain_eye_stage3.yaml](src/configs/brain_eye_stage3.yaml)`                   | Stage 3 joint fine-tune (reduced LR)                         | Done                      |
| `[Eye/Exploration/Eye_data_exploration.ipynb](Eye/Exploration/Eye_data_exploration.ipynb)` | Completed eye-tracking RDM analysis                         | Done                      |
| `[Eye/Exploration/target_rdm_3x3.npy](Eye/Exploration/target_rdm_3x3.npy)`                 | **Eye-tracking supervision signal** (3×3, normalized [0,1]) | Ready to use              |
| `[Eye/Exploration/target_rdm_3x3_raw.npy](Eye/Exploration/target_rdm_3x3_raw.npy)`         | Raw Euclidean distances (pre-normalization)                 | Ready to use              |
| `[Eye/Exploration/target_rdm_3x3_heatmap.png](Eye/Exploration/target_rdm_3x3_heatmap.png)` | Visualization of 3×3 RDM                                    | Reference                 |
| `[src/data/jigsaws_vit_dataset.py](src/data/jigsaws_vit_dataset.py)`                       | JIGSAWS single-task data loader                             | Done                      |
| `[src/models/losses.py](src/models/losses.py)`                                             | Loss functions — `brain_mode: 'eye'` branch active          | Done                      |
| `[run_8fold_louo_brain.sh](run_8fold_louo_brain.sh)`                                        | 8-fold LOUO with brain alignment                            | Done                      |
| `[scripts/run_ablations.sh](scripts/run_ablations.sh)`                                      | Ablation studies (brain, λ, finetune, layers)               | Done                      |
| `[data/splits/*.json](data/splits/)`                                                       | LOUO train/test splits (all 3 tasks, 8 folds each)          | Done                      |
| `[checkpoints/](checkpoints/)`                                                             | Trained model weights (multiple tasks × folds × epochs)     | Partial (KT undertrained) |
| `[embeddings/*.pkl](embeddings/)`                                                          | Pre-extracted embeddings for visualization                  | Partial (fold 1 only)     |
| `[cache/eeg_rdms/](cache/eeg_rdms/)`                                                       | Precomputed EEG RDMs (loaded by `BrainRDM`)                 | **Empty — blocked**       |
| `[src/data/eeg_processor.py](src/data/eeg_processor.py)`                                   | EEG preprocessing — not yet run                             | Not run                   |
| `[scripts/precompute_eeg_rdms.py](scripts/precompute_eeg_rdms.py)`                           | EEG RDM precomputation — not yet run                        | Not run                   |
| `[src/safety/dvrk_interface.py](src/safety/dvrk_interface.py)`                              | dVRK robot interface (relative motion replay)              | Done                      |
| `[src/safety/filters.py](src/safety/filters.py)`                                           | Safety post-processing filters                              | Done                      |


