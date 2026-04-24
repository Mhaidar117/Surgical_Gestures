# CLAUDE.md — Project Context for Surgical Gestures

**Repository docs:** Only this file and [`README.md`](README.md) are versioned as project documentation. Optional long-form notes (embedding workflows, legacy README text, Phase 1 schema detail) may live in a local `docs/` directory, which is gitignored.

## Project Overview

ViT-based surgical gesture recognition and kinematics prediction for the da Vinci Research Kit (dVRK), with a multi-phase **EEG–Eye Bridge** pipeline that derives neuroscientific regularizers from EEG and eye-tracking data. The model watches surgical video and simultaneously predicts robot kinematics, gesture labels, and surgeon skill level.

On top of the training stack, a **Skill-Manifold Comparison** pipeline (Gromov–Wasserstein between JIGSAWS and the EEG/Eye simulator) asks whether the two datasets organize skill in compatible geometries, with a family of trial-level null tests, bootstraps, and modality splits.

**Dataset:** JIGSAWS (JHU-ISI Gesture and Skill Assessment Working Set) — three tasks: Suturing, Needle Passing, Knot Tying. Video at 30 Hz, 76-dim kinematics (4 arms × 19D), 15 gesture classes, 3 skill levels (E/I/N), 8 surgeons.

**Evaluation protocol:** 8-fold Leave-One-User-Out (LOUO) cross-validation.

## Datasets: JIGSAWS vs. EEG vs. Eye — Do Not Conflate

This project uses **three distinct datasets** that are often confused because they all involve robot-assisted surgery on da Vinci hardware. **They come from different studies, different institutions, different subjects, and different hardware.** Only the **EEG and Eye datasets share tasks and subjects**; JIGSAWS is wholly separate from both.

### Quick disambiguation

| | **JIGSAWS** | **EEG (NIBIB-RPCCC-RAS)** | **Eye (NIBIB-RPCCC-RAS)** |
|---|---|---|---|
| Source | JHU + Intuitive Surgical, MICCAI M2CAI 2014 (Gao et al.) | Roswell Park, PhysioNet 2023 (Shafiei et al., NIH R01EB029398) | **Same study as EEG** (same paper, same release) |
| Hardware | **Real** da Vinci Surgical System (dVSS) with research API | da Vinci **Skills Simulator** (virtual, Mimic Technologies) + 128-ch AntNeuro EEG | Same simulator as EEG + Tobii Pro eyeglasses |
| Subjects | **8 surgeons**, IDs `{B, C, D, E, F, G, H, I}` | **25 participants** (age 20–67, RAS experience 0 to >1000 h) | **Same 25 participants** as EEG |
| Tasks | **3 bench-top tasks**: Suturing, Knot-Tying, Needle-Passing | **27 virtual simulator tasks** across 6 modules (Task IDs 1–27) | **Same 27 simulator tasks** as EEG |
| Signals | Kinematics (76-d @ 30 Hz) + stereo endoscope video (640×480 @ 30 Hz) | Raw EEG, 128 channels @ 500 Hz, `.edf` | 20 gaze/pupil metrics @ 50 Hz, `.csv` |
| Skill label | Modified OSATS GRS (6 items × 1–5 Likert, summed) + self-reported E/I/N | Simulator performance score 0–100 (per trial) | Same performance score as EEG |
| Trials | 5 reps/subject/task; 39 SU + 36 KT + 28 NP usable | ≥2 tries per task/subject until a ≥70 score was reached; 1636 `.edf` files | Recorded **simultaneously** with EEG; 1559 `.csv` files (some eye trials lost) |

### Key relationships

- **EEG ↔ Eye** share study, subjects, tasks, and were recorded **simultaneously**. A filename `<participantID>_<taskID>_<try>` refers to the same trial in both `EEG/` (.edf) and `Eye/EYE/` (.csv). Alignment + performance scores live in `Eye/PerformanceScores.csv`; the 27 simulator task IDs and their names are in `Eye/Table1.csv`.
- **JIGSAWS ↔ EEG/Eye**: **no shared subjects, no shared tasks, no shared hardware, no shared trial identity.** JIGSAWS is physical surgery on a bench-top tissue model with a real dVSS; the EEG/Eye dataset is a virtual simulator running different exercises entirely.
- **How this project bridges them:** because there is no trial-level correspondence across datasets, the skill-manifold pipeline (`pipeline/skill_manifold_gw.py`) uses **Gromov–Wasserstein** on intra-set distances only. A soft semantic bridge from the 27 simulator task IDs to JIGSAWS task families is provided as *transfer-plausibility scores* in `src/eeg_eye_bridge/phase3_rdm/task_bridge.py` (`needle_driving=0.95`, `needle_control=0.85`, `other=0.45`). These are heuristic weights, **not** shared trial IDs.

### Structural detail

**JIGSAWS** (`data/`, `Gestures/`). Per trial: `<Task>_<SubjectLetter><RepNumber>`, e.g. `Knot_Tying_B001`. Each trial ships with a 76-d kinematics file at 30 Hz (Left MTM 19-d + Right MTM 19-d + PSM1 19-d + PSM2 19-d; per manipulator = xyz(3) + R(9) + linear vel(3) + angular vel(3) + gripper(1)), two synchronized AVI endoscopic streams (`capture1`/`capture2` = left/right), a gesture transcription labeling frame ranges with G1–G15, and a meta file with OSATS GRS total + self-reported E/I/N. CV: 8-fold LOUO (one surgeon out) or 5-fold LOSO (one repetition out); this project uses LOUO.

**EEG** (`EEG/`). Filenames `<participantID>_<taskID>_<try>.edf`, participantID ∈ 1..25, taskID ∈ 1..27. 128-channel EEG at 500 Hz, reference = Cz. Four leads (`EEGHEOGRCPz`, `EEGHEOGLCPz`, `EEGVEOGUCPz`, `EEGVEOGLCPz`) are **electrooculogram and must be excluded**. Channels F8, POz, AF4, AF8, F6, FC3 are flagged low-quality on some recordings.

**Eye** (`Eye/EYE/`). Filenames `<ParticipantID>_<TaskID>_<try>.csv`, **aligned 1:1 with EEG filenames for the same trial**. 50 Hz Tobii Pro. 20 columns per row: gaze point 2D (px), gaze point 3D (mm), gaze direction L/R 3D, pupil position L/R 3D (mm), pupil diameter L/R (mm), and eye-movement-type index (`1` = fixation, `2` = saccade, `0` = unknown). Tobii Pro 2 preprocessing: 3-point moving average + 30 deg/s angular-velocity fixation/saccade threshold. Eye trial count (1559) < EEG trial count (1636) because some trials lost eye or EEG but never both — see `PerformanceScores.csv`.

**Shared metadata** (EEG + Eye only):
- `Eye/PerformanceScores.csv` — one row per trial: participantID, taskID, try, performance (0–100), age, dominant hand, plus flags for EEG/eye availability.
- `Eye/Table1.csv` — maps simulator `taskID` 1..27 to names: Pick and Place, Pegboard 1/2, Match Board 1/2/3, Ring and Rail 1/2, Camera Targeting 1/2, Scaling, Ring Walk 1/2/3, Needle Targeting, Thread the Rings, Suture Sponge 1/2/3, Dots and Needles 1/2, Tubes, Energy Switching 1/2, Energy Dissection 1/2/3.

### Naming traps (seen in prior confusions)

- "Subject B" in JIGSAWS is a **surgeon** (one of 8 real humans, identified by letter). "Participant 1..25" in EEG/Eye is a **simulator user** (a different group of real humans). **No known overlap.**
- "Suturing" (JIGSAWS) is a physical bench-top task on a tissue model. "Suture Sponge 1/2/3" (EEG/Eye) is a virtual simulator exercise. **Different tasks.**
- "Skill" in JIGSAWS = OSATS GRS (6 × Likert, summed) plus self-reported E/I/N. "Skill" in EEG/Eye = simulator performance 0–100. **Not numerically comparable**; the GW pipeline tertiles each side independently before any cross-dataset comparison.
- "da Vinci" appears in both, but JIGSAWS uses the **real dVSS** and EEG/Eye use the **da Vinci Skills Simulator** (virtual training system by Mimic Technologies). A Mimic task is rendered; a JIGSAWS trial is physical.

## Quick Reference — How to Run Things

```bash
# Always set PYTHONPATH first
export PYTHONPATH=src

# Generate cross-validation splits
python pipeline/generate_splits.py

# Full EEG–Eye bridge pipeline (Phase 1 → 2 → 3 → optional Phase 4 training)
python pipeline/run_full_pipeline.py
python pipeline/run_full_pipeline.py --phase1-synthetic --skip-train  # smoke test

# Or interactively: run cells in pipeline/pipeline.ipynb

# 8-fold LOUO baseline training
./run_8fold_louo.sh                          # all tasks, baseline config
./run_8fold_louo.sh Suturing 1 4             # single task, folds 1-4
./run_8fold_louo.sh all 1 8 brain_eye.yaml   # custom config

# 8-fold LOUO with brain alignment
./run_8fold_louo_brain.sh

# Skill-manifold GW comparison (JIGSAWS vs EEG/Eye), Comparisons A + B
python pipeline/skill_manifold_gw.py                     # full run, outputs -> reports/skill_manifold/
python pipeline/skill_manifold_gw.py --smoke             # 10% subsample smoke pass
python pipeline/skill_manifold_gw.py --n_perms 200       # faster null
# Or interactively: pipeline/skills_manifold.ipynb

# Post-hoc skill separability on a trained checkpoint's embeddings
python pipeline/skill_manifold_analysis.py --checkpoint checkpoints/brain_eye/all/fold_1/best_model.pth \
    --data_root . --task all --split fold_1

# Linear + k-NN probes aggregated across folds (skill / gesture / task / surgeon)
python pipeline/representation_probe.py --aggregate_root checkpoints/brain_eye/all \
    --data_root . --task all --split_family louo --output_dir analysis/representation_probe --stem brain_eye

# Hyperparameter sweep driver (Cartesian product over a base YAML)
python pipeline/run_hparam_sweep.py --base_config src/configs/kinematics_rsa.yaml \
    --sweep '{"loss_weights.brain": [0.01, 0.05, 0.1]}' --task all --split fold_1 \
    --split_family louo --output_root checkpoints/sweeps/kin_rsa_brain_weight --primary_metric gesture

# Run all EEG–eye bridge tests
python tests/eeg_eye_bridge/phase1/test_phase1_eeg.py
python -m pytest tests/eeg_eye_bridge/phase2/test_phase2_eye_consistency.py -v
python -m pytest tests/eeg_eye_bridge/phase3/test_phase3_rdms.py -v
python tests/eeg_eye_bridge/phase4/test_phase4_vit_regularizer.py
python tests/eeg_eye_bridge/test_phase5_integration_coordinator.py

# Run all skill-manifold unit + smoke tests
python -m pytest tests/skill_manifold -v
```

**Windows (PowerShell)** — same commands with `python` and `$env:PYTHONPATH = "src"` instead of `export`. Use Git Bash or WSL for `./run_8fold_louo.sh` / `./run_8fold_louo_brain.sh`, or invoke `python`/`bash` explicitly.

**Evaluation reports for LOUO aggregation:** [`src/eval/evaluate.py`](src/eval/evaluate.py) accepts optional `--output_dir eval_results` together with `--split fold_N` to write `<Task>_test_fold_<n>_results.txt` files compatible with [`pipeline/aggregate_louo_results.py`](pipeline/aggregate_louo_results.py). [`run_8fold_louo.sh`](run_8fold_louo.sh) passes `--output_dir` after each fold’s training.

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
- **Script:** `pipeline/phase1_run_export.py`
- **Source:** `src/eeg_eye_bridge/phase1_eeg/`
- **Input:** Raw EDF files (or synthetic EEG via `--synthetic_only`)
- **Process:** Bandpass filter (1–40 Hz) → notch filter (50 Hz) → sliding windows → baseline encoder (64-dim) + predictive coding encoder (64-dim + prediction errors)
- **Output:** `cache/eeg_eye_bridge/phase1/` — per-trial pickles with embeddings, manifest.json

### Phase 2: Eye Consistency
- **Script:** `pipeline/phase2_run_phase2.py`
- **Source:** `src/eeg_eye_bridge/phase2_eye_latents/`
- **Input:** Phase 1 cache + eye-tracking CSVs from `Eye/EYE/`
- **Process:** Parse gaze/pupil data → clean (remove noise, interpolate blinks) → compute eye summary vectors → score EEG–eye consistency per trial
- **Output:** `cache/eeg_eye_bridge/phase2/` — eye_summaries/, consistency scores, selected representations

### Phase 3: RDM Construction
- **Script:** `pipeline/phase3_build_rdms.py`
- **Source:** `src/eeg_eye_bridge/phase3_rdm/`
- **Input:** Phase 1 + Phase 2 caches
- **Process:** Group trials by task/subskill → aggregate feature vectors → build 6+ candidate RDMs (eye-only, EEG-only, joint, performance-tier) using 1−Spearman metric → score by stability/transfer plausibility → rank and write manifest
- **Output:** `cache/eeg_eye_bridge/phase3/` — rdm_manifest.json, rdms/*.pkl

### Phase 4: ViT Training with Brain Regularizer
- **Script:** `src/training/train_vit_system.py`
- **Source:** `src/eeg_eye_bridge/phase4_vit/` (target loading, label grouping)
- **Process:** Load target RDM from Phase 3 manifest → during training, compute model centroid RDM from ViT embeddings grouped by task → RSA loss as soft regularizer
- **Config:** `src/configs/bridge_eeg_rdm.yaml`

## Skill-Manifold Comparison (Gromov–Wasserstein)

A separate analysis pipeline, independent of training, that asks: **do JIGSAWS and the EEG/Eye simulator organize surgical skill into compatible geometries?** Uses Gromov–Wasserstein (GW) because the two datasets live in disjoint feature spaces with no shared task taxonomy — GW only needs intra-set dissimilarities.

- **Orchestrator:** [`pipeline/skill_manifold_gw.py`](pipeline/skill_manifold_gw.py) (steps 1–12 + Comparisons A/B)
- **Library:** `src/skill_manifold/` — `features_jigsaws.py`, `features_eeg_eye.py`, `residualize.py`, `binning.py`, `rdms.py`, `gw.py`, `trial_null.py`, `io.py`
- **Configs:** [`src/configs/skill_manifold.yaml`](src/configs/skill_manifold.yaml) (seed, `n_perms`, `gw_epsilon`, subsample/bootstrap counts, gesture pool, OSATS axes) and [`src/configs/skill_manifold_task_modules.yaml`](src/configs/skill_manifold_task_modules.yaml) (27 simulator task IDs → 9 modules)
- **Tests:** `tests/skill_manifold/` — unit tests per module plus an end-to-end smoke test against a synthetic fixture
- **Notebook:** [`pipeline/skills_manifold.ipynb`](pipeline/skills_manifold.ipynb)
- **Outputs:** `reports/skill_manifold/` — `report_comparison_B.md`, `results_comparison_B.json`, plots in `plots/`

### Pipeline stages (in `skill_manifold_gw.py::run`)

1. **JIGSAWS features** — 14-d gesture histogram over the pooled gesture set + per-arm kinematic summaries (Slave-Left + Slave-Right, 12 scalars each: speed / rot-speed stats, jerk, path length, economy of motion, gripper stats + open-rate) + duration → ~39 d per trial.
2. **EEG/Eye features** — mean of Phase 1 `baseline_embeddings` (64 d) + mean of Phase 1 `pc_embeddings` (64 d) + reconstructed 18-d eye-summary vector (occupancy, transition-matrix diagonal, mean dwell, blink fraction, event summary). Filter to `Try == 1` trials.
3. **Residualization** — per-feature OLS on nuisance covariates. JIGSAWS: `task`, `surgeon`, `trial_index_within_surgeon_task`. EEG/Eye: `task_module`, `subject_id`, `dominant_hand`, `age`. Check `max_post_fit_r2 ≈ 0`.
4. **Tertile binning** — per-side Low/Mid/High tiers from `grs_total` (JIGSAWS) and `performance` (EEG/Eye). Fixed-cutoff binning (JIGSAWS 16/22, Mimic 70/85) is also computed as a sensitivity.
5. **Centroid RDMs** — 3×3 cosine-distance RDMs over tier centroids on each side.
6. **Headline GW** — `ot.gromov_wasserstein` over the two 3×3 RDMs → distance + 3×3 coupling + argmax assignment.
7. **Permutation null** — tier-shuffle null (`n_perms`, default 1000) on the centroid GW distance → p-value, z-score, null histogram.
8. **OSATS axis breakdown** — rerun Steps 4–7 with JIGSAWS tiers defined by each OSATS axis (`respect_for_tissue`, `suture_needle_handling`, `time_and_motion`, `flow_of_operation`, `overall_performance`, `quality_of_final_product`) against the same EEG/Eye centroid RDM.
9. **Trial-level block-diagonality null** — entropic GW on the full NJ×NE coupling (epsilon `gw_epsilon`, default 0.01); aggregate to a 3×3 block-mass `B`; null shuffles tier labels on each side and recomputes `diag_mass = trace(B)`. Primary = all-trials (unbalanced, tier-proportional marginals); sensitivities = balanced 100-per-tier (`subsample_per_tier`), epsilon sweep, fixed-cutoff, and stratified bootstrap (tier-preserving) under both tertile and fixed-cutoff binning. Bonferroni z threshold for 3 diagonal cells is `BONFERRONI_Z_THREE_CELLS ≈ 2.394`.
10. **Mimic-side modality split** — repeat the trial-level block null on three disjoint column subsets of the residualized EEG/Eye matrix: `eeg_baseline` (64 d), `eeg_predictive_coding` (64 d), `eye` (18 d). Diagnostics: per-trial Pearson correlation between baseline and PC EEG, pooled-128 random-split negative control for the baseline-vs-PC delta, and an eye-only coupling-matrix diagnostic.
11. **JIGSAWS-side modality split** — same logic on the robot side, splitting into `gestures` vs `kinematics` subsets.
12. **MDS plots** — 2-D MDS scatters of residualized features colored by tier on each side.

### Comparisons A and B

- **Comparison B (skill manifold)** — the main analysis above: JIGSAWS tiered by OSATS `grs_total`, Mimic tiered by `performance`. Asks whether *observed* skill level maps across datasets.
- **Comparison A (practice manifold)** — rerun with a lighter residualization (drop the tier-defining nuisance: `surgeon` for JIGSAWS, `subject_id` for Mimic) and different tier definitions: JIGSAWS uses self-reported E/I/N (Novice → Low, Intermediate → Mid, Expert → High); Mimic uses tertile of per-subject non-first-try `experience_trials` count (practice-depth proxy). Asks whether *practice depth* aligns across datasets.

### Key knobs (in `skill_manifold.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `n_perms` | 1000 | permutation resamples for steps 7 and the trial-level null |
| `gw_epsilon` | 0.01 | entropic regularization for the NJ×NE trial-level GW |
| `subsample_per_tier` | 100 | balanced subsample size per tier (Step 9 balanced sensitivity) |
| `trial_null_n_subsamples` | 30 | B in the stratified bootstrap loops |
| `modality_split_n_bootstraps` | 8 | B per (modality × binning) in Steps 10/11 (cost multiplies) |
| `modality_split_n_perms` | 400 | perms for the modality-split primary |
| `gestures_pool` | G1–G15 except G7 | gestures retained in the histogram |
| `osats_axes` | list | OSATS columns for the Step 8 breakdown |

### Verdict rule (stratified bootstrap)

Each bootstrap yields per-cell diagonal z-scores; the verdict combines median per-cell z, fraction with z > 0, and fraction with |z| > Bonferroni (≈2.394) across B replicates → `GO / CAUTION / NO-GO`. See `stratified_bootstrap_verdict` in `src/skill_manifold/trial_null.py`.

### Modality-split postmortem

`pipeline/skills_manifold_modality_split_postmortem.txt` documents a known asymmetry: Step 10 splits only the Mimic side, not the JIGSAWS side (Step 11 partially addresses this). It also flags that `eeg_baseline` and `eeg_predictive_coding` are not statistically independent channels, so a "localization" claim across EEG streams is weak. The pooled-128 EEG random-split diagnostic is the intended negative control for that claim.

## Post-hoc representation analyses

Two companion scripts probe what an already-trained ViT checkpoint actually encodes, independent of the GW pipeline.

- [`pipeline/skill_manifold_analysis.py`](pipeline/skill_manifold_analysis.py) — per fold, pools the temporal-aggregator memory to a 384-d embedding per sample, then reports within-skill vs between-skill distances, separability ratio, and silhouette coefficient; stratified repeat by gesture class. Writes `<stem>_metrics.json`, `<stem>_pca.png`, `<stem>_dispersion.png` to `analysis/skill_manifold/`.
- [`pipeline/representation_probe.py`](pipeline/representation_probe.py) — aggregates per-sample embeddings across folds (each fold uses its held-out checkpoint, no leakage) and trains ridge-regularized logistic regression + k-NN (k=5) under 5-fold CV for each probe target: `skill`, `gesture`, `task`, `surgeon`. Surgeon probe is interpreted as *lower is better* (invariance to motor style).

## Hyperparameter sweeps

[`pipeline/run_hparam_sweep.py`](pipeline/run_hparam_sweep.py) takes a base YAML + a JSON sweep spec like `'{"loss_weights.brain": [0.01, 0.05, 0.1]}'`, runs each configuration as a subprocess of `train_vit_system.py`, then aggregates val metrics from every `best_model.pth` into a ranked summary. Supports nested dotted keys and `--dry_run` to print the plan without launching.

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
│   ├── skill_manifold/                 # GW skill-manifold comparison library
│   │   ├── io.py                       # Paths, config loader, OSATS column list
│   │   ├── features_jigsaws.py         # Gesture histogram + per-arm kinematic summaries
│   │   ├── features_eeg_eye.py         # Phase 1/2 means + 18-d eye summary
│   │   ├── residualize.py              # Per-feature OLS nuisance regression
│   │   ├── binning.py                  # Tertile + fixed-cutoff Low/Mid/High tiers
│   │   ├── rdms.py                     # Centroid + pairwise cosine RDMs
│   │   ├── gw.py                       # GW + entropic GW + permutation null
│   │   └── trial_null.py               # Block-mass null, bootstrap, modality splits, diagnostics
│   ├── training/
│   │   └── train_vit_system.py         # Main training loop + EEGInformedViTModel
│   ├── eval/                           # Evaluation (metrics, postprocess)
│   ├── inference/                      # Prediction, embedding extraction, visualization
│   └── safety/                         # dVRK workspace validation, filters
│
├── pipeline/                           # All pipeline-run Python scripts + wrapper notebooks
│   ├── pipeline.ipynb                   # Interactive wrapper: EEG–Eye bridge phases + visualize
│   ├── skills_manifold.ipynb            # Interactive wrapper: GW skill-manifold comparison
│   ├── run_full_pipeline.py            # Orchestrator: Phase 1 → 2 → 3 → 4
│   ├── phase1_run_export.py
│   ├── phase2_run_phase2.py
│   ├── phase3_build_rdms.py
│   ├── skill_manifold_gw.py            # Comparison A + B orchestrator (GW + null + bootstrap)
│   ├── skill_manifold_analysis.py      # Post-hoc skill separability on a checkpoint
│   ├── representation_probe.py         # Linear + k-NN probes across folds (skill/gesture/task/surgeon)
│   ├── run_hparam_sweep.py             # Cartesian-product sweep driver for train_vit_system.py
│   ├── generate_splits.py              # LOUO split generator
│   ├── aggregate_louo_results.py       # Aggregates fold results
│   ├── manuscript_writer.py            # LaTeX manuscript generator
│   ├── export_ablation_analysis.py     # Ablation comparison export
│   ├── precompute_eeg_rdms.py          # Offline EEG RDM caching
│   ├── precompute_raft.py              # Optical flow precompute
│   └── skills_manifold_modality_split_postmortem.txt  # Design critique of Step 10
│
├── tests/
│   ├── eeg_eye_bridge/                 # Per-phase tests (Phase 1–5)
│   └── skill_manifold/                 # Per-module tests + end-to-end smoke test
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
├── analysis/                          # Post-hoc analyses (skill_manifold/, representation_probe/, ...)
├── reports/                           # Test and analysis reports (JSON + Markdown)
│   └── skill_manifold/                # report_comparison_B.md, results_comparison_B.json, plots/
├── run_8fold_louo.sh                  # Baseline 8-fold LOUO training
├── run_8fold_louo_brain.sh            # Brain-aligned 8-fold LOUO training
├── run_ablation_study.ps1             # Full 4-condition ablation runner
└── misc/                              # Historical / unrelated scripts and outputs
```

## Key Concepts

**RDM (Representational Dissimilarity Matrix):** A K×K symmetric matrix where entry (i,j) measures dissimilarity between conditions i and j. Diagonal is always 0. The project uses 1−Spearman as the default metric. RDMs allow cross-modal comparison — e.g., does the ViT's internal representation of "Suturing vs Knot Tying" match how the brain/eyes differentiate them?

**RSA (Representational Similarity Analysis):** The technique of comparing RDMs across systems (model vs brain). The loss is 1−Pearson(model_rdm, target_rdm).

**Task Bridge:** Maps 27 simulator task IDs to JIGSAWS task families via transfer plausibility scores (needle_driving=0.95, needle_control=0.85, other=0.45). See `src/eeg_eye_bridge/phase3_rdm/task_bridge.py`.

**Centroid RDM:** During training, embeddings are grouped by condition (e.g., task), averaged to centroids, then pairwise distances give a model RDM. See `src/modules/brain_rdm.py → compute_task_centroid_rdm()`.

**Gromov–Wasserstein (GW):** Optimal transport between two metric-measure spaces using only *intra-set* distances — no cross-set correspondence required. Right tool for comparing JIGSAWS and EEG/Eye when they share no trial identity. Entropic GW (`ot.entropic_gromov_wasserstein`, regularization `epsilon`) is used on the full NJ×NE trial-level coupling; plain `ot.gromov_wasserstein` on the 3×3 tier-centroid RDMs.

**Block-mass diagonality (`diag_mass`):** Aggregate an N×N coupling into a 3×3 block-mass matrix by tier, then read `trace(B)`. Under a tier-shuffle null with balanced tiers, `E[diag_mass] = 1/3`; observed `diag_mass >> 1/3` means the coupling respects tier boundaries. Used as the trial-level null statistic because GW is invariant to row/column permutations of its distance matrices, so shuffling trial order alone is a degenerate null.

## Dependencies

Python 3.11+. Key packages: torch 2.7+, torchvision, timm, numpy (<2.0 for torch compat), scipy, pandas, scikit-learn, mne (EEG), opencv-python, pyyaml, tqdm, matplotlib, seaborn, **POT** (`pot`, for Gromov–Wasserstein in `src/skill_manifold/gw.py`). See `requirements.txt`.

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

**Run a quick smoke test:** `python pipeline/run_full_pipeline.py --phase1-synthetic --skip-train`

**Train a single fold:** `python src/training/train_vit_system.py --config src/configs/baseline.yaml --task Suturing --split fold_1 --data_root . --output_dir checkpoints/test`

**Run the skill-manifold GW comparison:** `python pipeline/skill_manifold_gw.py` (add `--smoke` for a 10% pass or `--n_perms 200` for a faster null). Outputs `reports/skill_manifold/report_comparison_B.md` + plots.

**Add a new trial-level modality split:** extend `mimic_modality_columns()` or `jigsaws_modality_columns()` in `src/skill_manifold/features_*.py`, then `modality_split_analysis` / `jigsaws_modality_split_analysis` in `src/skill_manifold/trial_null.py` automatically picks up the new subset.

**Change GW knobs for a sweep:** edit `src/configs/skill_manifold.yaml` (`n_perms`, `gw_epsilon`, `modality_split_n_bootstraps`, …) or pass `--n_perms` / `--subsample_per_tier` on the CLI.
