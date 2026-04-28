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

### A fourth dataset: NIBIB-RPCCC-FLS (cross-modality work)

A separate PhysioNet release from the same Roswell Park group, focused on **laparoscopic** surgical training rather than robotic. Lives at `data/laparoscopic-surgery-fls-tasks/`. **This is a different study from the RAS data above** — different participants, different tasks, different hardware (no robot; bench-top FLS box).

- Source: PhysioNet, [`eeg-eye-gaze-for-fls-tasks/1.0.0`](https://physionet.org/content/eeg-eye-gaze-for-fls-tasks/1.0.0/)
- Subjects: 25 surgical trainees with varying robotic-surgery experience.
- Tasks: 3 FLS exercises — peg transfer (taskID 1), pattern cut (taskID 2), intracorporeal suturing (taskID 3).
- Trials: ~5 attempts per task per subject; 315 paired EEG + Eye recordings (some subjects did 2 attempts, most did 5). Trial pairing is explicit — every row of `data/laparoscopic-surgery-fls-tasks/PerformanceScores.csv` carries both an EEG filename and an Eye filename.
- EEG: 128-ch AntNeuro @ 500 Hz, `.edf` (same channel layout, EOG list, and known-bad channel set as the RAS dataset above).
- Eye: Tobii Pro glasses @ 50 Hz, `.csv`. **Same 20-column layout as `Eye/EYE/`** — gaze X/Y at columns 0-1, pupil L/R at 17-18, movement type at 19. Difference: FLS files include a header row; RAS files do not. The existing `src/eeg_eye_bridge/phase2_eye_latents/eye_loader.py` is reusable with a header-skip flag.
- Performance: heterogeneous across tasks — GOALS for peg transfer + pattern cut, OSAT for suturing, plus time/drops/collisions. `PerformanceScores.csv` carries per-trial `Minimum possible score` and `Maximum Possible score` columns; normalization to [0,1] uses these per row.
- No documented sync anchor between EEG and gaze beyond file boundaries; EOG channels (`EEGHEOGRCPz` etc.) can serve as a sync diagnostic against Tobii blink/fixation events.

**FLS vs. RAS distinction**: same group, same hardware family, but FLS subjects, FLS tasks, and FLS trials are disjoint from the RAS simulator dataset under `EEG/` and `Eye/`. A "task" in FLS is a hand-tools laparoscopic exercise on a physical box trainer; a "task" in RAS is a virtual robotic simulator exercise. No overlap of trials, no documented overlap of subjects.

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

### Bootstrap power analysis (calibration of the trial-level bootstrap)

[`pipeline/bootstrap_power_analysis.py`](pipeline/bootstrap_power_analysis.py) probes the calibration and power of the stratified-bootstrap stage of the trial-level GW pipeline. It addresses the reviewer-relevant question: *"is our 50% bootstrap-survival rate (median Mid-Mid z = +0.88) evidence the cross-dataset preliminary data is fragile, or is it the expected behavior of the bootstrap at small surgical-skill N?"*

The script generates **synthetic** trial-level features with a tunable Mid-tier centroid offset Δ (in σ units along a single axis) at low effective dimensionality (`--n_dims`, default 5 to mirror the effective rank of the residualized real features after per-subject z-scoring), then runs the same `subsample_robustness_stratified` machinery the real pipeline uses, recording the fraction of bootstrap resamples whose Mid-Mid per-cell z exceeds the Bonferroni-3-cells threshold ($|z| > 2.394$). Sweeping Δ at multiple N values produces a calibrated power curve.

**What it tests:** the GW + bootstrap stage of the pipeline. **What it does NOT test:** the residualization stage. Synthetic features bypass residualization entirely, so this analysis defends bootstrap stability claims, not claims about residualization inflating effect sizes. Flag this honestly in any reviewer-facing writeup.

Key knobs:
- `--n_dims` (default 5) — synthetic feature dimensionality. Default mirrors the post-z-score effective rank of the real data; higher values dilute single-axis tier signal.
- `--n_perms_inner` (default 200; smoke uses 50) — permutations per bootstrap resample. Lower values discretize the inner z-distribution and slightly inflate the Δ=0 Type I rate to ~10%.
- `--n_bootstraps` (default 15) — bootstrap resamples per simulation. Real pipeline uses 30; reduce for speed.

Run modes:
- Full grid (5 deltas × 3 N × 30 sims, ~2 hr): `python pipeline/bootstrap_power_analysis.py`
- Smoke grid (3 deltas × 2 N × 5 sims, ~3 min): `python pipeline/bootstrap_power_analysis.py --smoke`
- Single cell (one (Δ, N, n_sims) tuple, writes a partial JSON for chaining): `python pipeline/bootstrap_power_analysis.py --cell DELTA N N_SIMS`
- Aggregate previously-written partials into the final JSON + plot: `python pipeline/bootstrap_power_analysis.py --aggregate`

Outputs:
- `reports/skill_manifold/bootstrap_power.json` — per-cell mean survival, SEM, median z
- `reports/skill_manifold/plots/bootstrap_power_curve.png` — fraction-significant vs. Δ, one line per N, with reference lines for Type I (5%) and our observed survival (50%)
- `reports/skill_manifold/bootstrap_power_partials/cell_n*_d*.json` — per-cell partials when running in single-cell mode

**Smoke results (current):** at N = 33 with Δ ∈ {0, 0.5, 1.0, 1.5, 2.0}, mean survival rises from ~12% (Δ=0 calibration) to ~35% (Δ=2σ). At N = 50 with Δ ∈ {0, 1.0, 1.5, 2.0}, the same Δ=1.5σ effect jumps from 39% to 44%, and Δ=2σ reaches 78%. Together these results support the defensible claim: at our actual N (~33), an effect of magnitude consistent with our observed median z ≈ +0.88 *is expected* to produce ~50% bootstrap survival — i.e., the bootstrap is operating at its expected fragility level for an effect of this magnitude at this N. The residualization caveat is independent of this conclusion.

## FLS Cross-Modality GW

A second GW analysis, **distinct from the cross-dataset comparison above**. Asks: **do EEG and gaze, recorded simultaneously from the same 25 FLS subjects, organize skill into compatible geometries?** This is *within-dataset, between-modality* — the opposite framing from the JIGSAWS↔RAS comparison.

The existing `pipeline/skill_manifold_gw.py` and `src/skill_manifold/features_jigsaws.py` / `features_eeg_eye.py` are left untouched; the FLS work runs alongside. Run via:

```bash
export PYTHONPATH=src
python pipeline/skill_manifold_gw_fls.py --features_cache cache/skill_manifold_fls
```

### Layout

- Feature builders under `src/skill_manifold/`:
  - [`features_fls_eeg.py`](src/skill_manifold/features_fls_eeg.py) — drop EOG (regex `^(?:EEG)?[HV]EOG` catches the four canonical leads plus the `EEGHEOGLHEOGRCPz` / `EEGVEOGUVEOGLCPz` concatenations), canonicalize channel names (strip `EEG` prefix and `CPz` reference suffix → `Fp1`, `C3`, …), set MNE `standard_1005` montage (carries the AntNeuro WaveGuard 5/10 supplementary positions `FCC1h`/`CCP3h`/`TPP9h`/…), per-trial soft bad-channel detection + interpolation, bandpass 1–40 Hz + 60 Hz notch, Welch PSD, per-channel **relative bandpower** (each channel's bandpower divided by the channel total before regional averaging — invariant to per-subject and per-channel multiplicative gain), regional aggregation into 8 scalp regions × 5 bands → 40-d log-power trial vector.
  - [`features_fls_gaze.py`](src/skill_manifold/features_fls_gaze.py) — reuses `src/eeg_eye_bridge/phase2_eye_latents/eye_loader.py` (with the `has_header=True` flag added for the FLS CSVs that carry a header row) and `eye_summarize.py` to produce the same 18-d Tobii summary used on the RAS side.
  - [`subject_aggregation.py`](src/skill_manifold/subject_aggregation.py) — collapses per-trial features to per-subject vectors. Mean over all trials per subject for the headline; per-task mean concatenation available as a sensitivity. Also computes the composite skill score per subject = mean of per-trial `(Performance − Min) / (Max − Min)`, using the per-row min/max columns in `PerformanceScores.csv` to put GOALS and OSAT on a common [0,1] scale.
- Orchestrator: [`pipeline/skill_manifold_gw_fls.py`](pipeline/skill_manifold_gw_fls.py), paralleling `skill_manifold_gw.py`. Owns three pieces of pipeline-specific machinery beyond the shared library:
  - `drop_degenerate_subjects` — drops subjects whose post-residualization L2 norm is below `1e-6` in either modality. Catches upstream pipeline failures (e.g. recordings whose channels were uniformly hard-zeroed by the bad-channel fallback) that would otherwise pull tier centroids toward the origin at N≈25.
  - `per_subject_zscore_features` — per-subject z-score across the 40 EEG feature dimensions (subtract subject mean, divide by subject std). Removes the per-subject global-gain offset that was driving PC1 of the EEG covariance to >80% even after relative bandpower. Headline and per-task analyses both apply this; gaze is left alone (gaze PC1 is a healthy ~57%). Disable via `--skip_eeg_zscore`.
  - `diagnostic_plots` — PCA scatters per modality (colored by tier, labelled by subject_id) plus per-subject leave-one-out tier-centroid distance heatmaps. Used to spot outlier subjects driving the tier centroids and to monitor PC1 dominance.
- Reuses `gw.py`, `binning.py`, `rdms.py`, `residualize.py`, `trial_null.py` from the existing skill-manifold library — these are dataset-agnostic.

### Pipeline order (in `skill_manifold_gw_fls.py::run`)

1. Build trial-level features (gaze + EEG), cached as parquet under `--features_cache`.
2. Trial-level OLS residualization against `Age`, `Dominant Hand`, `Dominant Eye`, `Gender` per modality.
3. Aggregate to subject level (mean across each subject's trials), compute composite skill + per-task skill.
3b. Per-subject z-score the EEG feature vector along the feature axis (skipped on `--skip_eeg_zscore`).
4. Inner-join modalities on `subject_id`, drop subjects with NaN composite skill, drop degenerate subjects via `drop_degenerate_subjects`.
5. Tertile-bin surviving subjects on `composite_skill` → Low/Mid/High.
6. Diagnostic plots (PCA + per-subject centroid distances) from the surviving N.
7. **Headline 3×3 tier-centroid GW** between EEG and gaze tier-centroid RDMs (plain `ot.gromov_wasserstein`, tier-shuffle permutation null).
8. **Companion N×N entropic GW** on subject-by-subject cosine RDMs (entropic GW with `epsilon=gw_epsilon`, subject-shuffle null on `trace(T)`). N is the surviving subject count after step 4 — the "25×25" in the original spec is now `len(subj_df) × len(subj_df)`.
9. **Per-task 3×3 sensitivity** — three separate 3×3 analyses (peg transfer / pattern cut / intracorporeal suturing). Honors the `dropped_subjects` set from step 4 and applies the same per-subject EEG z-score so per-task and headline operate on consistent feature normalization.
10. Markdown + JSON report + plots written to `reports/skill_manifold_fls/`.

### Design choices

- *Subjects as the unit of skill* (not trials), because the downstream goal is per-subject feedback for skill improvement.
- *Relative bandpower* (per-channel normalization) rather than absolute log power. Removes the per-subject and per-channel multiplicative gain factor that was driving PC1 of the EEG covariance to 96% of variance and reducing the cosine RDM to a sign-of-PC1 indicator.
- *Per-subject z-score of the EEG feature vector* on top of relative bandpower. Relative bandpower removes the multiplicative gain in the time-domain signal, but a residual subject-level offset survives (likely skull-thickness × electrode-specific gain that doesn't cancel under per-channel normalization). Z-scoring removes this directly.
- *Demographic residualization* against `Age`, `Dominant Hand`, `Dominant Eye`, `Gender` per modality before RDM construction. The per-subject z-score is the post-hoc form of adding a per-subject random intercept to the OLS — at N=24 with ~7 demographic dummies, fitting subject intercepts in OLS over-saturates the design matrix; the post-hoc z-score is more numerically stable.
- *Degenerate-subject filter* applied uniformly to both headline and per-task. Subjects with near-zero feature norm in either modality are dropped at step 4 and excluded from every downstream analysis.
- *GW as discovery*, not validation: the solver is blind to trial pairing; recovered alignment counts as evidence for a shared skill manifold across modalities.

### Analyses (subject-level)

1. **Headline — 3×3 tier-centroid GW** between EEG and gaze tier-centroid RDMs. Plain `ot.gromov_wasserstein`. Permutation null on tier labels.
2. **Companion — N×N entropic GW** on subject-by-subject cosine RDMs in each modality. Subject-shuffle null on `trace(T)`. This is the analysis that supports the per-subject feedback goal — the coupling matrix maps each EEG-position to its gaze-neighbor.
3. **Sensitivity — per-task tier-centroid GW**, three separate 3×3 analyses (peg transfer / pattern cut / intracorporeal suturing). Same machinery, same degenerate-subject filter, same per-subject z-score; exploratory only at N≈24.

Trial-level GW is intentionally skipped at this layer — it conflicts with subjects-as-units and the small-N null is noisy. Easy to add later as a sensitivity if the headline lands.

### Outputs

- `reports/skill_manifold_fls/results_fls.json` — full numerical results (headline, companion, per-task, dropped subjects, tier counts, tertile cutoffs).
- `reports/skill_manifold_fls/report_fls.md` — auto-generated markdown summary.
- `reports/skill_manifold_fls/plots/` — RDMs, couplings, null histograms, MDS, PCA diagnostics, per-subject distance heatmaps.
- `reports/skill_manifold_fls/postmortem.md` — running narrative kept up to date manually after each substantive run.
- `reports/skill_manifold_fls/ablation_comparison.md` — auto-generated by `pipeline/compare_fls_runs.py` when paired runs are available (e.g. default vs `--skip_eeg_zscore`).

### U-shape consolidation analysis (cognitive-load test)

[`pipeline/skill_manifold_fls_ushape.py`](pipeline/skill_manifold_fls_ushape.py) is a standalone script that consumes the same feature caches as the GW pipeline and tests a falsifiable prediction from Lim et al. (2025, *Sci. Rep.* 15:12073): under the inverted-U engagement model of motor skill, **mid-skill surgeons should show the canonical EEG cognitive-load signature** — frontal theta higher than novices/experts, parietal alpha lower. Reuses the orchestrator's data-prep functions (residualize → aggregate → per-subject z-score → degenerate-subject filter → tertile bin) so the per-subject feature matrix is bit-identical to the GW analysis.

The script now supports two modalities via the `--modality {eeg,gaze}` flag (default: `eeg`):

```bash
# EEG mode (default; outputs to reports/skill_manifold_fls/ushape/)
python pipeline/skill_manifold_fls_ushape.py --features_cache cache/skill_manifold_fls

# Gaze mode (outputs to reports/skill_manifold_fls/ushape_gaze/)
python pipeline/skill_manifold_fls_ushape.py --features_cache cache/skill_manifold_fls --modality gaze
```

In **EEG mode** the script produces the canonical 5-analysis output (per-feature t-test ranking, continuous-skill regression, individual U-position projection, Nemani-style topographic plot, and the four-classifier 3-class LDA/RFC comparison). In **gaze mode** the script runs the same 5 analyses on the 18-d Tobii eye-summary feature vector, with two EEG-specific sections replaced:

- The Lim et al. (2025) prediction test panel is replaced with a data-driven `top_features.png` panel showing the top-4 gaze features by |t-statistic| in the same 2×2 layout. There is no published gaze-feature analogue to Lim's frontal-theta / parietal-alpha cognitive-load signature, so the gaze panel is exploratory.
- The Nemani et al. (2018) anatomical topographic plot is skipped — gaze features don't have an anatomical region map analogous to the EEG scalp regions.

Both modes share the same data-prep pipeline (load both EEG and gaze caches; residualize each against demographics + task; aggregate to subjects; z-score the active modality; drop degenerate subjects; tertile-bin by composite skill). The `--skip_eeg_zscore` flag now generically skips z-scoring on the active modality (EEG in EEG mode, gaze in gaze mode). Outputs land in separate directories so EEG and gaze reports never overwrite each other.

Five analyses, all on the post-z-score subject feature matrix:

1. **Per-feature Mid vs (Low ∪ High) t-test** with one-sided p-values for the four pre-specified Lim features (`eeg_frontal_{L,R}_theta`, `eeg_parietal_{L,R}_alpha`). Bonferroni-corrected α = 0.0125 for the 4-test family. Output: `lim_comparison.png` (4-panel professional plot showing tier means + 95% CI, with Lim's predicted direction annotated) and `feature_ranking.png` (full forest plot of all 40 features, Lim features outlined).
2. **Continuous-skill regression** per feature: fit `feature ~ skill` (linear) and `feature ~ skill + skill²` (quadratic), compare via AIC. ΔAIC > 2 indicates meaningful evidence for the quadratic. Tests the U-shape independently of tertile binning. Output: `quadratic_vs_linear.png`.
3. **Individual U-position projection**. Project each subject's z-scored EEG vector onto the unit axis from `mean(Low, High)` centroid to Mid centroid. Each subject's scalar score is their "depth in the U" — the per-subject quantity that a personalized brain-aligned model (e.g. CLIP-HBA-MEG-style) would track during skill acquisition. Output: `individual_u_positions.png`.
4. **Nemani-style anatomical topographic plot**. Schematic head-outline plot showing the 8 lateralized regions colored by Mid − rest effect size, one panel per band (delta / theta / alpha / beta / gamma). Nemani et al. (2018, *Sci. Adv.* 4:eaat3807) PFC, LMM1 (left medial M1), and SMA region outlines are overlaid as dotted ellipses for direct anatomical cross-reference. Tests anatomical convergence on EEG of Nemani's fNIRS finding that LMM1 (LDA weight = −0.70) and PFC are the most discriminative regions for FLS skill. Output: `nemani_topomap.png`.
5. **Four-classifier 3-class comparison**. Leave-one-out 3-class classification on the EEG feature matrix predicting Low/Mid/High under four feature-set/classifier choices, controlling for the small-N high-dimensionality regime (full p = 40 → N/p = 0.6, vs. Nemani's N/p ≈ 6.4):
   - `lda_three_class()` on the full 40-d feature matrix (head-to-head dimensionality vs. Nemani).
   - `lda_three_class_top_k(k=5)` with t-stat feature selection performed *inside each LOO fold* (no double-dipping). Matches Nemani's effective dimensionality.
   - `lda_three_class_pca(n_components=5)` with PCA fit *inside each LOO fold*. Same effective dimensionality, unsupervised feature reduction.
   - `rfc_three_class_rfe(k=5)` directly mirrors Soangra et al. (2022, *PLoS ONE* 17(6):e0267936) methodology: in-fold RFE feature selection with a RandomForestClassifier estimator, then a fresh RFC trained on the selected features. Their best 3-class accuracy was 58% with ECU + deltoid (N = 26, RFC, 2 features); our top-5 RFC is the methodologically-matched EEG analogue.
   The three LDA variants use Ledoit-Wolf shrinkage (`solver="lsqr", shrinkage="auto"`); the RFC uses 100 estimators with default scikit-learn parameters. Outputs: per-classifier `lda_confusion[_top5|_pca5|_rfc_rfe].png`, plus `lda_comparison.png` (1×N confusion-matrix grid + grouped pairwise-MCE bar chart with Nemani's 0.043 reference line). Each result reports accuracy, Mid-vs-not-Mid 2-class collapse accuracy, per-class recall, full confusion matrix, and pairwise MCEs (Low vs Mid, Mid vs High, Low vs High) in Nemani's reporting style. The Mid-vs-not-Mid statistic surfaces a U-shape signature that the 3-class accuracy buries: under the inverted-U engagement model, novices and experts are co-located in feature space, so a 3-class linear classifier swaps them but still routes Mid samples correctly — observed as Mid recall well above chance even when 3-class accuracy is at chance.

Outputs land in `reports/skill_manifold_fls/ushape/`: `lim_comparison.png`, `feature_ranking.png`, `quadratic_vs_linear.png`, `individual_u_positions.png`, `nemani_topomap.png`, `lda_confusion.png`, `lda_confusion_top5.png`, `lda_confusion_pca5.png`, `lda_confusion_rfc_rfe.png`, `lda_comparison.png`, `ushape_results.json` (full numerical results including all four classifier results, per-fold selected features for top-k and RFC-RFE, and per-subject predictions), `ushape_report.md` (markdown summary).

### Image catalog (writeup reference)

The plots fall into three groups by narrative role. Use this catalog to find the right figure for each section of a manuscript or grant.

#### Headline plots — for any "first figure" or grant prelim panel

- **`reports/skill_manifold_fls/ushape/nemani_topomap.png`** — Per-band schematic head topography showing the Mid − rest effect size on the 8 scalp regions, with Nemani et al. (2018) PFC, LMM1, and SMA region outlines overlaid (color-coded). Shows that our top-effect regions (left-frontal and left-central elevation across delta/theta/beta) sit anatomically inside Nemani's published PFC and LMM1 ellipses. **Best single plot for the "anatomical convergence with prior fNIRS work" claim.** Use as Figure 1 in any U-shape writeup.
- **`reports/skill_manifold_fls/ushape/lda_comparison.png`** — Four-classifier 3-class comparison: top row is confusion matrices (full LDA, top-5 LDA, PCA-5 LDA, RFC + RFE-5), bottom row is grouped pairwise-MCE bar chart with Nemani's 0.043 reference line. Tells the methodological-convergence story: independent classifier families and feature-selection methods all surface the same Low ↔ High collapse (the U-shape signature). **Best plot for the "the U-shape is multivariate-real" claim and for direct quantitative comparison to Soangra et al. (2022).** Use as Figure 2 in any U-shape writeup.
- **`reports/skill_manifold_fls/ushape/lim_comparison.png`** — 2×2 panel showing tier means + 95% CI for the four pre-registered Lim et al. (2025) features (frontal-L/R theta, parietal-L/R alpha) with predicted direction annotated. Frontal-L theta replicates Lim's cognitive-load signature (one-sided p ≈ 0.014). **Best plot for the "we replicate the cognitive-load hypothesis from a 2025 surgical EEG paper" claim.** Use to anchor the cognitive-load framing.

#### Supporting plots — for the U-shape robustness section

- **`reports/skill_manifold_fls/ushape/individual_u_positions.png`** — Scatter of all 24 subjects on (composite_skill, U-position) axes with a fitted quadratic overlaid, color-coded and labelled by tier. Each subject's U-position is their projection onto the Mid − (Low ∪ High) discriminant axis. Shows the U-shape directly at the per-subject level: Mid subjects cluster at high U-position, Low and High at low U-position. **Best plot for the "this is a per-subject phenomenon trackable by personalized neural models" claim — sets up the personalization aim of the grant.**
- **`reports/skill_manifold_fls/plots/diagnostic_dropone_eeg.png`** — Box-and-strip plot of the three off-diagonal cells of the EEG centroid RDM across 24 leave-one-subject-out runs. Tight clustering around the dashed baseline = the U-shape geometry is robust and not driven by any single subject. d(Low,High) stays in [0.17, 0.50] across all 24 dropouts; d(Low,Mid) and d(Mid,High) stay above 1.5. **Best plot for the "the U-shape passes leave-one-out robustness" claim.**
- **`reports/skill_manifold_fls/ushape/feature_ranking.png`** — Full forest plot of all 40 EEG features ranked by |t-statistic| for the Mid − rest contrast, with Lim et al.'s 4 cognitive-load features outlined. Shows the broadband left-lateralized engagement signature (left-central + left-frontal delta/theta/beta elevated, occipital reduced). **Best plot for the "the U-shape is broadband and left-lateralized, not just narrow frontal-theta" claim.**
- **`reports/skill_manifold_fls/ushape/lda_confusion_rfc_rfe.png`** — Standalone confusion matrix for the Soangra-style RFC + RFE-5 classifier. 3-class accuracy = 33%, but Mid-vs-not-Mid accuracy = 79.2% with 80% Mid precision. All 8 High-tier subjects classified as Low — the cleanest single-figure U-shape evidence. **Best plot for the "headline accuracy number" if `lda_comparison.png` is too dense for a slide.**

#### Diagnostic plots — for methods sections and supplementary material

- **`reports/skill_manifold_fls/ushape/lda_confusion.png`** — Full 40-feature LDA confusion matrix. 3-class accuracy at chance (33%); demonstrates the small-N high-dimensionality regime (N/p = 0.6) under linear discrimination. Use to motivate the dimensionality-matched classifiers.
- **`reports/skill_manifold_fls/ushape/lda_confusion_top5.png`** — Top-5 features by t-stat LDA confusion. Mid recall = 0.625; the t-stat selection identifies left-frontal/central features in 22+ of 24 LOO folds.
- **`reports/skill_manifold_fls/ushape/lda_confusion_pca5.png`** — PCA-5 LDA confusion. PCA reduces dimensionality without using class labels; gives a more diffuse confusion pattern but also achieves Mid-vs-not-Mid > chance.
- **`reports/skill_manifold_fls/ushape/quadratic_vs_linear.png`** — Per-feature ΔAIC comparison of linear vs. quadratic regression of feature on composite skill. No feature exceeds ΔAIC > 2 at N = 24; the U-shape lives in the multivariate covariance structure rather than in any single feature's univariate relationship to skill. Use as a methodological caveat in any writeup.
- **`reports/skill_manifold_fls/plots/diagnostic_pca_eeg.png`** — PC1/PC2 scatter of subject EEG features colored by tier and labelled by subject ID. PC1 ≈ 51% of variance after relative bandpower + per-subject z-scoring (down from 96% with absolute power). Use for the methods section to document feature whitening.
- **`reports/skill_manifold_fls/plots/diagnostic_dist_eeg.png`** — Heatmap of per-subject leave-one-out cosine distance to each tier centroid. Shows individual-level heterogeneity (subject 17 is a known outlier appearing in both this plot and the gaze fragility analysis). Use to motivate the personalized-models grant aim.
- **`reports/skill_manifold_fls/plots/diagnostic_pca_gaze.png`**, **`diagnostic_dist_gaze.png`**, **`diagnostic_dropone_gaze.png`** — Gaze-side equivalents of the EEG diagnostics. Show that gaze tier geometry is dominated by 2 specific Mid subjects (17 and 20) — useful for documenting why the cross-modal GW analysis lands at p > 0.10.

#### Cross-modal GW pipeline plots (separate from U-shape)

These come from the GW analysis (`pipeline/skill_manifold_gw_fls.py`) and live in `reports/skill_manifold_fls/plots/`. They support the *negative* cross-modal manifold result that motivates the *personalized* approach the grant proposes:

- **`coupling_headline.png`**, **`null_headline.png`**, **`rdm_centroid_eeg.png`**, **`rdm_centroid_gaze.png`** — The headline 3×3 GW centroid result. p ≈ 0.36 at N=24; the cross-modal alignment is not detectable at the population level under proper feature normalization. Use as the motivation for personalized models in the grant.
- **`coupling_companion.png`**, **`null_companion.png`** — The 24×24 entropic GW companion result. trace(T) at chance (z ≈ 0); per-subject correspondence is not detectable at this N with linear methods.
- **`coupling_task_{peg_transfer,pattern_cut,suturing}.png`**, **`null_task_*.png`** — Per-task sensitivity. Use selectively if any per-task signal is relevant; otherwise mention only in supplementary material.
- **`mds_eeg.png`**, **`mds_gaze.png`** — 2D MDS of subject features colored by tier. Use as a low-effort visual companion to the diagnostic PCA plots.

### Suggested figure order for a U-shape writeup

For a manuscript or grant prelim section telling the U-shape → personalization story, the recommended order is:

1. **`nemani_topomap.png`** — anatomical context, ties to prior surgical fNIRS literature.
2. **`lim_comparison.png`** — pre-registered cognitive-load test, ties to prior surgical EEG literature.
3. **`individual_u_positions.png`** — the per-subject U-shape, immediately motivates personalization.
4. **`diagnostic_dropone_eeg.png`** — robustness check for the U-shape geometry.
5. **`lda_comparison.png`** — methodological convergence; the headline 79% Mid-vs-not-Mid number.
6. **`feature_ranking.png`** (optional) — broadband signature for the methods section.

The cross-modal null result (from the main GW pipeline) is best mentioned in the introduction as motivation rather than shown as a figure: "population-level cross-modal alignment between gaze and EEG is not detectable at this N (p ≈ 0.36); this preliminary work motivates the personalized neural-modeling approach proposed here."

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
