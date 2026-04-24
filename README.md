# Surgical Gestures — ViT, Kinematics, and EEG–Eye Brain Alignment

Vision Transformer–based models for **JIGSAWS** surgical video: predict **robot kinematics**, **gesture** labels (15 classes), and **skill** level (Novice / Intermediate / Expert), with optional **neural alignment** via an **EEG–Eye Bridge** (RDM targets, RSA loss). Evaluation follows **8-fold Leave-One-User-Out (LOUO)** across surgeons.

A parallel **Skill-Manifold Comparison** pipeline (Gromov–Wasserstein between JIGSAWS and the EEG/Eye simulator) sits alongside the trainer and asks whether the two datasets organize surgical skill into compatible geometries, with permutation nulls, stratified bootstraps, and per-modality splits.

**Documentation in this repository:** [`README.md`](README.md) (this file) and [`CLAUDE.md`](CLAUDE.md) (architecture, configs, EEG–eye phases, conventions). Deeper or personal notes can live in a local `docs/` directory if you create one; it is gitignored by default.

---

## Requirements

- **Python 3.11+**
- **PyTorch** — `pip install -r requirements.txt` installs pinned versions from PyPI. **Default PyPI wheels are CPU-only.** For an NVIDIA GPU, reinstall CUDA-enabled torch/torchvision (see the comment block at the top of [`requirements.txt`](requirements.txt)), e.g.:

  ```bash
  pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
  ```

- **`PYTHONPATH=src`** for all Python entrypoints (the shell tips below set this).

---

## Setup

```bash
cd Surgical_Gestures
python -m venv .venv          # or: conda create -n surgical_gestures_venv python=3.11 -y && conda activate surgical_gestures_venv
pip install -r requirements.txt
# If using GPU: reinstall torch/torchvision with the CUDA index (see above).
export PYTHONPATH=src
```

**Windows (PowerShell):**

```powershell
cd path\to\Surgical_Gestures
conda activate surgical_gestures_venv   # if using conda
$env:PYTHONPATH = "src"
python pipeline/generate_splits.py
```

---

## Data layout

Place the **JIGSAWS**-style tree under this repository (video, kinematics, transcriptions). Typically:

- **`Gestures/`** — per-task folders with `video/`, `kinematics/`, `transcriptions/`, meta files (see [`CLAUDE.md`](CLAUDE.md)).
- **`EEG/`, `Eye/`** — needed only for the EEG–eye bridge phases (large; not committed to git).

Large assets are listed in [`.gitignore`](.gitignore). **`data/splits/`** — generate LOUO split files locally:

```bash
python pipeline/generate_splits.py
```

---

## Train a single fold (main entrypoint)

```bash
export PYTHONPATH=src
python src/training/train_vit_system.py \
  --config src/configs/baseline.yaml \
  --data_root . \
  --task Knot_Tying \
  --split fold_1 \
  --output_dir checkpoints/knot_fold1
```

Useful configs include `baseline.yaml`, `brain_eye.yaml`, `bridge_eeg_rdm.yaml`, `bridge_joint_eye_eeg.yaml`. **`--task`** can be `Knot_Tying`, `Needle_Passing`, `Suturing`, or `all` where supported.

---

## 8-fold LOUO (bash)

Requires a Unix-like shell (Git Bash, WSL, or Linux/macOS).

```bash
export PYTHONPATH=src
./run_8fold_louo.sh                          # all tasks, folds 1–8, baseline config
./run_8fold_louo.sh Knot_Tying 1 3         # one task, folds 1–3
./run_8fold_louo.sh all 1 8 src/configs/brain_eye.yaml   # custom config
```

Brain-oriented 8-fold helper:

```bash
./run_8fold_louo_brain.sh
```

The script trains each fold, runs [`src/eval/evaluate.py`](src/eval/evaluate.py) on the test split, and expects to populate metrics under `eval_results/` for downstream aggregation.

---

## Aggregate LOUO metrics

```bash
python pipeline/aggregate_louo_results.py
```

Defaults: read `eval_results/` for files matching `<Task>_test_fold_*.txt`, write `louo_summary.txt` and `louo_results.json`. If no files match, the script reports that and exits.

---

## EEG–Eye Bridge (full pipeline)

Runs Phase 1 (EEG export) → Phase 2 (eye consistency) → Phase 3 (RDMs + manifest) → optional Phase 4 (ViT training with bridge config):

```bash
export PYTHONPATH=src
python pipeline/run_full_pipeline.py
```

Or run interactively via [`pipeline/pipeline.ipynb`](pipeline/pipeline.ipynb) (phase-by-phase with visualizations).

Smoke test (synthetic Phase 1, no training):

```bash
python pipeline/run_full_pipeline.py --phase1-synthetic --skip-train
```

Caches go under `cache/eeg_eye_bridge/`. Phase details and brain modes are documented in [`CLAUDE.md`](CLAUDE.md).

---

## Ablations

| Script | Role |
|--------|------|
| [`run_ablation_study.ps1`](run_ablation_study.ps1) | Windows: four configs × three tasks × eight folds → `checkpoints/<config>/<task>/fold_n/` |
| [`misc/run_ablations.sh`](misc/run_ablations.sh) | Unix (legacy): focused ablations (`brain`, `lambda`, `finetune`, `layers`, or `all`) into `checkpoints/ablations/` |

Arbitrary **hyperparameter sweeps** are driven through [`pipeline/run_hparam_sweep.py`](pipeline/run_hparam_sweep.py), which takes a base YAML and a JSON sweep spec (e.g. `'{"loss_weights.brain": [0.01, 0.05, 0.1]}'`) and writes ranked val-metric summaries alongside each run's `best_model.pth`.

---

## Skill-manifold comparison (Gromov–Wasserstein)

A separate analysis pipeline that compares the per-trial skill geometry of JIGSAWS against the EEG/Eye simulator. It produces a headline GW distance, a tier-shuffle permutation null, OSATS per-axis breakdowns, trial-level block-diagonality tests with stratified bootstraps, and per-modality splits on both sides (`eeg_baseline` / `eeg_predictive_coding` / `eye` on the Mimic side; `gestures` / `kinematics` on the JIGSAWS side). Two framings:

- **Comparison B — skill manifold:** JIGSAWS tiered by OSATS `grs_total`, Mimic by `performance`.
- **Comparison A — practice manifold:** JIGSAWS tiered by self-reported E/I/N, Mimic by per-subject experience tertile.

```bash
export PYTHONPATH=src
python pipeline/skill_manifold_gw.py                  # full run
python pipeline/skill_manifold_gw.py --smoke          # 10% subsample smoke pass
python pipeline/skill_manifold_gw.py --n_perms 200    # faster null
```

Outputs land in [`reports/skill_manifold/`](reports/skill_manifold): `report_comparison_B.md`, `results_comparison_B.json`, and plots under `plots/`. An interactive companion is [`pipeline/skills_manifold.ipynb`](pipeline/skills_manifold.ipynb). Knobs (seed, `n_perms`, `gw_epsilon`, subsample / bootstrap counts, gesture pool, OSATS axes) live in [`src/configs/skill_manifold.yaml`](src/configs/skill_manifold.yaml); the 27-task-ID → 9-module mapping is in [`src/configs/skill_manifold_task_modules.yaml`](src/configs/skill_manifold_task_modules.yaml).

See [`CLAUDE.md`](CLAUDE.md#skill-manifold-comparison-gromovwasserstein) for the full stage-by-stage description. The Step 10 design is critiqued in [`pipeline/skills_manifold_modality_split_postmortem.txt`](pipeline/skills_manifold_modality_split_postmortem.txt).

---

## Post-hoc representation analyses

Probe what a trained checkpoint actually encodes:

```bash
# Skill-separability metrics (within vs between, silhouette) on one fold
python pipeline/skill_manifold_analysis.py \
    --checkpoint checkpoints/brain_eye/all/fold_1/best_model.pth \
    --data_root . --task all --split fold_1

# Ridge + k-NN probes aggregated across folds (skill, gesture, task, surgeon)
python pipeline/representation_probe.py \
    --aggregate_root checkpoints/brain_eye/all \
    --data_root . --task all --split_family louo \
    --output_dir analysis/representation_probe --stem brain_eye
```

Surgeon probe is interpreted as *lower is better* — a high surgeon accuracy means the model memorized motor style rather than generalizing across operators.

---

## EEG–eye bridge tests

```bash
export PYTHONPATH=src
python tests/eeg_eye_bridge/phase1/test_phase1_eeg.py
python -m pytest tests/eeg_eye_bridge/phase2/test_phase2_eye_consistency.py -v
python -m pytest tests/eeg_eye_bridge/phase3/test_phase3_rdms.py -v
python tests/eeg_eye_bridge/phase4/test_phase4_vit_regularizer.py
python tests/eeg_eye_bridge/test_phase5_integration_coordinator.py
```

## Skill-manifold tests

```bash
export PYTHONPATH=src
python -m pytest tests/skill_manifold -v
```

---

## Citation

```bibtex
@software{surgical_gestures,
  title={ViT-Based Surgical Gesture Recognition and Kinematics Prediction for dVRK},
  author={Michael Haidar, Mai Bui, Aaron Liu},
  year={2025},
  url={https://github.com/Mhaidar117/Surgical_Gestures}
}
```

JIGSAWS:

```bibtex
@article{jigsaws,
  title={JIGSAWS: A Dataset for Surgical Skill Assessment},
  author={Gao, Yixin and others},
  journal={arXiv preprint arXiv:1506.04112},
  year={2015}
}
```

## License

See [LICENSE](LICENSE) if present in the repository.
