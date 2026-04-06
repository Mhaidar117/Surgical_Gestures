# Phase 1 EEG cache contract

Version: `phase1_eeg_v1` (see `CONTRACT_VERSION` in `export.py`).

Root: `cache/eeg_eye_bridge/phase1/` (relative to repo / `--data_root`).

## `manifest.json`

- `contract_version`: string
- `trials`: list of objects, each with at least:
  - `trial_id`, `participant_id`, `task_id`, `task_name`, `task_family`, `performance_score`
  - `trial_pkl`: relative path to `trials/{trial_id}.pkl`
  - `n_windows`, `baseline_embed_dim`, `pc_embed_dim` (optional but recommended)

## `trials/{trial_id}.pkl`

Pickle dict with:

| Key | Type | Description |
|-----|------|-------------|
| `trial_id` | str | `{subject}_{task}_{try}` stem |
| `participant_id` | int | Subject ID |
| `task_id` | int | Simulator task 1–27 |
| `task_name` | str | From `Eye/Table1.csv` |
| `task_family` | str | `needle_control`, `needle_driving`, or `other_nontransfer` |
| `performance_score` | float | 0–100 |
| `window_times` | `np.ndarray` float64, shape `(n_windows, 2)` | `[start_sec, end_sec]` |
| `baseline_embeddings` | `np.ndarray` float32, `(n_windows, D_b)` | Per-window baseline model embedding |
| `pc_embeddings` | `np.ndarray` float32, `(n_windows, D_p)` | Per-window predictive-coding latent |
| `prediction_errors` | `np.ndarray` float32, `(n_windows,)` | Local prediction error (last window may be 0) |
| `contract_version` | str | Same as manifest |

## `family_summaries.pkl`

Pickle dict:

- `contract_version`
- `families`: map `task_family` → object with:
  - `n_trials`, `trial_ids`
  - `mean_baseline_embedding`, `mean_pc_embedding` (numpy float32 vectors)
  - `baseline_dim`, `pc_dim`

All arrays are **numpy** (not torch) for stable pickle loads across versions.
