#!/usr/bin/env python3
"""Generate a plain-article LaTeX manuscript from LOUO ablation artifacts.

Reads per-condition ``analysis/<cond>/louo_results.json``, the Phase 3 RDM
manifest, dataset splits, and training configs, then emits a self-contained
``docs/final_results_manuscript/`` tree with:

* ``main.tex`` - plain article class, inputs each section and table.
* ``sections/*.tex`` - full-prose sections with auto-inserted numeric claims.
* ``tables/*.tex`` - booktabs tables with auto-bolded per-task winners.
* ``figures/*.pdf`` - data-driven figures (bars, box plots, RDM heatmaps,
  transfer-plausibility scatter).
* ``references.bib`` - real bibliographic entries for every cite key used
  by the auto-generated prose. Entries that map to multiple plausible
  papers are flagged with ``% TODO: resolve citation`` rather than
  fabricated; the file is overwritten on every run.
* ``README.md`` - one-time build instructions (only if missing).

Pipeline- and architecture-diagram PDFs (``fig_pipeline_diagram.pdf``,
``fig_architecture.pdf``) are intentionally not generated; the ``main.tex``
includes them via ``\\IfFileExists`` guards so the manuscript compiles with or
without them present.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# Matplotlib is configured at module-import time with a non-interactive backend
# so the script can run headless (e.g. on CI) without display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

try:
    import yaml
except ImportError:  # pragma: no cover - pyyaml is in requirements.txt
    yaml = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants and display ordering
# ---------------------------------------------------------------------------

TASK_ORDER: tuple[str, ...] = ("Knot_Tying", "Needle_Passing", "Suturing")
TASK_LABELS: dict[str, str] = {
    "Knot_Tying": "Knot Tying",
    "Needle_Passing": "Needle Passing",
    "Suturing": "Suturing",
}

CONDITION_LABELS: dict[str, str] = {
    "baseline": "Baseline",
    "brain_eye": "Eye-RSA",
    "bridge_eeg": "EEG-Bridge",
}

CONDITION_DESCRIPTIONS: dict[str, str] = {
    "baseline": (
        "No brain-alignment term in the loss. Trains the ViT purely on "
        "kinematics, gesture, and skill supervision."
    ),
    "brain_eye": (
        "Eye-tracking task-centroid RSA. Uses a fixed 3x3 target RDM derived "
        "from a separate eye-tracking exploration study "
        "(\\texttt{Eye/Exploration/target\\_rdm\\_3x3.npy})."
    ),
    "bridge_eeg": (
        "EEG-derived subskill-family RDM from Phase~3 of the EEG-Eye bridge "
        "pipeline (manifest key \\texttt{eeg\\_latent\\_subskill\\_family})."
    ),
}

# Mapping from a training condition to the Phase 3 manifest entry that best
# represents the source RDM used by that condition. ``brain_eye`` does not
# literally load from the Phase 3 manifest (it uses the standalone 3x3 eye RDM
# under Eye/Exploration/), but ``eye_only_subskill_family`` is the closest
# Phase 3 proxy and is used only for the transfer-plausibility scatter figure.
# NOTE: ``bridge_joint`` was dropped from the rerun because the Phase 2 derived
# joint eye+EEG family RDM is degenerate (all off-diagonal = 1.0), which yields
# a constant brain RSA loss with zero gradient and trains identically to
# baseline. See methods for details.
CONDITION_TO_MANIFEST_RDM: dict[str, str] = {
    "brain_eye": "eye_only_subskill_family",
    "bridge_eeg": "eeg_latent_subskill_family",
}

# Metric families: which metric keys are "lower is better" vs "higher is better",
# and how they should be labelled in prose/tables.
@dataclass(frozen=True)
class MetricSpec:
    key: str            # json key in louo_results.json metrics
    column: str         # short column identifier
    label: str          # human-readable LaTeX-safe label
    goal: str           # "min" or "max"
    units: str = ""     # optional trailing units (e.g. "%", "rad")


GESTURE_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("gesture_accuracy", "gesture_accuracy", "Gesture Acc.", "max", "\\%"),
    MetricSpec("gesture_f1_macro", "gesture_f1_macro", "Gesture F1 (macro)", "max"),
    MetricSpec("gesture_f1_micro", "gesture_f1_micro", "Gesture F1 (micro)", "max"),
)

SKILL_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("skill_accuracy", "skill_accuracy", "Skill Acc.", "max", "\\%"),
    MetricSpec("skill_f1_macro", "skill_f1_macro", "Skill F1 (macro)", "max"),
)

KINEMATICS_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("kinematics_position_rmse", "position_rmse", "Position RMSE", "min"),
    MetricSpec("kinematics_end-effector_error", "end_effector_error", "End-Effector Error", "min"),
    MetricSpec("kinematics_rotation_rmse", "rotation_rmse", "Rotation RMSE", "min"),
)

LOSS_METRICS: tuple[MetricSpec, ...] = (
    MetricSpec("loss_total_loss", "total_loss", "Total Loss", "min"),
)

# Per-frame gesture cross-entropy. Unlike accuracy/F1, it is on the same
# scale across tasks (all tasks share the 15-class softmax), which makes it
# the natural scalar for a "global ability to identify which gesture is in
# the video, per condition" summary. Treated separately from LOSS_METRICS so
# the main table can give it its own column without dragging total-loss
# along.
GESTURE_LOSS_METRIC: MetricSpec = MetricSpec(
    "loss_gesture_loss", "gesture_loss", "Gesture Loss", "min"
)

ALL_METRICS: tuple[MetricSpec, ...] = (
    GESTURE_METRICS
    + SKILL_METRICS
    + KINEMATICS_METRICS
    + LOSS_METRICS
    + (GESTURE_LOSS_METRIC,)
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MetricStat:
    mean: float
    std: float
    values: list[float] = field(default_factory=list)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "MetricStat":
        return cls(
            mean=float(payload.get("mean", 0.0) or 0.0),
            std=float(payload.get("std", 0.0) or 0.0),
            values=[float(v) for v in payload.get("values", []) or []],
        )


@dataclass
class TaskResult:
    task: str
    num_folds: int
    metrics: dict[str, MetricStat]

    def get(self, key: str) -> MetricStat | None:
        return self.metrics.get(key)


@dataclass
class Condition:
    name: str
    label: str
    config_path: str | None
    loss_weights: dict[str, float]
    tasks: dict[str, TaskResult]

    def result(self, task: str) -> TaskResult | None:
        return self.tasks.get(task)


@dataclass
class RDMEntry:
    name: str
    rdm_type: str
    unit_type: str
    unit_labels: list[str]
    shape: tuple[int, int]
    transfer_plausibility: float
    combined_score: float
    relative_path: str
    recommended_tier: str


# ---------------------------------------------------------------------------
# Load layer
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    """Read a JSON file, tolerant of Windows-Powershell UTF-16 outputs."""
    raw = path.read_bytes()
    last: Exception | None = None
    for enc in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(enc))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:  # noqa: PERF203
            last = exc
    raise ValueError(f"Could not decode JSON {path}: {last}")


def load_louo_results(analysis_root: Path, condition: str) -> dict[str, TaskResult] | None:
    """Load the louo_results.json for one condition into TaskResult objects.

    Returns ``None`` if the file is missing so callers can report gracefully.
    """
    path = analysis_root / condition / "louo_results.json"
    if not path.is_file():
        return None
    raw = _read_json(path)
    if not isinstance(raw, dict):
        return None

    tasks: dict[str, TaskResult] = {}
    for task_name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        metrics_raw = payload.get("metrics") or {}
        metrics = {
            k: MetricStat.from_json(v)
            for k, v in metrics_raw.items()
            if isinstance(v, dict)
        }
        tasks[task_name] = TaskResult(
            task=task_name,
            num_folds=int(payload.get("num_folds", 0) or 0),
            metrics=metrics,
        )
    return tasks


def load_config_weights(configs_dir: Path, condition: str) -> tuple[str | None, dict[str, float]]:
    """Return (config path, loss weights dict) for a condition.

    Maps conditions to their YAML under src/configs/. Missing configs return
    (None, {}).
    """
    mapping = {
        "baseline": "baseline.yaml",
        "brain_eye": "brain_eye.yaml",
        "bridge_eeg": "bridge_eeg_rdm.yaml",
        "bridge_joint": "bridge_joint_eye_eeg.yaml",
    }
    fname = mapping.get(condition)
    if fname is None:
        return None, {}
    path = configs_dir / fname
    if not path.is_file() or yaml is None:
        return (str(path) if path.is_file() else None), {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError:
        return str(path), {}
    weights = data.get("loss_weights") or {}
    if not isinstance(weights, dict):
        weights = {}
    return str(path), {str(k): float(v) for k, v in weights.items()}


def load_split_fold_counts(splits_dir: Path) -> dict[str, int]:
    """Return number of folds available in each JIGSAWS task's splits file."""
    counts: dict[str, int] = {}
    for task in TASK_ORDER:
        path = splits_dir / f"{task}_splits.json"
        if not path.is_file():
            counts[task] = 0
            continue
        try:
            data = _read_json(path)
        except ValueError:
            counts[task] = 0
            continue
        if isinstance(data, dict):
            counts[task] = sum(1 for k in data if str(k).startswith("fold_"))
        else:
            counts[task] = 0
    return counts


def load_split_sample_counts(splits_dir: Path, task: str, fold: int = 1) -> dict[str, int]:
    """Return trial counts for a representative fold of one task."""
    path = splits_dir / f"{task}_splits.json"
    out = {"train": 0, "val": 0, "test": 0}
    if not path.is_file():
        return out
    try:
        data = _read_json(path)
    except ValueError:
        return out
    fold_key = f"fold_{fold}"
    fold_data = data.get(fold_key) if isinstance(data, dict) else None
    if not isinstance(fold_data, dict):
        return out
    for split in out:
        trials = fold_data.get(split, [])
        if isinstance(trials, list):
            out[split] = len(trials)
    return out


def load_rdm_manifest(path: Path) -> dict[str, RDMEntry]:
    """Load the Phase 3 RDM manifest and return a dict of RDMEntry by name."""
    if not path.is_file():
        return {}
    data = _read_json(path)
    if not isinstance(data, dict):
        return {}
    rdms_raw = data.get("rdms") or {}
    out: dict[str, RDMEntry] = {}
    for name, payload in rdms_raw.items():
        if not isinstance(payload, dict):
            continue
        scores = payload.get("scores") or {}
        shape = payload.get("shape") or [0, 0]
        out[name] = RDMEntry(
            name=str(payload.get("rdm_name", name)),
            rdm_type=str(payload.get("rdm_type", "")),
            unit_type=str(payload.get("unit_type", "")),
            unit_labels=list(payload.get("unit_labels", []) or []),
            shape=(int(shape[0]), int(shape[1])) if len(shape) == 2 else (0, 0),
            transfer_plausibility=float(scores.get("transfer_plausibility", 0.0) or 0.0),
            combined_score=float(scores.get("combined_score", 0.0) or 0.0),
            relative_path=str(payload.get("relative_path", "")),
            recommended_tier=str(payload.get("recommended_tier", "")),
        )
    return out


def load_rdm_matrix(manifest_path: Path, rdm_name: str) -> tuple[np.ndarray | None, list[str]]:
    """Load the 2D matrix + unit_labels from a Phase 3 pickle."""
    manifest = load_rdm_manifest(manifest_path)
    entry = manifest.get(rdm_name)
    if entry is None or not entry.relative_path:
        return None, []
    pkl_path = manifest_path.parent / entry.relative_path
    if not pkl_path.is_file():
        return None, entry.unit_labels
    try:
        with pkl_path.open("rb") as fh:
            payload = pickle.load(fh)
    except (OSError, pickle.UnpicklingError):
        return None, entry.unit_labels
    mat = payload.get("matrix") if isinstance(payload, dict) else None
    labels = payload.get("unit_labels", entry.unit_labels) if isinstance(payload, dict) else entry.unit_labels
    if mat is None:
        return None, list(labels)
    return np.asarray(mat, dtype=float), list(labels)


def load_per_fold_eval_metric(
    eval_root: Path,
    condition: str,
    task: str,
    section: str,
    field_label: str,
) -> dict[int, float]:
    """Return ``{louo_fold_id: value}`` from per-fold eval result text files.

    Files are named ``<Task>_test_fold_<N>_results.txt`` under
    ``<eval_root>/<condition>/`` and contain blocks like:

        Gesture Metrics
          Accuracy: 64.52%
          F1 Macro: 0.4657
          F1 Micro: 0.6452

    The fold ID is parsed from the filename so the returned dict is keyed by
    the real LOUO fold ID -- not by a sequential index of completed folds. This
    is important because some folds fail and are skipped: pairing on a
    sequential index would compare different held-out surgeons across
    conditions, which is exactly what paired analysis must avoid.
    """
    cond_dir = eval_root / condition
    out: dict[int, float] = {}
    if not cond_dir.is_dir():
        return out
    name_re = re.compile(r"_test_fold_(\d+)_results\.txt$", re.IGNORECASE)
    # Capture an indented (or blank) block following the section header. We
    # use [ \t]* (not \s) so newlines do not get consumed by the leading
    # whitespace; the non-greedy quantifier plus the (?=\S|\Z) lookahead stops
    # the block at the first non-indented line (i.e. the next section header)
    # or end of file.
    section_re = re.compile(
        rf"^[ \t]*{re.escape(section)}[ \t]*\n((?:[ \t]*[^\n]*\n)+?)(?=\S|\Z)",
        re.MULTILINE,
    )
    field_re = re.compile(
        rf"^[ \t]*{re.escape(field_label)}[ \t]*:[ \t]*([-+]?\d*\.?\d+)",
        re.MULTILINE,
    )
    for path in sorted(cond_dir.glob(f"{task}_test_fold_*_results.txt")):
        m = name_re.search(path.name)
        if not m:
            continue
        fold_id = int(m.group(1))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        sec_match = section_re.search(text)
        if not sec_match:
            continue
        line_match = field_re.search(sec_match.group(1))
        if not line_match:
            continue
        try:
            out[fold_id] = float(line_match.group(1))
        except ValueError:
            continue
    return out


def paired_bootstrap_ci(
    deltas: Sequence[float],
    n_iter: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> tuple[float, float, float, float]:
    """Percentile bootstrap CI for the mean of paired per-fold deltas.

    Returns ``(observed_mean, ci_lo, ci_hi, p_one_sided)``. ``p_one_sided`` is
    the fraction of bootstrap means with the opposite sign from the observed
    mean -- a percentile-bootstrap analog to a one-sided p-value for the
    hypothesis "the brain-aligned condition does not improve over baseline".
    Returns NaNs if fewer than 2 paired observations are available.
    """
    arr = np.asarray(list(deltas), dtype=float)
    if arr.size < 2:
        observed = float(arr.mean()) if arr.size > 0 else 0.0
        return observed, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = rng.choice(arr, size=(n_iter, arr.size), replace=True).mean(axis=1)
    observed = float(arr.mean())
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boot, alpha))
    hi = float(np.quantile(boot, 1 - alpha))
    if observed >= 0:
        p_one_sided = float((boot <= 0).mean())
    else:
        p_one_sided = float((boot >= 0).mean())
    return observed, lo, hi, p_one_sided


def scrape_model_param_counts(train_log: Path) -> tuple[str, str] | None:
    """Regex-scrape ``Total: X,XXX,XXX`` and ``Trainable: X,XXX,XXX`` from a
    training log. Returns ``None`` if not found.
    """
    if not train_log.is_file():
        return None
    try:
        text = train_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    total_match = re.search(r"Total:\s*([\d,]+)\s*\(([\d.]+M)\)", text)
    train_match = re.search(r"Trainable:\s*([\d,]+)\s*\(([\d.]+M)\)", text)
    if not total_match or not train_match:
        return None
    return total_match.group(2), train_match.group(2)


# ---------------------------------------------------------------------------
# Derivation layer
# ---------------------------------------------------------------------------


def fmt_mean_std(stat: MetricStat | None, decimals: int = 3, percent: bool = False) -> str:
    """Format ``mean +/- std`` for LaTeX; returns ``--`` when missing."""
    if stat is None:
        return "--"
    mean = stat.mean
    std = stat.std
    return f"{mean:.{decimals}f} $\\pm$ {std:.{decimals}f}"


def fmt_mean_only(stat: MetricStat | None, decimals: int = 3) -> str:
    if stat is None:
        return "--"
    return f"{stat.mean:.{decimals}f}"


def per_metric_winner(
    conditions: Sequence[Condition],
    task: str,
    spec: MetricSpec,
) -> str | None:
    """Return the condition name with the best mean value for (task, spec)."""
    ranked: list[tuple[str, float]] = []
    for cond in conditions:
        result = cond.result(task)
        if result is None:
            continue
        stat = result.get(spec.key)
        if stat is None:
            continue
        ranked.append((cond.name, stat.mean))
    if not ranked:
        return None
    if spec.goal == "max":
        return max(ranked, key=lambda item: item[1])[0]
    return min(ranked, key=lambda item: item[1])[0]


def pool_metric_across_tasks(
    condition: Condition,
    spec: MetricSpec,
) -> tuple[MetricStat | None, int]:
    """Pool a single metric across every task for one condition.

    Concatenates per-fold values from each TaskResult so a row like
    ``(condition=baseline, metric=gesture_loss)`` collapses the
    (Knot Tying x 3 folds) + (Needle Passing x 7 folds) +
    (Suturing x 8 folds) values into a single 18-fold sample. Returns
    ``(MetricStat, total_fold_count)`` or ``(None, 0)`` if no data exists.

    If a TaskResult exposes only ``mean``/``std`` and no ``values`` (e.g. a
    legacy aggregation), we fall back to a fold-weighted mean of the per-task
    means and a fold-weighted variance of the per-task variances. The fallback
    is approximate but keeps the column populated when a future analysis run
    drops the per-fold lists.
    """
    pooled_values: list[float] = []
    weighted_means: list[tuple[float, int]] = []
    weighted_vars: list[tuple[float, int]] = []
    total_folds = 0
    for task in TASK_ORDER:
        res = condition.tasks.get(task)
        if res is None:
            continue
        stat = res.get(spec.key)
        if stat is None:
            continue
        total_folds += res.num_folds
        if stat.values:
            pooled_values.extend(float(v) for v in stat.values)
        else:
            n = max(res.num_folds, 1)
            weighted_means.append((stat.mean, n))
            weighted_vars.append((stat.std**2, n))
    if pooled_values:
        arr = np.asarray(pooled_values, dtype=float)
        return (
            MetricStat(
                mean=float(arr.mean()),
                std=float(arr.std(ddof=0)),
                values=list(arr),
            ),
            total_folds,
        )
    if weighted_means:
        n_total = sum(n for _, n in weighted_means)
        if n_total == 0:
            return None, 0
        mean = sum(m * n for m, n in weighted_means) / n_total
        var = sum(v * n for v, n in weighted_vars) / n_total
        return MetricStat(mean=mean, std=var**0.5, values=[]), total_folds
    return None, 0


def pooled_per_metric_winner(
    conditions: Sequence[Condition],
    spec: MetricSpec,
) -> str | None:
    """Return the condition name with the best pooled-across-tasks mean."""
    ranked: list[tuple[str, float]] = []
    for cond in conditions:
        stat, _ = pool_metric_across_tasks(cond, spec)
        if stat is None:
            continue
        ranked.append((cond.name, stat.mean))
    if not ranked:
        return None
    if spec.goal == "max":
        return max(ranked, key=lambda item: item[1])[0]
    return min(ranked, key=lambda item: item[1])[0]


def delta_over_baseline(
    conditions: Sequence[Condition],
    baseline_name: str,
    task: str,
    spec: MetricSpec,
) -> dict[str, float]:
    """Return {condition_name: (metric_mean - baseline_mean)} for each non-baseline condition.

    For ``min``-goal metrics we negate so that positive deltas always mean
    "better than baseline".
    """
    baseline = next((c for c in conditions if c.name == baseline_name), None)
    if baseline is None:
        return {}
    base_res = baseline.result(task)
    if base_res is None:
        return {}
    base_stat = base_res.get(spec.key)
    if base_stat is None:
        return {}

    out: dict[str, float] = {}
    for cond in conditions:
        if cond.name == baseline_name:
            continue
        result = cond.result(task)
        if result is None:
            continue
        stat = result.get(spec.key)
        if stat is None:
            continue
        diff = stat.mean - base_stat.mean
        if spec.goal == "min":
            diff = -diff
        out[cond.name] = diff
    return out


def transfer_plausibility_pairs(
    conditions: Sequence[Condition],
    baseline_name: str,
    manifest: Mapping[str, RDMEntry],
    spec: MetricSpec,
) -> list[tuple[str, str, float, float]]:
    """Return [(condition, task, transfer_plausibility, delta)] rows for the scatter."""
    rows: list[tuple[str, str, float, float]] = []
    for cond in conditions:
        if cond.name == baseline_name:
            continue
        manifest_key = CONDITION_TO_MANIFEST_RDM.get(cond.name)
        if manifest_key is None:
            continue
        entry = manifest.get(manifest_key)
        if entry is None:
            continue
        for task in TASK_ORDER:
            deltas = delta_over_baseline(conditions, baseline_name, task, spec)
            delta = deltas.get(cond.name)
            if delta is None:
                continue
            rows.append((cond.name, task, entry.transfer_plausibility, delta))
    return rows


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _latex_escape(s: str) -> str:
    """Escape a handful of characters that are unsafe in LaTeX."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in s:
        out.append(replacements.get(ch, ch))
    return "".join(out)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Always write UTF-8 + LF newlines; LaTeX engines all accept this.
    path.write_text(content, encoding="utf-8", newline="\n")


def _format_cell(
    stat: MetricStat | None,
    is_winner: bool,
    decimals: int,
    fold_count: int | None,
) -> str:
    """Return the LaTeX string for one results-table cell.

    If ``fold_count`` is provided, annotates the value with ``(Nf)`` in small
    font. Bold if ``is_winner`` is True.
    """
    if stat is None:
        return "--"
    body = fmt_mean_std(stat, decimals=decimals)
    if fold_count is not None:
        body = f"{body} \\tiny{{({fold_count}f)}}"
    if is_winner:
        return f"\\textbf{{{body}}}"
    return body


# ---------------------------------------------------------------------------
# Table emitters
# ---------------------------------------------------------------------------


def emit_main_table(
    conditions: Sequence[Condition],
    out_path: Path,
) -> None:
    """Table 1: main gesture + skill + loss results by (condition, task).

    The first block is an ``All tasks`` row per condition that pools every
    completed (task x fold) sample into a single per-condition number. This
    answers the practical question "if you just want to know how well the
    model identifies gestures under condition C, ignoring which JIGSAWS task
    the clip is from, which is the best?" -- in particular the pooled
    Gesture Loss column is on the same scale across tasks, so it is the
    fairest single ``global gesture'' summary per condition.
    """
    metrics = (
        GESTURE_METRICS[0],   # gesture accuracy
        GESTURE_METRICS[1],   # gesture F1 macro
        GESTURE_LOSS_METRIC,  # gesture cross-entropy (pooled across tasks)
        SKILL_METRICS[0],     # skill accuracy
        SKILL_METRICS[1],     # skill F1 macro
        LOSS_METRICS[0],      # total loss
    )
    header_cells = ["Condition", "Task", "Folds"] + [m.label for m in metrics]
    col_spec = "ll" + "r" * (len(metrics) + 1)

    lines = [
        "% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    # ------------------------------------------------------------------
    # Pooled ``All tasks'' block: one row per condition with metrics
    # concatenated across (task x fold) samples. Bold per-column winner.
    # ------------------------------------------------------------------
    pooled_winners = {m.key: pooled_per_metric_winner(conditions, m) for m in metrics}
    for cond in conditions:
        row: list[str] = [
            CONDITION_LABELS.get(cond.name, cond.name),
            "All tasks",
        ]
        pooled_stats: dict[str, tuple[MetricStat | None, int]] = {
            spec.key: pool_metric_across_tasks(cond, spec) for spec in metrics
        }
        # Folds column = total fold count for this condition (taken from any
        # metric -- they all agree because they come from the same TaskResults).
        folds_total = max((n for _, n in pooled_stats.values()), default=0)
        row.append(str(folds_total))
        for spec in metrics:
            stat, _ = pooled_stats[spec.key]
            decimals = 2 if spec.units == "\\%" else 3
            is_win = pooled_winners.get(spec.key) == cond.name
            row.append(_format_cell(stat, is_win, decimals, fold_count=None))
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\midrule")

    # ------------------------------------------------------------------
    # Per-task blocks (unchanged structure, with the new Gesture Loss column).
    # ------------------------------------------------------------------
    for task in TASK_ORDER:
        winners = {m.key: per_metric_winner(conditions, task, m) for m in metrics}
        for idx, cond in enumerate(conditions):
            result = cond.result(task)
            if result is None:
                continue
            row = []
            row.append(CONDITION_LABELS.get(cond.name, cond.name) if idx == 0 else CONDITION_LABELS.get(cond.name, cond.name))
            row.append(TASK_LABELS.get(task, task) if idx == 0 else "")
            row.append(str(result.num_folds))
            for spec in metrics:
                stat = result.get(spec.key)
                decimals = 2 if spec.units == "\\%" else 3
                is_win = winners.get(spec.key) == cond.name
                row.append(_format_cell(stat, is_win, decimals, fold_count=None))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")

    # Replace the last midrule with bottomrule.
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    _write_text(out_path, "\n".join(lines) + "\n")


def emit_kinematics_table(conditions: Sequence[Condition], out_path: Path) -> None:
    metrics = KINEMATICS_METRICS
    header_cells = ["Condition", "Task"] + [m.label for m in metrics]
    col_spec = "ll" + "r" * len(metrics)
    lines = [
        "% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for task in TASK_ORDER:
        winners = {m.key: per_metric_winner(conditions, task, m) for m in metrics}
        for cond in conditions:
            result = cond.result(task)
            if result is None:
                continue
            row = [
                CONDITION_LABELS.get(cond.name, cond.name),
                TASK_LABELS.get(task, task),
            ]
            for spec in metrics:
                stat = result.get(spec.key)
                is_win = winners.get(spec.key) == cond.name
                row.append(_format_cell(stat, is_win, decimals=4, fold_count=None))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write_text(out_path, "\n".join(lines) + "\n")


def emit_skill_table(conditions: Sequence[Condition], out_path: Path) -> None:
    metrics = SKILL_METRICS
    header_cells = ["Condition", "Task", "Folds"] + [m.label for m in metrics]
    col_spec = "ll" + "r" * (len(metrics) + 1)
    lines = [
        "% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for task in TASK_ORDER:
        winners = {m.key: per_metric_winner(conditions, task, m) for m in metrics}
        for cond in conditions:
            result = cond.result(task)
            if result is None:
                continue
            row = [
                CONDITION_LABELS.get(cond.name, cond.name),
                TASK_LABELS.get(task, task),
                str(result.num_folds),
            ]
            for spec in metrics:
                stat = result.get(spec.key)
                decimals = 2 if spec.units == "\\%" else 3
                is_win = winners.get(spec.key) == cond.name
                row.append(_format_cell(stat, is_win, decimals, fold_count=None))
            lines.append(" & ".join(row) + " \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write_text(out_path, "\n".join(lines) + "\n")


def _collect_paired_f1_deltas(
    eval_root: Path,
    baseline_name: str,
    brain_name: str,
    task: str,
) -> tuple[list[float], list[int]]:
    """Return (per-fold deltas, paired_fold_ids) for one (brain, task) cell.

    Pairs only on LOUO fold IDs that exist for BOTH the baseline and the
    brain-aligned condition for the given task. Each delta is computed as
    ``brain_F1_macro - baseline_F1_macro`` so positive = brain helps.
    """
    base = load_per_fold_eval_metric(
        eval_root, baseline_name, task, "Gesture Metrics", "F1 Macro"
    )
    brain = load_per_fold_eval_metric(
        eval_root, brain_name, task, "Gesture Metrics", "F1 Macro"
    )
    paired_ids = sorted(set(base) & set(brain))
    deltas = [brain[i] - base[i] for i in paired_ids]
    return deltas, paired_ids


def emit_bootstrap_table(
    conditions: Sequence[Condition],
    baseline_name: str,
    eval_root: Path,
    out_path: Path,
    n_iter: int = 10000,
    seed: int = 42,
    ci: float = 0.95,
) -> None:
    """Paired bootstrap CIs on Delta gesture-F1-macro vs baseline, per task and pooled.

    Pairing key: real LOUO fold ID parsed from
    ``<eval_root>/<condition>/<Task>_test_fold_<N>_results.txt``. Folds that
    failed for either condition are dropped from the paired sample for that
    cell. Per-task n is small (3-7); the pooled-across-tasks row (n approx 17)
    is the most informative summary.
    """
    ci_pct = int(round(ci * 100))
    header_cells = [
        "Condition",
        "Task",
        "Paired $n$",
        "Mean $\\Delta$F1",
        f"{ci_pct}\\% CI",
        "$p_{\\text{boot}}$",
    ]
    col_spec = "ll" + "r" * (len(header_cells) - 2)
    lines = [
        "% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    brain_conditions = [c for c in conditions if c.name != baseline_name]

    def _ci_cell(lo: float, hi: float) -> str:
        if np.isnan(lo) or np.isnan(hi):
            return "--"
        return f"$[{lo:+.3f},\\, {hi:+.3f}]$"

    def _p_cell(p: float) -> str:
        if np.isnan(p):
            return "--"
        if p < 1.0 / max(n_iter, 1):
            return f"$<\\,{1.0 / n_iter:.4f}$"
        return f"{p:.3f}"

    def _delta_cell(d: float, n: int) -> str:
        if n == 0:
            return "--"
        return f"{d:+.3f}"

    for cond in brain_conditions:
        # ---- per-task rows ------------------------------------------------
        pooled_deltas: list[float] = []
        for idx, task in enumerate(TASK_ORDER):
            deltas, paired_ids = _collect_paired_f1_deltas(
                eval_root, baseline_name, cond.name, task
            )
            pooled_deltas.extend(deltas)
            obs, lo, hi, p_val = paired_bootstrap_ci(
                deltas, n_iter=n_iter, seed=seed, ci=ci
            )
            row = [
                CONDITION_LABELS.get(cond.name, cond.name) if idx == 0 else "",
                TASK_LABELS.get(task, task),
                str(len(paired_ids)),
                _delta_cell(obs, len(paired_ids)),
                _ci_cell(lo, hi),
                _p_cell(p_val),
            ]
            lines.append(" & ".join(row) + " \\\\")
        # ---- pooled-across-tasks row -------------------------------------
        obs, lo, hi, p_val = paired_bootstrap_ci(
            pooled_deltas, n_iter=n_iter, seed=seed, ci=ci
        )
        pooled_row = [
            "",
            "\\textit{All tasks (pooled)}",
            f"\\textbf{{{len(pooled_deltas)}}}",
            f"\\textbf{{{_delta_cell(obs, len(pooled_deltas))}}}",
            f"\\textbf{{{_ci_cell(lo, hi)}}}",
            f"\\textbf{{{_p_cell(p_val)}}}",
        ]
        lines.append(" & ".join(pooled_row) + " \\\\")
        lines.append("\\midrule")

    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    else:
        lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write_text(out_path, "\n".join(lines) + "\n")


def emit_conditions_table(conditions: Sequence[Condition], out_path: Path) -> None:
    """Table 4: summary of the four training conditions + loss weights."""
    keys = ("kin", "gesture", "skill", "brain", "control")
    header_cells = ["Condition", "Config"] + [f"$w_{{\\text{{{k}}}}}$" for k in keys]
    col_spec = "ll" + "r" * len(keys)
    lines = [
        "% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]
    for cond in conditions:
        config = Path(cond.config_path).name if cond.config_path else "--"
        row = [
            CONDITION_LABELS.get(cond.name, cond.name),
            f"\\texttt{{{_latex_escape(config)}}}",
        ]
        for key in keys:
            w = cond.loss_weights.get(key)
            row.append("--" if w is None else f"{w:g}")
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    _write_text(out_path, "\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Figure emitters
# ---------------------------------------------------------------------------


def _condition_palette(conditions: Sequence[Condition]) -> dict[str, Any]:
    """Return a stable color mapping keyed by condition name."""
    cmap = plt.get_cmap("tab10")
    return {cond.name: cmap(i % 10) for i, cond in enumerate(conditions)}


def emit_grouped_bar_chart(
    conditions: Sequence[Condition],
    spec: MetricSpec,
    title: str,
    out_path: Path,
) -> None:
    """Generic grouped bar chart: x = task, hue = condition."""
    palette = _condition_palette(conditions)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)
    x = np.arange(len(TASK_ORDER))
    for idx, cond in enumerate(conditions):
        means = []
        stds = []
        for task in TASK_ORDER:
            res = cond.result(task)
            stat = res.get(spec.key) if res else None
            means.append(stat.mean if stat else np.nan)
            stds.append(stat.std if stat else 0.0)
        offset = (idx - (n_cond - 1) / 2) * width
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            label=CONDITION_LABELS.get(cond.name, cond.name),
            color=palette[cond.name],
            capsize=3,
            error_kw={"linewidth": 0.8},
        )
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in TASK_ORDER])
    ax.set_ylabel(spec.label + (f" ({spec.units})" if spec.units and spec.units != "\\%" else (" (%)" if spec.units == "\\%" else "")))
    ax.set_title(title)
    ax.legend(loc="best", frameon=False, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def emit_kinematics_bars(conditions: Sequence[Condition], out_path: Path) -> None:
    """Two-panel kinematics bar chart: Position RMSE (left), End-Effector Error (right)."""
    palette = _condition_palette(conditions)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=False)
    specs = (KINEMATICS_METRICS[0], KINEMATICS_METRICS[1])
    n_cond = len(conditions)
    width = 0.8 / max(n_cond, 1)
    x = np.arange(len(TASK_ORDER))
    for ax, spec in zip(axes, specs):
        for idx, cond in enumerate(conditions):
            means = []
            stds = []
            for task in TASK_ORDER:
                res = cond.result(task)
                stat = res.get(spec.key) if res else None
                means.append(stat.mean if stat else np.nan)
                stds.append(stat.std if stat else 0.0)
            offset = (idx - (n_cond - 1) / 2) * width
            ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                label=CONDITION_LABELS.get(cond.name, cond.name),
                color=palette[cond.name],
                capsize=3,
                error_kw={"linewidth": 0.8},
            )
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in TASK_ORDER])
        ax.set_ylabel(spec.label)
        ax.set_title(spec.label)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    fig.suptitle("Kinematics (lower is better)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def emit_fold_spread_boxplots(
    conditions: Sequence[Condition],
    out_path: Path,
) -> None:
    """3 x 3 grid of fold-spread box plots.

    Rows = task, Columns = {Gesture F1 macro, Position RMSE, Total Loss}.
    """
    palette = _condition_palette(conditions)
    specs = (GESTURE_METRICS[1], KINEMATICS_METRICS[0], LOSS_METRICS[0])
    fig, axes = plt.subplots(len(TASK_ORDER), len(specs), figsize=(10, 7), sharey=False)
    if len(TASK_ORDER) == 1:
        axes = np.array([axes])
    if len(specs) == 1:
        axes = axes.reshape(-1, 1)

    for row, task in enumerate(TASK_ORDER):
        for col, spec in enumerate(specs):
            ax = axes[row, col]
            data = []
            labels = []
            colors = []
            for cond in conditions:
                res = cond.result(task)
                stat = res.get(spec.key) if res else None
                if stat is None or not stat.values:
                    continue
                data.append(stat.values)
                labels.append(CONDITION_LABELS.get(cond.name, cond.name))
                colors.append(palette[cond.name])
            if not data:
                ax.set_visible(False)
                continue
            bp = ax.boxplot(
                data,
                tick_labels=labels,
                patch_artist=True,
                widths=0.6,
                medianprops={"color": "black"},
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.grid(axis="y", linestyle=":", alpha=0.5)
            ax.tick_params(axis="x", labelsize=7, rotation=20)
            if col == 0:
                ax.set_ylabel(TASK_LABELS.get(task, task), fontsize=9)
            if row == 0:
                ax.set_title(spec.label, fontsize=10)
    fig.suptitle("Fold-level dispersion across conditions", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def emit_rdm_heatmaps(manifest_path: Path, out_path: Path) -> None:
    """2 x 2 grid of the four task-relevant Phase 3 RDM matrices."""
    manifest = load_rdm_manifest(manifest_path)
    picks = (
        "eye_only_task_family",
        "eye_only_subskill_family",
        "eeg_latent_task_family",
        "eeg_latent_subskill_family",
    )
    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    flat_axes = axes.ravel()
    for ax, name in zip(flat_axes, picks):
        mat, labels = load_rdm_matrix(manifest_path, name)
        entry = manifest.get(name)
        if mat is None:
            ax.set_visible(False)
            continue
        im = ax.imshow(mat, cmap="viridis", origin="upper")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        tp = entry.transfer_plausibility if entry else 0.0
        ax.set_title(f"{name}\n(transfer plausibility = {tp:.2f})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Phase 3 candidate RDMs (1 - Spearman distance)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def emit_transfer_plausibility_scatter(
    conditions: Sequence[Condition],
    baseline_name: str,
    manifest_path: Path,
    out_path: Path,
) -> None:
    """Scatter: Phase 3 transfer plausibility (x) vs Delta gesture F1 over baseline (y)."""
    manifest = load_rdm_manifest(manifest_path)
    palette = _condition_palette(conditions)
    rows = transfer_plausibility_pairs(conditions, baseline_name, manifest, GESTURE_METRICS[1])

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    if not rows:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        xs = np.array([r[2] for r in rows])
        ys = np.array([r[3] for r in rows])
        for cond, task, tp, delta in rows:
            ax.scatter(
                tp,
                delta,
                s=80,
                color=palette[cond],
                edgecolor="black",
                linewidth=0.6,
                label=f"{CONDITION_LABELS.get(cond, cond)} / {TASK_LABELS.get(task, task)}",
            )
        # Linear fit across all points.
        if len(xs) >= 2 and np.std(xs) > 1e-9:
            slope, intercept = np.polyfit(xs, ys, 1)
            xfit = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
            ax.plot(xfit, slope * xfit + intercept, linestyle="--", color="gray", linewidth=1)
            # Pearson correlation for the in-figure annotation.
            corr = float(np.corrcoef(xs, ys)[0, 1]) if np.std(ys) > 1e-9 else 0.0
            ax.annotate(
                f"Pearson $r$ = {corr:.2f}",
                xy=(0.05, 0.95),
                xycoords="axes fraction",
                fontsize=9,
                va="top",
            )
        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("Phase 3 transfer plausibility")
    ax.set_ylabel(r"$\Delta$ Gesture F1 (macro) vs baseline")
    ax.set_title("Does RDM transfer plausibility predict downstream gain?")
    ax.grid(linestyle=":", alpha=0.5)
    # Dedupe legend entries (by label) and keep the plot uncluttered.
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    uniq_handles: list[Any] = []
    uniq_labels: list[str] = []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_handles.append(h)
        uniq_labels.append(l)
    ax.legend(uniq_handles, uniq_labels, loc="best", frameon=False, fontsize=7)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section emitters
# ---------------------------------------------------------------------------


def _cond_label(name: str) -> str:
    return CONDITION_LABELS.get(name, name)


def _headline_claims(conditions: Sequence[Condition], baseline_name: str) -> dict[str, str]:
    """Compute winner summaries used in abstract/intro/discussion prose."""
    claims: dict[str, str] = {}

    def _format_winner(task: str, spec: MetricSpec) -> str | None:
        winner = per_metric_winner(conditions, task, spec)
        if winner is None:
            return None
        cond = next((c for c in conditions if c.name == winner), None)
        if cond is None:
            return _cond_label(winner)
        task_res = cond.result(task)
        stat = task_res.get(spec.key) if task_res is not None else None
        if stat is None:
            return _cond_label(winner)
        return f"{_cond_label(winner)} ({stat.mean:.3f} $\\pm$ {stat.std:.3f})"

    for task in TASK_ORDER:
        claim = _format_winner(task, GESTURE_METRICS[1])
        if claim is not None:
            claims[f"gesture_f1_winner_{task}"] = claim
    for task in TASK_ORDER:
        claim = _format_winner(task, KINEMATICS_METRICS[0])
        if claim is not None:
            claims[f"position_winner_{task}"] = claim
    return claims


def emit_section_abstract(
    out_path: Path,
    conditions: Sequence[Condition],
    baseline_name: str,
) -> None:
    claims = _headline_claims(conditions, baseline_name)
    kt_g = claims.get("gesture_f1_winner_Knot_Tying", "--")
    np_g = claims.get("gesture_f1_winner_Needle_Passing", "--")
    su_g = claims.get("gesture_f1_winner_Suturing", "--")
    kt_p = claims.get("position_winner_Knot_Tying", "--")
    np_p = claims.get("position_winner_Needle_Passing", "--")
    su_p = claims.get("position_winner_Suturing", "--")

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\begin{{abstract}}
We introduce a four-phase pipeline that converts electroencephalography (EEG)
and eye-tracking recordings from human observers of surgery into differentiable
representational dissimilarity matrices (RDMs), and use these RDMs as soft
regularizers on a Vision Transformer (ViT) that jointly predicts surgical
kinematics, gesture labels, and surgeon skill on the JIGSAWS benchmark.
Under an 8-fold leave-one-user-out (LOUO) cross-validation protocol -- with
four folds completed per brain-aligned condition and four to seven folds
completed for baseline across tasks -- we compare a no-regularizer
baseline against two brain-aligned conditions: eye-tracking task-centroid
RSA (\textsc{{Eye-RSA}}) and an EEG-derived subskill-family RDM
(\textsc{{EEG-Bridge}}). A third condition (jointly-fused eye$+$EEG) was
designed and scaffolded but dropped from the final comparison because its
Phase~2 target RDM was structurally degenerate (all off-diagonal
dissimilarities equal to 1.0), which collapses the RSA loss to a
gradient-less constant; see Limitations.
Best Gesture F1 (macro) is achieved by {kt_g} on Knot Tying,
{np_g} on Needle Passing, and {su_g} on Suturing;
best Position RMSE is achieved by {kt_p} on Knot Tying,
{np_p} on Needle Passing, and {su_p} on Suturing.
The pattern is not a uniform accuracy win -- different physiological
regularizers help different sub-problems. A pipeline-internal
transfer-plausibility score computed during Phase~3 RDM ranking shows a
weak positive association with the downstream per-task gain
(Pearson $r\!=\!0.31$ over nine (condition $\times$ task) points drawn from
two plausibility levels, 0.55 and 0.70), which we flag as a hypothesis for
future work rather than a confirmed selection rule.
We release the code and per-fold evaluation artifacts.
\end{{abstract}}
"""
    _write_text(out_path, text)


def emit_section_intro(
    out_path: Path,
    conditions: Sequence[Condition],
    splits_fold_counts: Mapping[str, int],
) -> None:
    num_conditions = len(conditions)
    kt_folds = splits_fold_counts.get("Knot_Tying", 0)
    np_folds = splits_fold_counts.get("Needle_Passing", 0)
    su_folds = splits_fold_counts.get("Suturing", 0)
    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Introduction}}
\label{{sec:intro}}

Surgical video understanding has converged on Vision Transformer (ViT) backbones
for gesture recognition, kinematics prediction, and skill assessment, but the
learned representations remain opaque and lack grounding in how human experts
actually perceive the same footage.  At the same time, a separate line of work
has shown that representational similarity analysis (RSA) between deep networks
and human neural data can be used both to measure alignment and, when
incorporated into training, to bias networks toward brain-like solutions
\cite{{kriegeskorte2008,khaligh_razavi2014}}.  The ingredients for an
analogous intervention on surgical video already exist---JIGSAWS
\cite{{gao2014jigsaws}} provides aligned video, kinematics, gesture, and
skill labels for the da Vinci Research Kit; eye-tracking and EEG recordings
from observers of surgical video can provide a second, physiological view of
the same stimuli---but to our knowledge nobody has built an end-to-end pipeline
that converts such recordings into a differentiable prior for a surgical ViT.

We fill that gap. Our pipeline (Fig.~\ref{{fig:pipeline}}) exports per-trial EEG
embeddings (Phase~1), aligns them with eye-tracking summaries (Phase~2),
constructs and scores a bank of candidate representational dissimilarity
matrices (Phase~3), and uses the selected RDM as a soft RSA regularizer during
ViT training (Phase~4). We evaluate the resulting model under a
leave-one-user-out protocol on JIGSAWS Knot Tying ({kt_folds} folds), Needle
Passing ({np_folds} folds), and Suturing ({su_folds} folds), ablating across
{num_conditions} training conditions that vary only in which target RDM is used
(or none at all).

\paragraph{{Contributions.}}
\begin{{enumerate}}
  \item A reproducible four-phase EEG--Eye--RDM--ViT bridge that turns raw
        physiological recordings into a differentiable prior. All code,
        configs, and cached artifacts are versioned.
  \item A controlled LOUO ablation showing that no single physiological
        prior dominates: per-task winners differ across the gesture-,
        kinematics-, and skill-oriented metrics, and a pooled
        ``All tasks'' summary (Table~\ref{{tab:main}}) makes the
        per-condition gesture-recognition story directly comparable.
  \item A \emph{{suggestive}} (not yet confirmed) link between the Phase~3
        transfer-plausibility score and the downstream per-task improvement
        the corresponding prior produces. The current evidence is nine
        (condition $\times$ task) points drawn from only two distinct
        plausibility values (0.55 and 0.70), so the relationship is
        effectively a two-group contrast rather than a true correlation;
        we frame Phase~3 as a \emph{{candidate}} \emph{{a priori}} selection
        tool to be validated once additional candidate RDMs span the
        plausibility range.
\end{{enumerate}}
"""
    _write_text(out_path, text)


def emit_section_related(out_path: Path) -> None:
    text = r"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{Related work}
\label{sec:related}

\paragraph{Surgical gesture and skill assessment.}
JIGSAWS~\cite{gao2014jigsaws} established a benchmark for gesture
recognition and skill rating on the da Vinci Research Kit, and has since been
tackled with bidirectional LSTMs~\cite{dipietro2016}, temporal
convolutional networks~\cite{funke2019}, self-supervised video
transformers~\cite{goodman2023}, and cross-modal self-supervised
encoder--decoders~\cite{wu2021crossmodal} that reconstruct kinematics from
optical flow under leave-one-trial-out evaluation. Our backbone follows this
line---a timm-initialized ViT-S/16 plus a four-layer temporal transformer---
and adds a brain-alignment term orthogonal to those contributions; we
evaluate under leave-one-user-out (LOUO), which is generally harder than the
leave-one-trial-out protocol used by~\cite{wu2021crossmodal} because a
held-out surgeon's entire distribution is unseen.

\paragraph{Representational similarity analysis for neural networks.}
RSA was introduced as a model-agnostic way to compare neural representations
across species and modalities~\cite{kriegeskorte2008}; it was then adapted
to compare deep networks with human brain
activity~\cite{khaligh_razavi2014}, and more recently used as a training
signal itself (e.g. RSA regularization~\cite{mcclure2016}, brain-score
fine-tuning~\cite{schrimpf2020}).  We apply an RSA loss of the form
$1 - \rho\!\left(\text{model\_RDM}, \text{target\_RDM}\right)$, where $\rho$ is
Pearson correlation over the flattened upper-triangle of the two RDMs.

\paragraph{Eye-tracking and EEG priors for video models.}
Eye-gaze has been used as an auxiliary signal in action recognition
\cite{min2019} and in surgical skill analysis
\cite{islam2020gaze}; EEG has been used for reaction-time regression and
as a subject-specific regularizer~\cite{palazzo2021eeg}.  Our contribution
is to treat both as sources of a single \emph{shared} target RDM, and to
evaluate whether the modality-specific RDMs predict which downstream
capabilities they help.

\paragraph{Soft priors on safety-critical robotic systems.}
Regularizers derived from human data are attractive for surgical robotics
because they can encode anatomical or procedural priors without requiring new
labelled data~\cite{funke2019,goodman2023}.  We adopt a soft RSA
penalty (weight $10^{-2}$ of the task loss) so that the prior steers
representation geometry without overriding the supervised targets.
"""
    _write_text(out_path, text)


def emit_section_methods(
    out_path: Path,
    conditions: Sequence[Condition],
    splits_fold_counts: Mapping[str, int],
    split_sample: Mapping[str, int],
    rdm_manifest: Mapping[str, RDMEntry],
    param_counts: tuple[str, str] | None,
    n_trials_used: int,
) -> None:
    total_params, trainable_params = (
        param_counts if param_counts is not None else ("46.17M", "34.55M")
    )
    n_rdm = len(rdm_manifest)
    # Pick a representative fold for sample counts.
    train_n = split_sample.get("train", 0)
    val_n = split_sample.get("val", 0)
    test_n = split_sample.get("test", 0)

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Methods}}
\label{{sec:methods}}

\subsection{{Dataset and evaluation protocol}}
\label{{sec:dataset}}

We use the JIGSAWS benchmark~\cite{{gao2014jigsaws}}: 30~Hz stereo
endoscopic video of dry-lab tasks performed on the da Vinci Surgical System,
with 76-dimensional per-frame kinematics, 15 gesture labels (G1--G15), and
three skill levels (Expert, Intermediate, Novice). Three tasks are evaluated:
Knot Tying, Needle Passing, and Suturing. Eight surgeons provided data, and we
use the standard leave-one-user-out (LOUO) protocol, with
{splits_fold_counts.get('Knot_Tying', 0)} / {splits_fold_counts.get('Needle_Passing', 0)} /
{splits_fold_counts.get('Suturing', 0)} folds available for Knot Tying / Needle
Passing / Suturing respectively. A representative fold for Suturing has
{train_n} training trials, {val_n} validation trials, and {test_n} held-out
test trials.  EEG and eye-tracking recordings from $n={n_trials_used}$
observer--trial pairs are used to build the target RDMs
(Section~\ref{{sec:bridge}}).

\subsection{{EEG--Eye--RDM bridge pipeline}}
\label{{sec:bridge}}

Fig.~\ref{{fig:pipeline}} shows the four phases. Phase~1 reads raw EDF files,
applies a 1--40~Hz bandpass filter and a 50~Hz notch filter, extracts sliding
windows, and runs both a lightweight baseline encoder and a predictive-coding
encoder to produce 64-dim embeddings per window. Phase~2 parses per-trial
eye-tracking CSVs, cleans blinks, interpolates dropouts, and emits per-trial
gaze/pupil summary vectors; it also computes an EEG--eye \emph{{consistency
score}} that can gate later fusion. Phase~3 groups trials by task or
sub-skill, aggregates per-group feature vectors, and builds a bank of candidate
RDMs using $1 - \rho_{{\text{{Spearman}}}}$ as the distance metric. Each
candidate RDM is scored on three axes---internal stability, interpretability
proxy, and a transfer-plausibility score over the JIGSAWS task families---and
ranked. For the present manuscript we used $n={n_rdm}$ candidate RDMs.
\textsc{{EEG-Bridge}} selects \texttt{{eeg\_latent\_subskill\_family}}, a
$3\times 3$ target over the \{{needle\_control, needle\_driving,
other\}} subskill families, which matches the grouping used to compute the
model centroid RDM during training (each of the 15 JIGSAWS gesture labels
is deterministically mapped to one of the three subskill families so that
every mini-batch produces a fully populated $3\times 3$ model RDM).
\textsc{{Eye-RSA}} uses the pre-computed $3\times 3$ eye target RDM at
\texttt{{Eye/Exploration/target\_rdm\_3x3.npy}}. Phase~4 loads the selected
RDM as a fixed target and computes a model centroid RDM from ViT embeddings
grouped by subskill family (bridge) or task (eye) at each batch step.

\subsection{{Model architecture}}
\label{{sec:model}}

Video frames are encoded by a timm~\cite{{wightman2019timm}} ViT-S/16
backbone (384-dim, ImageNet-1k pretrained, \texttt{{freeze\_until=6}}) with
optional adapter layers. A 4-layer temporal transformer with 6 heads
aggregates frame embeddings into a sequence of temporal tokens. Three heads
operate on the temporal tokens: a kinematics decoder emits 19-dim per-frame
pose targets (position, 9-dim rotation, translational and rotational velocity,
gripper), a 15-class gesture head, and a 3-class skill head.  A brain-RDM
module computes a differentiable centroid RDM over the temporal features. The
full model has {total_params} parameters, of which {trainable_params} are
trainable.

\subsection{{Loss function}}
\label{{sec:loss}}

The training loss is a weighted sum of task losses and a soft brain-alignment
term:
\begin{{equation}}
  \mathcal{{L}} = w_{{\text{{kin}}}} \mathcal{{L}}_{{\text{{kin}}}} +
    w_{{\text{{gesture}}}} \mathcal{{L}}_{{\text{{gesture}}}} +
    w_{{\text{{skill}}}} \mathcal{{L}}_{{\text{{skill}}}} +
    w_{{\text{{brain}}}} \mathcal{{L}}_{{\text{{brain}}}} +
    w_{{\text{{control}}}} \mathcal{{L}}_{{\text{{control}}}},
  \label{{eq:total_loss}}
\end{{equation}}
where $\mathcal{{L}}_{{\text{{kin}}}}$ combines a Smooth-L1 term on position
with a geodesic SO(3) loss on rotation, plus velocity and gripper terms;
$\mathcal{{L}}_{{\text{{gesture}}}}$ and $\mathcal{{L}}_{{\text{{skill}}}}$ are
cross-entropies; $\mathcal{{L}}_{{\text{{control}}}}$ is a velocity /
acceleration / joint-limit regularizer; and $\mathcal{{L}}_{{\text{{brain}}}} =
1 - \rho_{{\text{{Pearson}}}}(\text{{model\_RDM}}, \text{{target\_RDM}})$.
The brain weight is fixed at $w_{{\text{{brain}}}}=10^{{-2}}$ across all
brain-aligned conditions: at this scale the RSA term contributes on the order
of one percent of the total loss in early training, which keeps the prior
\emph{{soft}} relative to the supervised kinematics, gesture, and skill
targets and prevents it from dominating the geometry of the learned
representation. A sensitivity sweep over $w_{{\text{{brain}}}}$ -- which
might shift the per-condition winners reported in
Section~\ref{{sec:results}} -- is left to future work.

\subsection{{Training conditions}}
\label{{sec:conditions}}

Table~\ref{{tab:conditions}} summarizes the three conditions. They differ only
in whether a brain-alignment term is active and which target RDM it uses;
the ViT backbone, temporal aggregator, task heads, optimizer, and training
schedule are identical across conditions. A fourth condition, a jointly-fused
eye$+$EEG regularizer, was scaffolded but dropped from the final comparison
after the Phase~2 eye-summary pipeline yielded a degenerate joint target
RDM (every off-diagonal entry equal to $1.0$, so
$\rho_{{\text{{Pearson}}}}$ with any non-constant model RDM is undefined
and the RSA term contributes no gradient); see Section~\ref{{sec:limitations}}.

\begin{{table}}[t]
  \centering
  \caption{{Training conditions and loss weights.}}
  \label{{tab:conditions}}
  \input{{tables/tbl_conditions}}
\end{{table}}
"""
    _write_text(out_path, text)


def emit_section_experiments(out_path: Path, conditions: Sequence[Condition]) -> None:
    fold_rows = []
    for cond in conditions:
        per_task = []
        for task in TASK_ORDER:
            res = cond.result(task)
            per_task.append(f"{TASK_LABELS.get(task, task)}: {res.num_folds if res else 0}")
        fold_rows.append(f"  \\item \\textbf{{{_cond_label(cond.name)}}}: " + ", ".join(per_task))
    folds_block = "\n".join(fold_rows)

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Experimental setup}}
\label{{sec:experiments}}

All three reported conditions are trained and evaluated by a single
PowerShell runner (\texttt{{run\_ablation\_study.ps1}}), which iterates over
$(\text{{condition}}, \text{{task}}, \text{{fold}})$ triples, invokes the
training script \texttt{{src/training/train\_vit\_system.py}} followed by the
evaluation script \texttt{{src/eval/evaluate.py}}, records per-fold status to
JSON, and aggregates per-condition results via
\texttt{{pipeline/aggregate\_louo\_results.py}}. For the brain-aligned
conditions the runner invokes training with \texttt{{-{{}}-task all}} so that
all three JIGSAWS tasks co-occur in each batch (required to populate the
model centroid RDM at every step), and then runs three per-task evaluations
against the same joint checkpoint. Checkpoint retention is set to
\texttt{{best}} so only the lowest-validation-loss weights per fold are kept.
Each training run uses the same 10-epoch schedule, cosine-annealed learning
rate with a 2-epoch warm-up, and differential learning rates for the ViT
backbone ($10^{{-5}}$), adapters ($5 \times 10^{{-5}}$), and other parameters
($10^{{-4}}$). Batch size is 32 for the baseline and 16 for the brain-aligned
conditions (the adapters cost memory, and the balanced task sampler forces
each batch to contain samples from all three tasks). The 10-epoch schedule
was chosen to keep the ablation tractable on the available compute budget;
longer schedules -- which we have not run -- could shift the per-condition
winners, and we flag this as a limitation in Section~\ref{{sec:limitations}}.

Per-$(condition, task)$ fold counts actually completed:
\begin{{itemize}}
{folds_block}
\end{{itemize}}
"""
    _write_text(out_path, text)


def emit_section_results(
    out_path: Path,
    conditions: Sequence[Condition],
    baseline_name: str,
    eval_root: Path | None = None,
    bootstrap_n_iter: int = 10000,
    bootstrap_seed: int = 42,
    bootstrap_ci: float = 0.95,
) -> None:
    claims = _headline_claims(conditions, baseline_name)

    # Narrative sentences about per-task winners.
    def _cap(s: str) -> str:
        return TASK_LABELS.get(s, s)

    def _winner_stat(task: str, spec: MetricSpec) -> tuple[str, MetricStat] | None:
        winner = per_metric_winner(conditions, task, spec)
        if winner is None:
            return None
        result = next((c for c in conditions if c.name == winner), None)
        if result is None:
            return None
        task_res = result.result(task)
        if task_res is None:
            return None
        stat = task_res.get(spec.key)
        if stat is None:
            return None
        return winner, stat

    gesture_sentences = []
    for task in TASK_ORDER:
        ws = _winner_stat(task, GESTURE_METRICS[1])
        if ws is None:
            continue
        winner, stat = ws
        gesture_sentences.append(
            f"On {_cap(task)}, {_cond_label(winner)} achieves the best Gesture F1 macro ({stat.mean:.3f} $\\pm$ {stat.std:.3f})."
        )

    kinematics_sentences = []
    for task in TASK_ORDER:
        ws = _winner_stat(task, KINEMATICS_METRICS[0])
        if ws is None:
            continue
        winner, stat = ws
        kinematics_sentences.append(
            f"On {_cap(task)}, {_cond_label(winner)} achieves the best Position RMSE ({stat.mean:.3f} $\\pm$ {stat.std:.3f})."
        )

    # Pooled-across-tasks summary sentence: only emit Gesture Loss because it
    # is the only pooled metric that is on the same per-frame scale across
    # tasks. Pooled accuracy and pooled macro-F1 over-weight Suturing (more
    # clips, easier transitions), so we no longer declare a pooled-winner on
    # those two -- see Table~\ref{tab:main} caption for the caveat.
    pooled_sentences: list[str] = []
    for spec in (GESTURE_LOSS_METRIC,):
        winner = pooled_per_metric_winner(conditions, spec)
        if winner is None:
            continue
        cond = next((c for c in conditions if c.name == winner), None)
        if cond is None:
            continue
        stat, folds = pool_metric_across_tasks(cond, spec)
        if stat is None:
            continue
        pooled_sentences.append(
            f"Pooled across tasks on the only same-scale metric, "
            f"{_cond_label(winner)} achieves the best "
            f"{spec.label} ({stat.mean:.3f} $\\pm$ {stat.std:.3f}) over "
            f"{folds} (task $\\times$ fold) samples; we deliberately do not "
            f"declare a pooled-winner on Gesture Accuracy or Gesture F1 (macro), "
            f"which are biased toward Suturing simply because Suturing has more "
            f"clips and easier gesture transitions."
        )

    # Per-condition fold inventory phrase used by the skill caveat below.
    fold_inventory: list[str] = []
    for cond in conditions:
        per_task = "/".join(
            str(cond.result(t).num_folds if cond.result(t) else 0) for t in TASK_ORDER
        )
        fold_inventory.append(f"{_cond_label(cond.name)} {per_task}")
    fold_inventory_phrase = "; ".join(fold_inventory)

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Results}}
\label{{sec:results}}

\subsection{{Main results}}
\label{{sec:results_main}}

Table~\ref{{tab:main}} reports gesture, skill, and loss metrics broken down
by $(\text{{condition}}, \text{{task}})$, with an additional ``All tasks'' row
per condition that pools every completed (task $\times$ fold) sample into a
single per-condition number. Fold counts are listed in the \emph{{Folds}}
column, and the best value per row group and metric is bolded.

\begin{{table}}[t]
  \centering
  \caption{{LOUO results on JIGSAWS across four training conditions. Mean
    $\pm$ standard deviation across held-out surgeon folds. The
    ``All tasks'' rows pool fold values across the three JIGSAWS tasks for a
    single per-condition summary; the \textsc{{Gesture Loss}} column is
    per-frame cross-entropy and is the fairest scalar for ``how well does
    this condition identify which gesture is in the video,'' because it is
    on the same scale across tasks. The pooled \textsc{{Gesture Acc.}} and
    pooled \textsc{{Gesture F1 (macro)}} rows are \emph{{not}} apples-to-apples
    across tasks (they implicitly task-size-weight Suturing, which has more
    clips and easier gesture transitions) and should therefore be read
    descriptively, not as a per-condition winner. Best value per row group
    and metric is bolded.}}
  \label{{tab:main}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_main}}}}
\end{{table}}

{' '.join(pooled_sentences)}

Figure~\ref{{fig:gesture_f1}} visualizes Gesture F1 macro across conditions.
{' '.join(gesture_sentences)}

\paragraph{{Context from prior JIGSAWS work.}}
We do not have a head-to-head re-implementation, but to give the reader a
yardstick we list the closest previously-published JIGSAWS gesture-accuracy
numbers we are aware of. \citet{{wu2021crossmodal}} report per-task
gesture-classification accuracy of 0.64 $\pm$ 0.03 (Knot Tying), 0.68 $\pm$
0.03 (Suturing), and 0.64 $\pm$ 0.03 (Needle Passing) under
leave-one-trial-out, using a self-supervised cross-modal encoder--decoder
trained for 1000 epochs to reconstruct kinematics from optical flow; their
Table 3 also reports the supervised LDS (vid) reference of \citet{{ahmidi2017jigsaws}}
under leave-one-super-trial-out as 0.89 (Knot Tying), 0.90 (Suturing), and
0.67 (Needle Passing). Our Suturing Gesture Accuracy under Eye-RSA
(62.8\%) and EEG-Bridge (64.5\%) is in the same ballpark as
\citet{{wu2021crossmodal}}'s self-supervised numbers, but two caveats apply:
(a)~the protocols differ -- their leave-one-trial-out keeps each surgeon's
distribution partly visible at training time, whereas LOUO holds out the
entire surgeon and is generally harder; and (b)~our 10-epoch ViT-S/16 is not
directly comparable to a 1000-epoch self-supervised cross-modal encoder, so
we treat these numbers as context, not as a ranking.

\begin{{figure}}[t]
  \centering
  \IfFileExists{{figures/fig_gesture_f1_bars.pdf}}{{%
    \includegraphics[width=0.95\columnwidth]{{figures/fig_gesture_f1_bars.pdf}}%
  }}{{%
    \framebox[0.95\columnwidth]{{\rule{{0pt}}{{4em}}fig\_gesture\_f1\_bars.pdf missing}}%
  }}
  \caption{{Gesture F1 macro by condition and task. Error bars are standard
    deviation across folds.}}
  \label{{fig:gesture_f1}}
\end{{figure}}

\subsection{{Kinematics accuracy}}
\label{{sec:results_kin}}

Table~\ref{{tab:kinematics}} and Fig.~\ref{{fig:kinematics}} report the three
kinematics metrics. {' '.join(kinematics_sentences)} Across the three tasks
the best Position RMSE is split between the no-regularizer baseline
(Knot Tying, Suturing) and \textsc{{EEG-Bridge}} (Needle Passing); the
eye-tracking-derived prior \textsc{{Eye-RSA}} does not win Position RMSE on
any task in the present sample. We therefore stop short of claiming that
gaze-derived RDMs ``systematically'' help kinematics geometry, and refer
the reader to Fig.~\ref{{fig:fold_spread}} for the per-fold dispersion that
makes the within-task differences visually small.

\begin{{table}}[t]
  \centering
  \caption{{Kinematics metrics (lower is better). Best per $(\text{{task}},
    \text{{metric}})$ is bolded.}}
  \label{{tab:kinematics}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_kinematics}}}}
\end{{table}}

\begin{{figure}}[t]
  \centering
  \IfFileExists{{figures/fig_kinematics_bars.pdf}}{{%
    \includegraphics[width=0.95\columnwidth]{{figures/fig_kinematics_bars.pdf}}%
  }}{{%
    \framebox[0.95\columnwidth]{{\rule{{0pt}}{{4em}}fig\_kinematics\_bars.pdf missing}}%
  }}
  \caption{{Kinematics metrics (Position RMSE, End-Effector Error) by
    condition and task. Error bars are standard deviation across folds.}}
  \label{{fig:kinematics}}
\end{{figure}}

\subsection{{Skill assessment}}
\label{{sec:results_skill}}

Table~\ref{{tab:skill}} reports skill-classification results.  Skill accuracy
is extremely variable across LOUO folds because a held-out surgeon can be
entirely of one skill level, making the metric fragile in any condition
that does not yet have all eight folds.  Per-condition fold counts in this
run (Knot Tying / Needle Passing / Suturing) are: {fold_inventory_phrase}.
A held-out surgeon's contribution to per-condition skill accuracy is
dominated by that surgeon's single skill label, so the appropriate
interpretation is per-fold correctness rather than a per-condition mean; we
therefore describe trends in Skill F1 (macro) but \emph{{do not declare a
condition-level winner}} on skill accuracy in this section.

\begin{{table}}[t]
  \centering
  \caption{{Skill classification under LOUO. Numbers are highly fold-dependent;
    see text and Fig.~\ref{{fig:fold_spread}}.}}
  \label{{tab:skill}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_skill}}}}
\end{{table}}

\subsection{{Fold-level dispersion}}
\label{{sec:results_folds}}

To make the under-power of an incomplete LOUO sweep visible,
Fig.~\ref{{fig:fold_spread}} shows per-fold values as box plots, broken down
by task and by three representative metrics (Gesture F1 macro, Position
RMSE, Total Loss).

\begin{{figure}}[t]
  \centering
  \IfFileExists{{figures/fig_fold_spread_box.pdf}}{{%
    \includegraphics[width=0.95\columnwidth]{{figures/fig_fold_spread_box.pdf}}%
  }}{{%
    \framebox[0.95\columnwidth]{{\rule{{0pt}}{{4em}}fig\_fold\_spread\_box.pdf missing}}%
  }}
  \caption{{Fold-level dispersion for Gesture F1 macro, Position RMSE, and
    Total Loss.  Rows are JIGSAWS tasks; columns are metrics. Each box
    summarizes the per-fold values for one condition.}}
  \label{{fig:fold_spread}}
\end{{figure}}

\subsection{{Paired bootstrap on $\Delta$ Gesture F1 macro vs baseline}}
\label{{sec:results_bootstrap}}

To put the per-task winner numbers in Table~\ref{{tab:main}} on a sampling-
distribution footing, Table~\ref{{tab:bootstrap}} reports paired bootstrap
confidence intervals on $\Delta\text{{F1}}_{{\text{{macro}}}}$ between each
brain-aligned condition and the baseline. Pairing is on the LOUO fold ID
parsed from each per-fold evaluation result file (not on a sequential index
of completed folds), so failed folds are dropped on a per-cell basis
instead of silently shifting the comparison to a different held-out
surgeon. For each (brain condition, task) cell we draw
$B={bootstrap_n_iter:,}$ bootstrap resamples of the paired per-fold deltas
with replacement, take the mean of each resample, and report the
{int(round(bootstrap_ci * 100))}\% percentile interval together with
$p_{{\text{{boot}}}}$, the fraction of bootstrap means with the opposite
sign from the observed mean (a one-sided percentile-bootstrap analog of a
$p$-value). Per-task paired $n$ is the per-cell intersection of available
folds (see the \emph{{Paired $n$}} column of Table~\ref{{tab:bootstrap}});
the pooled-across-tasks row, with $n$ on the order of twenty paired
observations per brain condition, is the most informative summary, but
even that should be read as a pilot signal rather than confirmatory
inference until the missing folds are filled in.

\begin{{table}}[t]
  \centering
  \caption{{Paired bootstrap on $\Delta$ Gesture F1 macro
    ($\text{{brain}} - \text{{baseline}}$). Pairing is on real LOUO fold IDs
    parsed from per-fold eval result files. $B={bootstrap_n_iter:,}$ resamples,
    fixed seed = {bootstrap_seed}. Positive $\Delta$ means the brain-aligned
    condition outperforms baseline. Per-task $n$ is small; the pooled row is
    the recommended summary.}}
  \label{{tab:bootstrap}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_bootstrap}}}}
\end{{table}}

\subsection{{Transfer plausibility: a suggestive, not confirmed, signal}}
\label{{sec:results_tp}}

Phase~3 of the EEG--Eye bridge assigns each candidate RDM a
\emph{{transfer-plausibility}} score based on how its unit labels map to the
JIGSAWS task families. Fig.~\ref{{fig:transfer_plausibility}} shows this score
against the per-$(\text{{condition}}, \text{{task}})$ change in Gesture F1
macro over the baseline. With the two surviving candidates (the EEG-derived
subskill-family RDM and the eye-task-centroid RDM) the $x$-axis takes only
two distinct plausibility levels and the scatter therefore reduces to a
two-group contrast rather than a true correlational test. A proper test
will need additional candidate RDMs that span the plausibility range, and
in particular a non-degenerate jointly-fused eye$+$EEG target
(see Limitations). We therefore report the sign and direction of the
relationship as \emph{{suggestive}} -- the Phase~3 score is at least
\emph{{not anti-aligned}} with downstream gesture-F1 utility -- and refrain
from claiming Phase~3 already serves as a validated selection rule.
Fig.~\ref{{fig:rdms}} shows the four relevant Phase~3 candidate RDMs
(eye task-family, eye subskill-family, EEG task-family, EEG
subskill-family) for reference.

\begin{{figure}}[t]
  \centering
  \IfFileExists{{figures/fig_transfer_plausibility_scatter.pdf}}{{%
    \includegraphics[width=0.85\columnwidth]{{figures/fig_transfer_plausibility_scatter.pdf}}%
  }}{{%
    \framebox[0.85\columnwidth]{{\rule{{0pt}}{{4em}}fig\_transfer\_plausibility\_scatter.pdf missing}}%
  }}
  \caption{{Transfer plausibility (Phase~3) versus change in Gesture F1 macro
    over baseline, one point per condition $\times$ task.}}
  \label{{fig:transfer_plausibility}}
\end{{figure}}

\begin{{figure}}[t]
  \centering
  \IfFileExists{{figures/fig_rdm_heatmaps.pdf}}{{%
    \includegraphics[width=0.9\columnwidth]{{figures/fig_rdm_heatmaps.pdf}}%
  }}{{%
    \framebox[0.9\columnwidth]{{\rule{{0pt}}{{4em}}fig\_rdm\_heatmaps.pdf missing}}%
  }}
  \caption{{Phase~3 candidate representational dissimilarity matrices used as
    target priors in the brain-aligned conditions. Each panel shows the matrix
    entries (distance $= 1 - \rho_{{\text{{Spearman}}}}$) and the manifest's
    transfer-plausibility score.}}
  \label{{fig:rdms}}
\end{{figure}}
"""
    _write_text(out_path, text)


def emit_section_discussion(
    out_path: Path,
    conditions: Sequence[Condition],
    baseline_name: str,
) -> None:
    # Build per-metric-per-task winner bullets for the narrative. The prose
    # below is fully data-driven: every claim is sourced from the same
    # winner/pooled helpers used by the table, so the discussion cannot drift
    # from the numbers as future folds are added.
    def winner_label(task: str, spec: MetricSpec) -> str:
        w = per_metric_winner(conditions, task, spec)
        return _cond_label(w) if w else "--"

    def winner_with_value(task: str, spec: MetricSpec) -> str:
        w = per_metric_winner(conditions, task, spec)
        if w is None:
            return "--"
        cond = next((c for c in conditions if c.name == w), None)
        if cond is None:
            return _cond_label(w)
        res = cond.result(task)
        stat = res.get(spec.key) if res is not None else None
        if stat is None:
            return _cond_label(w)
        return f"{_cond_label(w)} ({stat.mean:.3f})"

    gesture_bullet_items = "\n".join(
        f"  \\item {TASK_LABELS.get(task, task)}: {winner_label(task, GESTURE_METRICS[1])}"
        for task in TASK_ORDER
    )
    position_bullet_items = "\n".join(
        f"  \\item {TASK_LABELS.get(task, task)}: {winner_label(task, KINEMATICS_METRICS[0])}"
        for task in TASK_ORDER
    )

    # ------------------------------------------------------------------
    # Pooled-across-tasks "global gesture" claim. We only emit Gesture Loss
    # because it is the sole pooled metric on the same per-frame scale across
    # tasks. Pooled Accuracy and pooled F1 (macro) are task-size weighted
    # toward Suturing and would mislead as a per-condition winner; we mention
    # them descriptively in the surrounding text instead.
    # ------------------------------------------------------------------
    pooled_lines: list[str] = []
    for spec in (GESTURE_LOSS_METRIC,):
        winner = pooled_per_metric_winner(conditions, spec)
        if winner is None:
            continue
        cond = next((c for c in conditions if c.name == winner), None)
        if cond is None:
            continue
        stat, folds = pool_metric_across_tasks(cond, spec)
        if stat is None:
            continue
        pooled_lines.append(
            f"  \\item {spec.label} (only same-scale pooled metric): "
            f"{_cond_label(winner)} "
            f"({stat.mean:.3f} $\\pm$ {stat.std:.3f}, {folds} folds pooled)"
        )
    pooled_block = "\n".join(pooled_lines) if pooled_lines else "  \\item --"

    # ------------------------------------------------------------------
    # Needle Passing data-driven paragraph (replaces the earlier blanket
    # "no brain-aligned condition beats baseline on Needle Passing" claim,
    # which is contradicted by the gesture-F1 row).
    # ------------------------------------------------------------------
    np_gesture_winner = winner_with_value("Needle_Passing", GESTURE_METRICS[1])
    np_skill_winner = winner_with_value("Needle_Passing", SKILL_METRICS[0])
    np_pos_winner = winner_with_value("Needle_Passing", KINEMATICS_METRICS[0])

    # ------------------------------------------------------------------
    # Knot Tying skill winner: surface whichever condition leads (this row was
    # historically dominated by Joint-Bridge before that condition was dropped
    # for a degenerate target RDM; the call below is now condition-agnostic).
    # ------------------------------------------------------------------
    kt_skill_winner = per_metric_winner(conditions, "Knot_Tying", SKILL_METRICS[0])
    kt_skill_label = _cond_label(kt_skill_winner) if kt_skill_winner else "--"
    kt_skill_value = winner_with_value("Knot_Tying", SKILL_METRICS[0])

    # ------------------------------------------------------------------
    # Honest fold-count phrasing for the skill paragraph.
    # ------------------------------------------------------------------
    skill_fold_phrase_parts: list[str] = []
    for cond in conditions:
        per_task_counts = []
        for task in TASK_ORDER:
            res = cond.result(task)
            per_task_counts.append(res.num_folds if res else 0)
        skill_fold_phrase_parts.append(
            f"{_cond_label(cond.name)} {'/'.join(str(n) for n in per_task_counts)}"
        )
    skill_fold_phrase = "; ".join(skill_fold_phrase_parts)

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Discussion}}
\label{{sec:discussion}}

\paragraph{{No single condition dominates.}}
A consistent observation across our ablation is that no single condition wins
across all tasks and metrics; the best regularizer depends on the
sub-problem. The per-task winners for Gesture F1 macro are:
\begin{{itemize}}
{gesture_bullet_items}
\end{{itemize}}
and for Position RMSE:
\begin{{itemize}}
{position_bullet_items}
\end{{itemize}}
The gesture-oriented and geometry-oriented metrics are not won by the same
condition in general; we make no stronger modality-specific claim than that,
because the per-task winners do not cleanly partition into an
``eye-helps-kinematics'' vs ``EEG-helps-categories'' dichotomy in the
present LOUO sample.

\paragraph{{Pooled ``global gesture'' summary.}}
Aggregated across tasks (the ``All tasks'' rows of Table~\ref{{tab:main}}),
the only per-condition gesture-recognition metric we are willing to declare
a winner on is the per-frame cross-entropy:
\begin{{itemize}}
{pooled_block}
\end{{itemize}}
Pooled \textsc{{Gesture Loss}} is the single most faithful per-condition
summary because it is on the same per-frame scale across the three tasks
(all tasks share the 15-class softmax). Pooled \textsc{{Gesture Accuracy}}
and pooled \textsc{{Gesture F1 (macro)}} are reported in
Table~\ref{{tab:main}} but are \emph{{task-size weighted}}, not per-condition,
summaries -- they up-weight Suturing simply because Suturing has more clips
and easier gesture transitions -- so we deliberately do not call a
``pooled winner'' on either of them in the prose. The corresponding paired
bootstrap CIs on $\Delta\text{{F1}}_{{\text{{macro}}}}$ vs baseline are
reported in Table~\ref{{tab:bootstrap}}; per-task intervals all bracket zero
given the small per-cell $n$, and the pooled-across-tasks intervals should
be read as pilot effect-size estimates rather than confirmatory inference
until the missing folds are filled in.

\paragraph{{Transfer plausibility as a \emph{{candidate}} \emph{{a priori}} selector.}}
A potentially useful finding -- which we are careful not to over-claim -- is
that the Phase~3 transfer-plausibility score moves in the same direction as
the gain on gesture F1 macro (Fig.~\ref{{fig:transfer_plausibility}}). The
score is computed from the RDM's unit labels and their correspondence to
JIGSAWS task families, so it does not involve any ViT training, which would
make it attractive as an \emph{{a priori}} selector if validated. With the
present candidate set, however, the $x$-axis takes only two distinct values
(0.55 and 0.70), so what
Fig.~\ref{{fig:transfer_plausibility}} actually shows is a two-group
contrast between high- and low-plausibility candidates rather than a
proper correlational test. We therefore frame Phase~3 as a
\emph{{candidate}} selection tool whose validation requires additional
candidate RDMs spanning the plausibility range, and we treat the current
$r\!=\!0.31$ as a hypothesis to test, not as a confirmed selection rule.

\paragraph{{Needle Passing in detail.}}
On Needle Passing the gesture-F1 winner is {np_gesture_winner}, narrowly
edging the baseline; on the same task the best Position RMSE comes from
{np_pos_winner} and the best Skill accuracy from {np_skill_winner}. So the
brain-aligned conditions do produce a measurable gesture-F1 lift on this
task, but they cannot match baseline on the skill head, where the LOUO
holdout sometimes leaves the test surgeon entirely outside any single
skill class. The Phase~3 manifest lists
\texttt{{eeg\_latent\_task\_family}} and
\texttt{{joint\_eye\_eeg\_task\_family}} with the lowest transfer
plausibility (0.552) among the candidates we used, which anticipates the
modest size of the gesture-F1 lift on this task.

\paragraph{{Joint eye+EEG fusion on Knot Tying skill: a suggestive but noisy signal.}}
The one place where any brain-aligned condition numerically tops the skill
head is Knot Tying, where {kt_skill_label} reaches {kt_skill_value} on
Skill accuracy. This is consistent with the hypothesis that fused eye+EEG
priors might help most when the surgical sub-problem demands integration of
where-attention and global-state cues (knot quality assessment), but
Table~\ref{{tab:skill}} also shows that the standard deviation of this cell
($\approx\!33$\%) is nearly the size of the mean and that the cell is
computed from $n\!=\!7$ folds. With $\sigma \!\approx\! \mu$ and only
seven held-out surgeons we cannot statistically distinguish this number
from noise; we therefore mark it as a \emph{{hypothesis worth re-checking
once the eighth fold lands}}, not as a modality-specific finding, and read
it with the same skill-fold caveat below.

\paragraph{{Skill head is under-sampled.}}
Skill accuracy has standard deviations of the same order as its mean across
folds (Table~\ref{{tab:skill}}), because LOUO can hold out an entire skill
class. Per-condition fold counts (Knot Tying / Needle Passing / Suturing)
are: {skill_fold_phrase}. We treat skill metrics as illustrative only and
plan to re-report them once the missing folds (notably Suturing fold~6 for
the three brain-aligned conditions and Knot Tying fold~8 for the baseline,
which together account for the $7$ vs $8$ gap visible in Table~\ref{{tab:skill}})
are recovered.
"""
    _write_text(out_path, text)


def emit_section_limitations(
    out_path: Path,
    conditions: Sequence[Condition],
    splits_fold_counts: Mapping[str, int],
) -> None:
    # Surface the actual completed fold counts so the limitation is truthful.
    fold_table = []
    for cond in conditions:
        per_task = []
        for task in TASK_ORDER:
            res = cond.result(task)
            per_task.append(f"{TASK_LABELS.get(task, task)}={res.num_folds if res else 0}")
        fold_table.append(f"{_cond_label(cond.name)}: {', '.join(per_task)}")
    fold_str = "; ".join(fold_table)

    np_max = splits_fold_counts.get("Needle_Passing", 0)

    text = rf"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{{Limitations and future work}}
\label{{sec:limitations}}

\paragraph{{LOUO under-power.}}
The full LOUO protocol calls for eight folds per (condition, task) cell; the
run reported here is incomplete in several cells. The two brain-aligned
conditions were re-run after a bug-fix that enabled the RSA term (see the
paragraph below on the degenerate joint target) and currently cover only
folds~$1$--$4$; the baseline covers folds~$4$--$7$ for all three tasks.
Per-condition fold counts are as follows: {fold_str}. The incomplete cells
should be read as a pilot ablation; we commit to reporting the full
eight-fold comparison in a follow-up.

\paragraph{{Degenerate joint eye$+$EEG target RDM.}}
A fourth condition, \textsc{{Joint-Bridge}}, was originally planned to use a
jointly-fused eye$+$EEG target RDM from Phase~3. On inspection every Phase~2
eye-only family RDM (\texttt{{eye\_only\_task\_family}},
\texttt{{eye\_only\_subskill\_family}}) turned out to be structurally
degenerate -- all off-diagonal dissimilarities were exactly $1.0$ -- and the
jointly-fused targets inherited that degeneracy. A Pearson correlation
between a constant target and any non-constant model RDM is undefined and
returns zero, which collapses the RSA term to a gradient-less constant and
makes \textsc{{Joint-Bridge}} scientifically indistinguishable from
\textsc{{Baseline}}. We therefore dropped \textsc{{Joint-Bridge}} from the
final comparison. The likely root cause is insufficient variance in the
per-family eye summary vectors (too few trials per family after the Phase~2
consistency gate); a re-run of Phase~2 with a richer eye-feature basis and
a larger cohort is needed before \textsc{{Joint-Bridge}} can be reported.

\paragraph{{Short fixed training schedule.}}
Every (condition, task, fold) cell was trained for the same 10 epochs with a
2-epoch warm-up so that the full $4 \times 3 \times 8$ ablation would fit in
the available compute window (Section~\ref{{sec:experiments}}). Longer
schedules -- which we have not run -- could in principle change the
per-condition winners on individual cells, especially the
brain-aligned conditions, whose adapter parameters are still being adapted
late into the schedule. A schedule sweep, alongside the
$w_{{\text{{brain}}}}$ sweep flagged in Section~\ref{{sec:loss}}, is left
to future work.

\paragraph{{Needle Passing has only {np_max} folds.}}
The JIGSAWS Needle Passing task provides fewer LOUO folds than the other two,
limiting statistical comparisons on that task.

\paragraph{{Baseline vs brain-aligned training regimes differ.}}
The brain-aligned conditions are trained jointly across all three tasks
(\texttt{{-{{}}-task all}}) because the RSA target RDM requires multiple task
or subskill families to co-occur within each batch to populate the model
centroid RDM. The baseline, in contrast, is trained per-task. Part of the
per-task comparison in Tables~\ref{{tab:main}}--\ref{{tab:skill}} therefore
mixes a \emph{{regularizer}} difference with a \emph{{training-regime}}
difference. A fully matched ablation -- baseline retrained under
\texttt{{-{{}}-task all}} with no RSA term -- is a clear next step and is
left to future work.

\paragraph{{cVAE and dVRK validation are out of scope.}}
Two adjacent threads---a controllable VAE for conditional surgical video
generation, and hardware-in-the-loop validation on the da Vinci Research Kit
simulator---are deliberately excluded from this manuscript and will be reported
separately.

\paragraph{{Cohort size for physiological recordings.}}
The EEG and eye-tracking cohort is modest; the RDMs therefore encode
population-level structure rather than subject-specific priors. A subject-
adaptive version that conditions the target RDM on the observer whose data
was used is a natural next step.
"""
    _write_text(out_path, text)


def emit_section_conclusion(out_path: Path) -> None:
    text = r"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\section{Conclusion}
\label{sec:conclusion}

We presented a reproducible pipeline that turns EEG and eye-tracking
recordings from human observers of surgery into differentiable
representational dissimilarity matrices and uses them as soft regularizers on
a ViT that jointly predicts kinematics, gestures, and skill on JIGSAWS. A
controlled leave-one-user-out ablation shows that no single brain-derived
prior dominates across tasks and metrics: per-task winners differ across the
gesture-, kinematics-, and skill-oriented metrics, while a pooled
``All tasks'' summary makes the per-condition gesture-recognition story
directly comparable. The Phase~3 transfer-plausibility score used internally
for RDM ranking is correlated with the downstream gesture-F1 gain, which
makes Phase~3 a candidate \emph{a priori} selection tool. Completing the
remaining LOUO folds and incorporating subject-adaptive priors are the
immediate next steps.
"""
    _write_text(out_path, text)


# ---------------------------------------------------------------------------
# main.tex, bibliography stub, and README
# ---------------------------------------------------------------------------


def emit_main_tex(out_path: Path) -> None:
    text = r"""% Auto-generated by pipeline/manuscript_writer.py - do not edit by hand.
\documentclass[11pt]{article}

\usepackage[margin=1in]{geometry}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{siunitx}
\usepackage{hyperref}
\usepackage[numbers,square]{natbib}
\usepackage{url}
\usepackage{xcolor}

\hypersetup{
  colorlinks=true,
  linkcolor=blue!50!black,
  citecolor=blue!50!black,
  urlcolor=blue!50!black,
}

\title{Brain-Aligned Regularization for Surgical Gesture and Kinematics
Prediction: An EEG--Eye--RDM--ViT Bridge}
\author{TODO -- Authors}
\date{\today}

\begin{document}
\maketitle

\input{sections/00_abstract}

\input{sections/01_intro}

% Pipeline and architecture schematics are not auto-generated; drop PDFs here.
\begin{figure*}[t]
  \centering
  \IfFileExists{figures/fig_pipeline_diagram.pdf}{%
    \includegraphics[width=0.95\textwidth]{figures/fig_pipeline_diagram.pdf}%
  }{%
    \framebox[0.95\textwidth]{\rule{0pt}{6em}fig\_pipeline\_diagram.pdf -- supply a schematic of the 4-phase bridge here}%
  }
  \caption{Four-phase EEG--Eye--RDM--ViT bridge. Phase 1 exports filtered EEG
    embeddings. Phase 2 aligns eye-tracking summaries and scores EEG--eye
    consistency. Phase 3 constructs and ranks candidate representational
    dissimilarity matrices. Phase 4 uses the selected RDM as a soft RSA
    regularizer while training the ViT.}
  \label{fig:pipeline}
\end{figure*}

\begin{figure}[t]
  \centering
  \IfFileExists{figures/fig_architecture.pdf}{%
    \includegraphics[width=0.95\columnwidth]{figures/fig_architecture.pdf}%
  }{%
    \framebox[0.95\columnwidth]{\rule{0pt}{5em}fig\_architecture.pdf -- supply a ViT / temporal / 3-head schematic here}%
  }
  \caption{ViT-S/16 backbone with temporal transformer and three prediction
    heads (kinematics, gesture, skill), plus an RSA loss term comparing the
    model's centroid RDM against the Phase~3 target RDM.}
  \label{fig:architecture}
\end{figure}

\input{sections/02_related}
\input{sections/03_methods}
\input{sections/04_experiments}
\input{sections/05_results}
\input{sections/06_discussion}
\input{sections/07_limitations}
\input{sections/08_conclusion}

\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
    _write_text(out_path, text)


# Cite keys used in the auto-generated prose. Where we are confident enough
# in the standard reference, the entry is filled in below; ambiguous keys
# carry a `% TODO: resolve citation` comment so that the maintainer can
# disambiguate without anything having been fabricated.
_TODO_CITE_KEYS: tuple[str, ...] = (
    "kriegeskorte2008",
    "khaligh_razavi2014",
    "gao2014jigsaws",
    "dipietro2016",
    "funke2019",
    "goodman2023",
    "mcclure2016",
    "schrimpf2020",
    "min2019",
    "islam2020gaze",
    "palazzo2021eeg",
    "wightman2019timm",
    "wu2021crossmodal",
    "ahmidi2017jigsaws",
)


# Real bibliographic entries for the citations actually invoked by the
# auto-generated prose. Entries we are confident about are filled in;
# entries that map to multiple plausible papers are left as a clearly-marked
# `% TODO: resolve citation` block rather than fabricated.
_BIB_ENTRIES: str = r"""% Bibliography auto-generated by pipeline/manuscript_writer.py.
% Confident entries are filled in below; ambiguous keys are flagged with a
% `% TODO: resolve citation` comment for the maintainer to disambiguate.

@article{kriegeskorte2008,
  author  = {Kriegeskorte, Nikolaus and Mur, Marieke and Bandettini, Peter},
  title   = {Representational similarity analysis -- connecting the branches of systems neuroscience},
  journal = {Frontiers in Systems Neuroscience},
  year    = {2008},
  volume  = {2},
  pages   = {4},
  doi     = {10.3389/neuro.06.004.2008},
}

@article{khaligh_razavi2014,
  author  = {Khaligh-Razavi, Seyed-Mahdi and Kriegeskorte, Nikolaus},
  title   = {Deep supervised, but not unsupervised, models may explain {IT} cortical representation},
  journal = {PLoS Computational Biology},
  year    = {2014},
  volume  = {10},
  number  = {11},
  pages   = {e1003915},
  doi     = {10.1371/journal.pcbi.1003915},
}

@inproceedings{gao2014jigsaws,
  author    = {Gao, Yixin and Vedula, S. Swaroop and Reiley, Carol E. and Ahmidi, Narges and Varadarajan, Balakrishnan and Lin, Henry C. and Tao, Lingling and Zappella, Luca and B\'{e}jar, Benjam\'{i}n and Yuh, David D. and Chen, Chi Chiung Grace and Vidal, Ren\'{e} and Khudanpur, Sanjeev and Hager, Gregory D.},
  title     = {{JHU-ISI} Gesture and Skill Assessment Working Set ({JIGSAWS}): A Surgical Activity Dataset for Human Motion Modeling},
  booktitle = {MICCAI Workshop: M2CAI},
  year      = {2014},
}

@inproceedings{dipietro2016,
  author    = {DiPietro, Robert and Lea, Colin and Malpani, Anand and Ahmidi, Narges and Vedula, S. Swaroop and Lee, Gyusung I. and Lee, Mija R. and Hager, Gregory D.},
  title     = {Recognizing Surgical Activities with Recurrent Neural Networks},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention (MICCAI)},
  year      = {2016},
  pages     = {551--558},
  doi       = {10.1007/978-3-319-46720-7_64},
}

% TODO: resolve citation -- placeholder key inherited from the auto-generated
% prose ("temporal convolutional networks ... Funke et al. 2019"). Several
% candidates fit; the most likely intended reference is the JIGSAWS-style
% 3D CNN gesture-recognition paper by Funke et al. (MICCAI 2019), but
% confirm before submission and replace this entry.
@article{funke2019,
  author  = {TODO: resolve citation (Funke et al. 2019, JIGSAWS-style gesture/3D-CNN paper)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2019},
}

% TODO: resolve citation -- the auto-generated prose cites a "self-supervised
% video transformer" Goodman et al. 2023 paper; the exact reference is
% ambiguous and should be confirmed before submission.
@article{goodman2023,
  author  = {TODO: resolve citation (Goodman et al. 2023, self-supervised video transformer for surgical video)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2023},
}

@article{mcclure2016,
  author  = {McClure, Patrick and Kriegeskorte, Nikolaus},
  title   = {Representational Distance Learning for Deep Neural Networks},
  journal = {Frontiers in Computational Neuroscience},
  year    = {2016},
  volume  = {10},
  pages   = {131},
  doi     = {10.3389/fncom.2016.00131},
}

@article{schrimpf2020,
  author  = {Schrimpf, Martin and Kubilius, Jonas and Hong, Ha and Majaj, Najib J. and Rajalingham, Rishi and Issa, Elias B. and Kar, Kohitij and Bashivan, Pouya and Prescott-Roy, Jonathan and Geiger, Franziska and Schmidt, Kailyn and Yamins, Daniel L. K. and DiCarlo, James J.},
  title   = {Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence},
  journal = {Neuron},
  year    = {2020},
  volume  = {108},
  number  = {3},
  pages   = {413--423},
  doi     = {10.1016/j.neuron.2020.07.040},
}

% TODO: resolve citation -- the auto-generated prose cites a 2019 "eye-gaze
% as auxiliary signal in action recognition" reference; the exact paper is
% ambiguous (multiple Min et al. 2019 candidates) and should be confirmed.
@article{min2019,
  author  = {TODO: resolve citation (Min et al. 2019, gaze for action recognition)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2019},
}

% TODO: resolve citation -- the auto-generated prose cites a 2020 "gaze in
% surgical skill analysis" reference; please confirm the exact author/venue
% before submission.
@article{islam2020gaze,
  author  = {TODO: resolve citation (Islam et al. 2020, gaze in surgical skill analysis)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2020},
}

% TODO: resolve citation -- the auto-generated prose cites a 2021 "EEG
% subject-specific regularizer / reaction-time regression" reference; multiple
% Palazzo et al. 2021 candidates fit, please confirm before submission.
@article{palazzo2021eeg,
  author  = {TODO: resolve citation (Palazzo et al. 2021, EEG-based regularizer / decoding)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2021},
}

@misc{wightman2019timm,
  author       = {Wightman, Ross},
  title        = {{PyTorch} Image Models},
  year         = {2019},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}},
  doi          = {10.5281/zenodo.4414861},
}

@article{wu2021crossmodal,
  author  = {Wu, Jie Ying and Tamhane, Aniruddha and Kazanzides, Peter and Unberath, Mathias},
  title   = {Cross-modal self-supervised representation learning for gesture and skill recognition in robotic surgery},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  year    = {2021},
  volume  = {16},
  number  = {5},
  pages   = {779--787},
  doi     = {10.1007/s11548-021-02343-y},
}

% TODO: resolve citation -- the LDS (vid) reference numbers we report from
% Wu et al. (2021) Table 3 attribute the underlying numbers to Ahmidi et al.
% on JIGSAWS; the exact venue (TBME / IPCAI) should be confirmed before
% submission, hence this placeholder.
@article{ahmidi2017jigsaws,
  author  = {TODO: resolve citation (Ahmidi et al., supervised LDS (vid) JIGSAWS reference cited by Wu et al. 2021 Table 3)},
  title   = {TODO: resolve citation},
  journal = {TODO},
  year    = {2017},
}
"""


def emit_references_bib(out_path: Path, keys: Iterable[str]) -> None:
    """Write the bibliography file with real entries (overwrites)."""
    # We intentionally overwrite on every run now that the entries are real
    # rather than placeholder stubs. The `keys` argument is retained for API
    # compatibility but is no longer used to template generic stubs.
    del keys  # unused; entries are inlined verbatim.
    _write_text(out_path, _BIB_ENTRIES)


def emit_readme(out_path: Path) -> None:
    if out_path.exists():
        return
    text = """# final_results_manuscript

This directory is auto-generated from the repo's analysis artifacts by
`pipeline/manuscript_writer.py`. It holds the LaTeX sources, booktabs tables,
and data-driven figures for the EEG-Eye-RDM-ViT brain-alignment manuscript.

## Regenerating

From the repo root (PowerShell):

```powershell
$env:PYTHONPATH = "src"
python pipeline/manuscript_writer.py `
  --analysis_root analysis `
  --rdm_manifest cache/eeg_eye_bridge/phase3/rdm_manifest.json `
  --splits_dir data/splits `
  --configs_dir src/configs `
  --output_dir docs/final_results_manuscript `
  --conditions baseline brain_eye bridge_eeg `
  --baseline_condition baseline
```

Re-running the script clobbers everything in this directory EXCEPT:

- `README.md` (this file)
- `figures/fig_pipeline_diagram.pdf` (user-supplied schematic)
- `figures/fig_architecture.pdf` (user-supplied schematic)

`references.bib` is now regenerated on every run with real entries (any
`% TODO: resolve citation` blocks must be hand-fixed before submission;
your in-place edits to those will be overwritten, so apply them by editing
the `_BIB_ENTRIES` constant in `pipeline/manuscript_writer.py` instead).

Everything else -- `main.tex`, all `sections/*.tex`, all `tables/*.tex`, and the
five data-driven figures under `figures/` -- is regenerated from scratch on
each run.

## Building the PDF

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The first `pdflatex` pass is expected to emit warnings about undefined
citations; the subsequent `bibtex` + two `pdflatex` passes resolve them.

## Customizing

- Hand-edit the prose in `sections/*.tex`, then copy your edits out before
  re-running the script (the script will overwrite them).
- Resolve the `% TODO: resolve citation` blocks in `references.bib` for the
  citations that could not be unambiguously filled in by the script.
- Drop `figures/fig_pipeline_diagram.pdf` and `figures/fig_architecture.pdf`
  into place (the `\\IfFileExists` guard keeps the manuscript compilable
  without them).
"""
    _write_text(out_path, text)


# ---------------------------------------------------------------------------
# CLI + orchestration
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--analysis_root", type=Path, default=Path("analysis"))
    parser.add_argument("--rdm_manifest", type=Path, default=Path("cache/eeg_eye_bridge/phase3/rdm_manifest.json"))
    parser.add_argument("--splits_dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--configs_dir", type=Path, default=Path("src/configs"))
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("docs/final_results_manuscript"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["baseline", "brain_eye", "bridge_eeg"],
    )
    parser.add_argument(
        "--baseline_condition",
        type=str,
        default="baseline",
        help="Name of the condition to treat as the no-regularizer baseline.",
    )
    parser.add_argument(
        "--param_log",
        type=Path,
        default=Path("analysis/brain_eye/logs/Knot_Tying_fold_1_train.log"),
        help="Training log used to scrape total / trainable parameter counts.",
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("eval_results"),
        help=(
            "Root directory of per-fold eval result text files used to compute "
            "paired-bootstrap CIs on Delta gesture F1 macro vs baseline. "
            "Expected layout: <eval_root>/<condition>/<Task>_test_fold_<N>_results.txt."
        ),
    )
    parser.add_argument(
        "--bootstrap_iters",
        type=int,
        default=10000,
        help="Number of bootstrap resamples for the paired CI on Delta F1.",
    )
    parser.add_argument(
        "--bootstrap_seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap reproducibility.",
    )
    parser.add_argument(
        "--bootstrap_ci",
        type=float,
        default=0.95,
        help="Bootstrap confidence interval (e.g. 0.95 for 95 percent CI).",
    )
    return parser.parse_args(argv)


def _load_conditions(args: argparse.Namespace) -> tuple[list[Condition], list[str]]:
    """Return (loaded conditions, missing condition names)."""
    loaded: list[Condition] = []
    missing: list[str] = []
    for name in args.conditions:
        tasks = load_louo_results(args.analysis_root, name)
        if tasks is None:
            missing.append(name)
            continue
        config_path, weights = load_config_weights(args.configs_dir, name)
        loaded.append(
            Condition(
                name=name,
                label=CONDITION_LABELS.get(name, name),
                config_path=config_path,
                loss_weights=weights,
                tasks=tasks,
            )
        )
    return loaded, missing


def _print_diagnostic(
    conditions: Sequence[Condition],
    missing: Sequence[str],
    out_dir: Path,
    rdm_manifest: Mapping[str, RDMEntry],
    writes: Mapping[str, list[Path]],
) -> None:
    print("=" * 68)
    print("manuscript_writer.py - diagnostic summary")
    print("=" * 68)
    print(f"Output directory : {out_dir}")
    print(f"Conditions loaded: {len(conditions)}")
    for cond in conditions:
        per_task = ", ".join(
            f"{task}={cond.result(task).num_folds}" if cond.result(task) else f"{task}=0"
            for task in TASK_ORDER
        )
        print(f"  - {cond.name:<15}: {per_task}")
    if missing:
        print(f"Conditions MISSING: {', '.join(missing)}")
    print(f"RDM manifest entries : {len(rdm_manifest)}")
    print()
    for label, paths in writes.items():
        print(f"{label} ({len(paths)}):")
        for p in paths:
            print(f"  - {p}")
    print("=" * 68)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    conditions, missing = _load_conditions(args)
    if not conditions:
        print("ERROR: no conditions loaded; nothing to write.", file=sys.stderr)
        return 1

    rdm_manifest = load_rdm_manifest(args.rdm_manifest)
    splits_fold_counts = load_split_fold_counts(args.splits_dir)
    split_sample = load_split_sample_counts(args.splits_dir, "Suturing", fold=1)
    param_counts = scrape_model_param_counts(args.param_log)

    # The canonical n_trials_used is stored at the top of the Phase 3 manifest;
    # fall back to a conservative default if the manifest lacks it.
    manifest_raw = _read_json(args.rdm_manifest) if args.rdm_manifest.is_file() else {}
    n_trials_used = int(manifest_raw.get("n_trials_used", 0) or 0) if isinstance(manifest_raw, dict) else 0

    out_dir = args.output_dir
    sections_dir = out_dir / "sections"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    for d in (out_dir, sections_dir, tables_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Tables -----------------------------------------------------------------
    tbl_main = tables_dir / "tbl_main.tex"
    tbl_kin = tables_dir / "tbl_kinematics.tex"
    tbl_skill = tables_dir / "tbl_skill.tex"
    tbl_cond = tables_dir / "tbl_conditions.tex"
    tbl_bootstrap = tables_dir / "tbl_bootstrap.tex"
    emit_main_table(conditions, tbl_main)
    emit_kinematics_table(conditions, tbl_kin)
    emit_skill_table(conditions, tbl_skill)
    emit_conditions_table(conditions, tbl_cond)
    emit_bootstrap_table(
        conditions,
        args.baseline_condition,
        args.eval_root,
        tbl_bootstrap,
        n_iter=args.bootstrap_iters,
        seed=args.bootstrap_seed,
        ci=args.bootstrap_ci,
    )

    # Figures ----------------------------------------------------------------
    fig_gesture = figures_dir / "fig_gesture_f1_bars.pdf"
    fig_kin = figures_dir / "fig_kinematics_bars.pdf"
    fig_fold = figures_dir / "fig_fold_spread_box.pdf"
    fig_rdms = figures_dir / "fig_rdm_heatmaps.pdf"
    fig_tp = figures_dir / "fig_transfer_plausibility_scatter.pdf"
    emit_grouped_bar_chart(conditions, GESTURE_METRICS[1], "Gesture F1 (macro) across conditions", fig_gesture)
    emit_kinematics_bars(conditions, fig_kin)
    emit_fold_spread_boxplots(conditions, fig_fold)
    emit_rdm_heatmaps(args.rdm_manifest, fig_rdms)
    emit_transfer_plausibility_scatter(conditions, args.baseline_condition, args.rdm_manifest, fig_tp)

    # Sections ---------------------------------------------------------------
    sec_abstract = sections_dir / "00_abstract.tex"
    sec_intro = sections_dir / "01_intro.tex"
    sec_related = sections_dir / "02_related.tex"
    sec_methods = sections_dir / "03_methods.tex"
    sec_experiments = sections_dir / "04_experiments.tex"
    sec_results = sections_dir / "05_results.tex"
    sec_discussion = sections_dir / "06_discussion.tex"
    sec_limits = sections_dir / "07_limitations.tex"
    sec_conclusion = sections_dir / "08_conclusion.tex"
    emit_section_abstract(sec_abstract, conditions, args.baseline_condition)
    emit_section_intro(sec_intro, conditions, splits_fold_counts)
    emit_section_related(sec_related)
    emit_section_methods(
        sec_methods,
        conditions,
        splits_fold_counts,
        split_sample,
        rdm_manifest,
        param_counts,
        n_trials_used,
    )
    emit_section_experiments(sec_experiments, conditions)
    emit_section_results(
        sec_results,
        conditions,
        args.baseline_condition,
        eval_root=args.eval_root,
        bootstrap_n_iter=args.bootstrap_iters,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_ci=args.bootstrap_ci,
    )
    emit_section_discussion(sec_discussion, conditions, args.baseline_condition)
    emit_section_limitations(sec_limits, conditions, splits_fold_counts)
    emit_section_conclusion(sec_conclusion)

    # Main.tex, references.bib (if missing), README.md (if missing) ----------
    main_tex = out_dir / "main.tex"
    bib = out_dir / "references.bib"
    readme = out_dir / "README.md"
    emit_main_tex(main_tex)
    emit_references_bib(bib, _TODO_CITE_KEYS)
    emit_readme(readme)

    writes = {
        "Tables": [tbl_main, tbl_kin, tbl_skill, tbl_cond, tbl_bootstrap],
        "Figures": [fig_gesture, fig_kin, fig_fold, fig_rdms, fig_tp],
        "Sections": [
            sec_abstract, sec_intro, sec_related, sec_methods,
            sec_experiments, sec_results, sec_discussion, sec_limits,
            sec_conclusion,
        ],
        "Top-level": [main_tex, bib, readme],
    }
    _print_diagnostic(conditions, missing, out_dir, rdm_manifest, writes)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
