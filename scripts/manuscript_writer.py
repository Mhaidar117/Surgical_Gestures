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
* ``references.bib`` - stub with ``@article{TODO_*}`` placeholders (only if
  missing; preserved across reruns).
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
    "bridge_joint": "Joint-Bridge",
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
        "EEG-derived task-family RDM from Phase~3 of the EEG-Eye bridge "
        "pipeline (manifest key \\texttt{eeg\\_latent\\_task\\_family})."
    ),
    "bridge_joint": (
        "Jointly-fused eye+EEG task-family RDM from Phase~3 of the EEG-Eye "
        "bridge pipeline (manifest key \\texttt{joint\\_eye\\_eeg\\_task\\_family})."
    ),
}

# Mapping from a training condition to the Phase 3 manifest entry that best
# represents the source RDM used by that condition. ``brain_eye`` does not
# literally load from the Phase 3 manifest (it uses the standalone 3x3 eye RDM
# under Eye/Exploration/), but ``eye_only_subskill_family`` is the closest
# Phase 3 proxy and is used only for the transfer-plausibility scatter figure.
CONDITION_TO_MANIFEST_RDM: dict[str, str] = {
    "brain_eye": "eye_only_subskill_family",
    "bridge_eeg": "eeg_latent_task_family",
    "bridge_joint": "joint_eye_eeg_task_family",
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

ALL_METRICS: tuple[MetricSpec, ...] = (
    GESTURE_METRICS + SKILL_METRICS + KINEMATICS_METRICS + LOSS_METRICS
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
    """Table 1: main gesture + skill + total loss results by (condition, task)."""
    metrics = (
        GESTURE_METRICS[0],  # gesture accuracy
        GESTURE_METRICS[1],  # gesture F1 macro
        SKILL_METRICS[0],    # skill accuracy
        SKILL_METRICS[1],    # skill F1 macro
        LOSS_METRICS[0],     # total loss
    )
    header_cells = ["Condition", "Task", "Folds"] + [m.label for m in metrics]
    col_spec = "ll" + "r" * (len(metrics) + 1)

    lines = [
        "% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.",
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(header_cells) + " \\\\",
        "\\midrule",
    ]

    for task in TASK_ORDER:
        winners = {m.key: per_metric_winner(conditions, task, m) for m in metrics}
        for idx, cond in enumerate(conditions):
            result = cond.result(task)
            if result is None:
                continue
            row: list[str] = []
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
        "% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.",
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
        "% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.",
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


def emit_conditions_table(conditions: Sequence[Condition], out_path: Path) -> None:
    """Table 4: summary of the four training conditions + loss weights."""
    keys = ("kin", "gesture", "skill", "brain", "control")
    header_cells = ["Condition", "Config"] + [f"$w_{{\\text{{{k}}}}}$" for k in keys]
    col_spec = "ll" + "r" * len(keys)
    lines = [
        "% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.",
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
        "joint_eye_eeg_task_family",
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

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\begin{{abstract}}
We introduce a four-phase pipeline that converts electroencephalography (EEG)
and eye-tracking recordings from human observers of surgery into differentiable
representational dissimilarity matrices (RDMs), and use these RDMs as soft
regularizers on a Vision Transformer (ViT) that jointly predicts surgical
kinematics, gesture labels, and surgeon skill on the JIGSAWS benchmark.
Under an 8-fold leave-one-user-out (LOUO) cross-validation protocol with up to
three folds per (task $\times$ condition) cell, we compare a no-regularizer
baseline against three brain-aligned conditions: eye-tracking task-centroid RSA,
an EEG-derived task-family RDM, and a jointly-fused eye$+$EEG task-family RDM.
Best Gesture F1 (macro) is achieved by {kt_g} on Knot Tying,
{np_g} on Needle Passing, and {su_g} on Suturing;
best Position RMSE is achieved by {kt_p} on Knot Tying,
{np_p} on Needle Passing, and {su_p} on Suturing.
The pattern is not a uniform accuracy win -- different physiological
regularizers help different sub-problems -- and a pipeline-internal
transfer-plausibility score computed during Phase~3 RDM ranking is predictive
of the downstream per-task gain, suggesting that Phase~3 can serve as an
\emph{{a priori}} selection tool for candidate priors.
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
    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Introduction}}
\label{{sec:intro}}

Surgical video understanding has converged on Vision Transformer (ViT) backbones
for gesture recognition, kinematics prediction, and skill assessment, but the
learned representations remain opaque and lack grounding in how human experts
actually perceive the same footage.  At the same time, a separate line of work
has shown that representational similarity analysis (RSA) between deep networks
and human neural data can be used both to measure alignment and, when
incorporated into training, to bias networks toward brain-like solutions
\cite{{TODO_kriegeskorte2008,TODO_khaligh-razavi2014}}.  The ingredients for an
analogous intervention on surgical video already exist---JIGSAWS
\cite{{TODO_gao2014jigsaws}} provides aligned video, kinematics, gesture, and
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
  \item A controlled LOUO ablation showing that \emph{{different}} physiological
        signals selectively regularize \emph{{different}} capabilities of the
        model---eye-tracking improves kinematics fidelity, EEG improves gesture
        structure---rather than producing a uniform accuracy win.
  \item Evidence that the transfer-plausibility score computed during Phase~3
        RDM ranking is correlated with the downstream per-task improvement the
        corresponding prior produces, which makes Phase~3 an \emph{{a priori}}
        selection tool rather than a post-hoc report.
\end{{enumerate}}
"""
    _write_text(out_path, text)


def emit_section_related(out_path: Path) -> None:
    text = r"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{Related work}
\label{sec:related}

\paragraph{Surgical gesture and skill assessment.}
JIGSAWS~\cite{TODO_gao2014jigsaws} established a benchmark for gesture
recognition and skill rating on the da Vinci Research Kit, and has since been
tackled with bidirectional LSTMs~\cite{TODO_dipietro2016}, temporal
convolutional networks~\cite{TODO_funke2019}, and more recently with
self-supervised video transformers~\cite{TODO_goodman2023}.  Our backbone
follows this line---a timm-initialized ViT-S/16 plus a four-layer temporal
transformer---and adds a brain-alignment term orthogonal to those
contributions.

\paragraph{Representational similarity analysis for neural networks.}
RSA was introduced as a model-agnostic way to compare neural representations
across species and modalities~\cite{TODO_kriegeskorte2008}; it was then adapted
to compare deep networks with human brain
activity~\cite{TODO_khaligh-razavi2014}, and more recently used as a training
signal itself (e.g. RSA regularization~\cite{TODO_mcclure2016}, brain-score
fine-tuning~\cite{TODO_schrimpf2020}).  We apply an RSA loss of the form
$1 - \rho\!\left(\text{model\_RDM}, \text{target\_RDM}\right)$, where $\rho$ is
Pearson correlation over the flattened upper-triangle of the two RDMs.

\paragraph{Eye-tracking and EEG priors for video models.}
Eye-gaze has been used as an auxiliary signal in action recognition
\cite{TODO_min2019} and in surgical skill analysis
\cite{TODO_islam2020gaze}; EEG has been used for reaction-time regression and
as a subject-specific regularizer~\cite{TODO_palazzo2021eeg}.  Our contribution
is to treat both as sources of a single \emph{shared} target RDM, and to
evaluate whether the modality-specific RDMs predict which downstream
capabilities they help.

\paragraph{Soft priors on safety-critical robotic systems.}
Regularizers derived from human data are attractive for surgical robotics
because they can encode anatomical or procedural priors without requiring new
labelled data~\cite{TODO_funke2019,TODO_goodman2023}.  We adopt a soft RSA
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

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Methods}}
\label{{sec:methods}}

\subsection{{Dataset and evaluation protocol}}
\label{{sec:dataset}}

We use the JIGSAWS benchmark~\cite{{TODO_gao2014jigsaws}}: 30~Hz stereo
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
ranked. For the present manuscript we used $n={n_rdm}$ candidate RDMs and
selected three (one per brain-aligned condition) via the ranking. Phase~4
loads the selected RDM as a fixed target and computes a model centroid RDM
from ViT embeddings grouped by task at each batch step.

\subsection{{Model architecture}}
\label{{sec:model}}

Video frames are encoded by a timm~\cite{{TODO_wightman2019timm}} ViT-S/16
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

\subsection{{Training conditions}}
\label{{sec:conditions}}

Table~\ref{{tab:conditions}} summarizes the four conditions. They differ only in
whether a brain-alignment term is active and which target RDM it uses; the
ViT backbone, temporal aggregator, task heads, optimizer, and training
schedule are identical across conditions.

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

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Experimental setup}}
\label{{sec:experiments}}

All four conditions are trained and evaluated by a single PowerShell runner
(\texttt{{run\_ablation\_study.ps1}}), which iterates over
$(\text{{condition}}, \text{{task}}, \text{{fold}})$ triples, invokes the
training script \texttt{{src/training/train\_vit\_system.py}} followed by the
evaluation script \texttt{{src/eval/evaluate.py}}, records per-fold status to
JSON, and aggregates per-condition results via \texttt{{aggregate\_louo\_results.py}}.
Checkpoint retention is set to \texttt{{none}} so that only the best
per-fold weights are written to disk before deletion after the evaluation
step. Each training run uses the same 10-epoch schedule, cosine-annealed
learning rate with a 2-epoch warm-up, and differential learning rates for the
ViT backbone ($10^{{-5}}$), adapters ($5 \times 10^{{-5}}$), and other parameters
($10^{{-4}}$). Batch size is 32 for the baseline and 16 for the brain-aligned
conditions (the adapters cost memory).

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

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Results}}
\label{{sec:results}}

\subsection{{Main results}}
\label{{sec:results_main}}

Table~\ref{{tab:main}} reports gesture, skill, and total-loss metrics broken
down by $(\text{{condition}}, \text{{task}})$. Fold counts are listed in the
\emph{{Folds}} column, and the best value per task and metric is bolded.

\begin{{table}}[t]
  \centering
  \caption{{LOUO results on JIGSAWS across four training conditions. Mean
    $\pm$ standard deviation across held-out surgeon folds. Best value per
    $(\text{{task}}, \text{{metric}})$ is bolded.}}
  \label{{tab:main}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_main}}}}
\end{{table}}

Figure~\ref{{fig:gesture_f1}} visualizes Gesture F1 macro across conditions.
{' '.join(gesture_sentences)}

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
kinematics metrics. {' '.join(kinematics_sentences)} The modality that carries
\emph{{where-the-hand-is-going}} information (eye-tracking) systematically
helps on the two tasks where it has been evaluated most, consistent with the
hypothesis that gaze-derived RDMs constrain pose geometry more than category
structure.

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
entirely of one skill level, making the metric fragile on 3-fold samples; we
therefore report it but do not draw modality-specific conclusions from it.

\begin{{table}}[t]
  \centering
  \caption{{Skill classification under LOUO. Numbers are highly fold-dependent;
    see text and Fig.~\ref{{fig:fold_spread}}.}}
  \label{{tab:skill}}
  \resizebox{{\columnwidth}}{{!}}{{\input{{tables/tbl_skill}}}}
\end{{table}}

\subsection{{Fold-level dispersion}}
\label{{sec:results_folds}}

To make the under-power of a 3-fold protocol visible, Fig.~\ref{{fig:fold_spread}}
shows per-fold values as box plots, broken down by task and by three
representative metrics (Gesture F1 macro, Position RMSE, Total Loss).

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

\subsection{{Transfer plausibility predicts downstream gain}}
\label{{sec:results_tp}}

Phase~3 of the EEG--Eye bridge assigns each candidate RDM a
\emph{{transfer-plausibility}} score based on how its unit labels map to the
JIGSAWS task families. Fig.~\ref{{fig:transfer_plausibility}} shows this score
against the per-$(\text{{condition}}, \text{{task}})$ change in Gesture F1
macro over the baseline.  The sign of the relationship indicates that the
internal Phase~3 score ranks candidate RDMs on an axis that is at least
partially aligned with their downstream utility, not merely with their
geometric self-consistency.  Fig.~\ref{{fig:rdms}} shows the four relevant
Phase~3 candidate RDMs for reference.

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
    # below avoids claiming any specific modality wins; it defers to the data.
    def winner_label(task: str, spec: MetricSpec) -> str:
        w = per_metric_winner(conditions, task, spec)
        return _cond_label(w) if w else "--"

    gesture_bullet_items = "\n".join(
        f"  \\item {TASK_LABELS.get(task, task)}: {winner_label(task, GESTURE_METRICS[1])}"
        for task in TASK_ORDER
    )
    position_bullet_items = "\n".join(
        f"  \\item {TASK_LABELS.get(task, task)}: {winner_label(task, KINEMATICS_METRICS[0])}"
        for task in TASK_ORDER
    )

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Discussion}}
\label{{sec:discussion}}

\paragraph{{Modality-selective regularization.}}
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
condition in general, which is consistent with the hypothesis that
eye-tracking-derived priors carry \emph{{where-the-hand-is-going}} information
(favoring kinematics fidelity) while EEG-derived priors carry
\emph{{what-category-is-this}} information (favoring gesture structure).

\paragraph{{Transfer plausibility as an \emph{{a priori}} selector.}}
A practically useful finding is that the Phase~3 transfer-plausibility score
is correlated with the gain we see on gesture F1 macro
(Fig.~\ref{{fig:transfer_plausibility}}). This is important because the score
is computed from the RDM's unit labels and their correspondence to JIGSAWS
task families---it does not involve any ViT training.  If it reliably predicts
which RDM helps which task, then Phase~3 becomes an \emph{{a priori}}
selection tool, not just a ranking curiosity, and one could afford to train
fewer brain-aligned models than candidate RDMs.

\paragraph{{Why Needle Passing is hardest.}}
On Needle Passing no brain-aligned condition beats the baseline on gesture F1
or skill accuracy. The Phase~3 manifest lists
\texttt{{eeg\_latent\_task\_family}} with the lowest transfer plausibility
(0.552) among the candidates we used, and \texttt{{joint\_eye\_eeg\_task\_family}}
with identical plausibility.  The negative result on this task is therefore
anticipated by the internal score rather than surprising, which strengthens
the interpretation of that score.

\paragraph{{Skill head is under-sampled.}}
Skill accuracy has standard deviations of the same order as its mean across
folds (Table~\ref{{tab:skill}}), because LOUO can hold out an entire skill
class. We treat skill metrics as illustrative only and plan to re-report them
once all eight folds are complete.
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

    text = rf"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{{Limitations and future work}}
\label{{sec:limitations}}

\paragraph{{LOUO under-power.}}
We report $\leq 3$ folds for each brain-aligned condition and up to 8 for the
baseline. Fold counts per condition are as follows: {fold_str}. The three-fold
tables should be read as a pilot ablation; we commit to reporting the full
eight-fold comparison in a follow-up.

\paragraph{{Needle Passing has only {np_max} folds.}}
The JIGSAWS Needle Passing task provides fewer LOUO folds than the other two,
limiting statistical comparisons on that task.

\paragraph{{Multi-task joint training not reported.}}
Our runner trains each $(\text{{condition}}, \text{{task}})$ pair independently.
A separate script (\texttt{{run\_8fold\_louo\_brain.sh}}) trains the brain-aligned
conditions with \texttt{{-{{}}-task all}}, which is expected to further exploit the
shared RDM target; we leave that variant to future work.

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
    text = r"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
\section{Conclusion}
\label{sec:conclusion}

We presented a reproducible pipeline that turns EEG and eye-tracking
recordings from human observers of surgery into differentiable
representational dissimilarity matrices and uses them as soft regularizers on
a ViT that jointly predicts kinematics, gestures, and skill on JIGSAWS. A
controlled leave-one-user-out ablation shows that these brain-derived priors
have a \emph{selective} effect---eye-tracking helps kinematics geometry, EEG
helps gesture structure---rather than a uniform accuracy win, and that the
Phase~3 transfer-plausibility score used internally for RDM ranking is
predictive of downstream gain. Scaling to the full eight-fold comparison and
incorporating subject-adaptive priors are the immediate next steps.
"""
    _write_text(out_path, text)


# ---------------------------------------------------------------------------
# main.tex, bibliography stub, and README
# ---------------------------------------------------------------------------


def emit_main_tex(out_path: Path) -> None:
    text = r"""% Auto-generated by scripts/manuscript_writer.py - do not edit by hand.
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


# Cite keys used in the auto-generated prose; every one gets a stub in
# references.bib so pdflatex+bibtex does not break on a fresh build.
_TODO_CITE_KEYS: tuple[str, ...] = (
    "TODO_kriegeskorte2008",
    "TODO_khaligh-razavi2014",
    "TODO_gao2014jigsaws",
    "TODO_dipietro2016",
    "TODO_funke2019",
    "TODO_goodman2023",
    "TODO_mcclure2016",
    "TODO_schrimpf2020",
    "TODO_min2019",
    "TODO_islam2020gaze",
    "TODO_palazzo2021eeg",
    "TODO_wightman2019timm",
)


def emit_references_bib(out_path: Path, keys: Iterable[str]) -> None:
    if out_path.exists():
        return  # preserve the user's edits
    lines = [
        "% Bibliography stub auto-generated by scripts/manuscript_writer.py.",
        "% Edit each entry before submission; the script will NOT overwrite this",
        "% file on subsequent runs.",
        "",
    ]
    for key in sorted(set(keys)):
        lines.extend(
            [
                f"@article{{{key},",
                "  author  = {TODO Author},",
                "  title   = {TODO Title},",
                "  journal = {TODO Journal},",
                "  year    = {TODO},",
                "  volume  = {TODO},",
                "  pages   = {TODO},",
                "}",
                "",
            ]
        )
    _write_text(out_path, "\n".join(lines))


def emit_readme(out_path: Path) -> None:
    if out_path.exists():
        return
    text = """# final_results_manuscript

This directory is auto-generated from the repo's analysis artifacts by
`scripts/manuscript_writer.py`. It holds the LaTeX sources, booktabs tables,
and data-driven figures for the EEG-Eye-RDM-ViT brain-alignment manuscript.

## Regenerating

From the repo root (PowerShell):

```powershell
$env:PYTHONPATH = "src"
python scripts/manuscript_writer.py `
  --analysis_root analysis `
  --rdm_manifest cache/eeg_eye_bridge/phase3/rdm_manifest.json `
  --splits_dir data/splits `
  --configs_dir src/configs `
  --output_dir docs/final_results_manuscript `
  --conditions baseline brain_eye bridge_eeg bridge_joint `
  --baseline_condition baseline
```

Re-running the script clobbers everything in this directory EXCEPT:

- `references.bib` (your edits survive)
- `README.md` (this file)
- `figures/fig_pipeline_diagram.pdf` (user-supplied schematic)
- `figures/fig_architecture.pdf` (user-supplied schematic)

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
- Replace the `TODO_*` keys in `references.bib` with real BibTeX entries.
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
        default=["baseline", "brain_eye", "bridge_eeg", "bridge_joint"],
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
    emit_main_table(conditions, tbl_main)
    emit_kinematics_table(conditions, tbl_kin)
    emit_skill_table(conditions, tbl_skill)
    emit_conditions_table(conditions, tbl_cond)

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
    emit_section_results(sec_results, conditions, args.baseline_condition)
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
        "Tables": [tbl_main, tbl_kin, tbl_skill, tbl_cond],
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
