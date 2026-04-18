#!/usr/bin/env python3
"""
Export writeup-ready analysis artifacts from per-condition LOUO summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


TASK_ORDER = ["Knot_Tying", "Needle_Passing", "Suturing"]

METRIC_SPECS = [
    {
        "json_key": "gesture_accuracy",
        "column": "gesture_accuracy",
        "label": "Gesture Accuracy (%)",
        "goal": "max",
    },
    {
        "json_key": "gesture_f1_macro",
        "column": "gesture_f1_macro",
        "label": "Gesture F1 Macro",
        "goal": "max",
    },
    {
        "json_key": "gesture_f1_micro",
        "column": "gesture_f1_micro",
        "label": "Gesture F1 Micro",
        "goal": "max",
    },
    {
        "json_key": "skill_accuracy",
        "column": "skill_accuracy",
        "label": "Skill Accuracy (%)",
        "goal": "max",
    },
    {
        "json_key": "skill_f1_macro",
        "column": "skill_f1_macro",
        "label": "Skill F1 Macro",
        "goal": "max",
    },
    {
        "json_key": "kinematics_position_rmse",
        "column": "position_rmse",
        "label": "Position RMSE",
        "goal": "min",
    },
    {
        "json_key": "kinematics_end-effector_error",
        "column": "end_effector_error",
        "label": "End-Effector Error",
        "goal": "min",
    },
    {
        "json_key": "kinematics_rotation_rmse",
        "column": "rotation_rmse",
        "label": "Rotation RMSE",
        "goal": "min",
    },
    {
        "json_key": "loss_total_loss",
        "column": "total_loss",
        "label": "Total Loss",
        "goal": "min",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CSV/TXT artifacts for ablation analysis and writeup."
    )
    parser.add_argument(
        "--analysis_root",
        type=Path,
        default=Path("analysis"),
        help="Root containing per-condition louo_results.json outputs.",
    )
    parser.add_argument(
        "--eval_root",
        type=Path,
        default=Path("eval_results"),
        help="Root containing per-condition evaluation text files.",
    )
    parser.add_argument(
        "--runner_script",
        type=str,
        default="run_ablation_study.ps1",
        help="Runner script path to reference in the generated usage prompt.",
    )
    parser.add_argument(
        "--retain_checkpoints",
        type=str,
        default="none",
        help="Checkpoint retention mode used by the run.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        required=True,
        help="Condition names to export.",
    )
    return parser.parse_args()


def write_csv(path: Path, fieldnames: Iterable[str], rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_json(path: Path) -> object:
    raw = path.read_bytes()
    last_error: Exception | None = None

    # Older PowerShell runs may have written JSON with a Windows code page
    # instead of UTF-8, so accept a small set of safe fallbacks.
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            return json.loads(raw.decode(encoding))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            last_error = exc

    raise ValueError(f"Unable to parse JSON file '{path}': {last_error}") from last_error


def load_metadata(path: Path) -> Dict:
    if not path.is_file():
        return {}
    return load_json(path)


def status_priority(status: str) -> int:
    if status == "completed":
        return 3
    if status.startswith("skipped_"):
        return 2
    return 1


def normalize_statuses(rows: List[Dict]) -> List[Dict]:
    deduped: Dict[Tuple[str, str, str, str], Dict] = {}

    for row in rows:
        key = (
            str(row.get("condition", "")),
            str(row.get("task", "")),
            str(row.get("fold", "")),
            str(row.get("stage", "")),
        )
        current = deduped.get(key)
        if current is None:
            deduped[key] = row
            continue

        current_status = str(current.get("status", ""))
        candidate_status = str(row.get("status", ""))
        if status_priority(candidate_status) > status_priority(current_status):
            deduped[key] = row
            continue

        if status_priority(candidate_status) == status_priority(current_status):
            current_timestamp = str(current.get("timestamp", ""))
            candidate_timestamp = str(row.get("timestamp", ""))
            if candidate_timestamp >= current_timestamp:
                deduped[key] = row

    return list(deduped.values())


def load_statuses(path: Path) -> List[Dict]:
    if not path.is_file():
        return []
    data = load_json(path)
    return normalize_statuses(data) if isinstance(data, list) else []


def ordered_tasks(results: Dict[str, Dict]) -> List[str]:
    tasks = list(results.keys())
    return sorted(tasks, key=lambda task: TASK_ORDER.index(task) if task in TASK_ORDER else len(TASK_ORDER))


def fmt_mean_std(mean: float | None, std: float | None, decimals: int = 4) -> str:
    if mean is None or std is None:
        return ""
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def metric_stat(task_payload: Dict, json_key: str) -> Dict | None:
    return (task_payload.get("metrics") or {}).get(json_key)


def choose_best(rows: List[Dict[str, object]], metric_column: str, goal: str) -> Tuple[str, float] | Tuple[None, None]:
    ranked = [
        (row["condition"], float(row[f"{metric_column}_mean"]))
        for row in rows
        if row.get(f"{metric_column}_mean") not in ("", None)
    ]
    if not ranked:
        return None, None
    if goal == "max":
        return max(ranked, key=lambda item: item[1])
    return min(ranked, key=lambda item: item[1])


def build_usage_prompt(
    runner_script: str,
    conditions: List[str],
    retain_checkpoints: str,
    analysis_root: Path,
    eval_root: Path,
) -> str:
    joined_conditions = ",".join(conditions)
    return "\n".join(
        [
            "Use the following prompt in a future session to run the full ablation pipeline:",
            "",
            f"Run `.\\{runner_script} -Conditions {joined_conditions} -RetainCheckpoints {retain_checkpoints}` from the repo root in PowerShell.",
            "",
            "Useful variations:",
            f"- Smoke test one fold/task per condition: `.\\{runner_script} -Conditions baseline,brain_eye -Tasks Suturing -StartFold 1 -EndFold 1 -RetainCheckpoints {retain_checkpoints}`",
            f"- Keep best checkpoints for debugging: `.\\{runner_script} -RetainCheckpoints best`",
            f"- Dry-run the command plan only: `.\\{runner_script} -DryRun`",
            "",
            "Expected outputs after a successful full run:",
            f"- Per-condition eval files under `{eval_root.as_posix()}/<condition>/`",
            f"- Per-condition summaries under `{analysis_root.as_posix()}/<condition>/louo_summary.txt` and `louo_results.json`",
            f"- Comparison tables and figure CSVs under `{(analysis_root / 'comparisons').as_posix()}/`",
            "- A writeup summary report and this run-instructions prompt file",
            "",
            "Do not consider the project ready for writeup until the following exist for every condition:",
            "- `louo_results.json`",
            "- `task_metrics.csv`",
            "- `fold_metrics.csv`",
            "- comparison CSVs under `analysis/comparisons/`",
            "- `writeup_summary.txt`",
        ]
    )


def main() -> None:
    args = parse_args()
    comparison_dir = args.analysis_root / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows: List[Dict[str, object]] = []
    figure_fold_rows: List[Dict[str, object]] = []
    condition_status_rows: List[Dict[str, object]] = []
    missing_conditions: List[str] = []

    for condition in args.conditions:
        condition_dir = args.analysis_root / condition
        louo_json = condition_dir / "louo_results.json"
        metadata = load_metadata(condition_dir / "run_metadata.json")
        statuses = load_statuses(condition_dir / "run_status.json")
        condition_status_rows.extend(statuses)

        if not louo_json.is_file():
            missing_conditions.append(condition)
            continue

        results = load_json(louo_json)

        task_rows: List[Dict[str, object]] = []
        fold_rows: List[Dict[str, object]] = []

        for task in ordered_tasks(results):
            task_payload = results[task]
            num_folds = task_payload.get("num_folds", 0)

            wide_row: Dict[str, object] = {
                "condition": condition,
                "task": task,
                "num_folds": num_folds,
                "config_path": metadata.get("config_path", ""),
                "retain_checkpoints": metadata.get("retain_checkpoints", args.retain_checkpoints),
            }

            for spec in METRIC_SPECS:
                stats = metric_stat(task_payload, spec["json_key"])
                if stats is None:
                    wide_row[f"{spec['column']}_mean"] = ""
                    wide_row[f"{spec['column']}_std"] = ""
                    wide_row[f"{spec['column']}_formatted"] = ""
                    continue

                mean = stats.get("mean")
                std = stats.get("std")
                min_value = stats.get("min")
                max_value = stats.get("max")
                values = stats.get("values") or []

                task_rows.append(
                    {
                        "condition": condition,
                        "task": task,
                        "metric_key": spec["json_key"],
                        "metric_column": spec["column"],
                        "metric_label": spec["label"],
                        "goal": spec["goal"],
                        "num_folds": num_folds,
                        "mean": mean,
                        "std": std,
                        "min": min_value,
                        "max": max_value,
                    }
                )

                for fold_idx, value in enumerate(values, start=1):
                    fold_row = {
                        "condition": condition,
                        "task": task,
                        "metric_key": spec["json_key"],
                        "metric_column": spec["column"],
                        "metric_label": spec["label"],
                        "fold": fold_idx,
                        "value": value,
                    }
                    fold_rows.append(fold_row)
                    figure_fold_rows.append(fold_row)

                wide_row[f"{spec['column']}_mean"] = mean
                wide_row[f"{spec['column']}_std"] = std
                wide_row[f"{spec['column']}_formatted"] = fmt_mean_std(mean, std)

            comparison_rows.append(wide_row)

        write_csv(
            condition_dir / "task_metrics.csv",
            [
                "condition",
                "task",
                "metric_key",
                "metric_column",
                "metric_label",
                "goal",
                "num_folds",
                "mean",
                "std",
                "min",
                "max",
            ],
            task_rows,
        )
        write_csv(
            condition_dir / "fold_metrics.csv",
            ["condition", "task", "metric_key", "metric_column", "metric_label", "fold", "value"],
            fold_rows,
        )

    comparison_fieldnames = [
        "condition",
        "task",
        "num_folds",
        "config_path",
        "retain_checkpoints",
    ]
    for spec in METRIC_SPECS:
        comparison_fieldnames.extend(
            [
                f"{spec['column']}_mean",
                f"{spec['column']}_std",
                f"{spec['column']}_formatted",
            ]
        )

    write_csv(comparison_dir / "condition_comparison.csv", comparison_fieldnames, comparison_rows)

    write_csv(
        comparison_dir / "paper_table_main.csv",
        [
            "condition",
            "task",
            "gesture_accuracy_formatted",
            "gesture_f1_macro_formatted",
            "gesture_f1_micro_formatted",
            "skill_accuracy_formatted",
            "skill_f1_macro_formatted",
            "total_loss_formatted",
        ],
        comparison_rows,
    )

    write_csv(
        comparison_dir / "paper_table_kinematics.csv",
        [
            "condition",
            "task",
            "position_rmse_formatted",
            "end_effector_error_formatted",
            "rotation_rmse_formatted",
            "total_loss_formatted",
        ],
        comparison_rows,
    )

    write_csv(
        comparison_dir / "paper_table_skill.csv",
        [
            "condition",
            "task",
            "skill_accuracy_formatted",
            "skill_f1_macro_formatted",
            "gesture_accuracy_formatted",
            "gesture_f1_macro_formatted",
        ],
        comparison_rows,
    )

    gesture_figure_rows = []
    for row in comparison_rows:
        for metric in ("gesture_accuracy", "gesture_f1_macro", "gesture_f1_micro"):
            gesture_figure_rows.append(
                {
                    "condition": row["condition"],
                    "task": row["task"],
                    "metric": metric,
                    "value": row.get(f"{metric}_mean", ""),
                    "std": row.get(f"{metric}_std", ""),
                }
            )
    write_csv(
        comparison_dir / "figure_bar_gesture_f1.csv",
        ["condition", "task", "metric", "value", "std"],
        gesture_figure_rows,
    )

    kinematics_figure_rows = []
    for row in comparison_rows:
        for metric in ("position_rmse", "end_effector_error", "rotation_rmse"):
            kinematics_figure_rows.append(
                {
                    "condition": row["condition"],
                    "task": row["task"],
                    "metric": metric,
                    "value": row.get(f"{metric}_mean", ""),
                    "std": row.get(f"{metric}_std", ""),
                }
            )
    write_csv(
        comparison_dir / "figure_bar_kinematics.csv",
        ["condition", "task", "metric", "value", "std"],
        kinematics_figure_rows,
    )

    write_csv(
        comparison_dir / "figure_fold_spread.csv",
        ["condition", "task", "metric_key", "metric_column", "metric_label", "fold", "value"],
        figure_fold_rows,
    )

    write_csv(
        comparison_dir / "run_status_summary.csv",
        ["condition", "task", "fold", "stage", "status", "error_message", "log_path", "timestamp"],
        condition_status_rows,
    )
    write_csv(
        comparison_dir / "failed_runs.csv",
        ["condition", "task", "fold", "stage", "status", "error_message", "log_path", "timestamp"],
        [row for row in condition_status_rows if row.get("status") != "completed"],
    )

    summary_lines = [
        "Full 8-fold ablation summary",
        "============================",
        "",
        f"Conditions processed: {', '.join(args.conditions)}",
        f"Checkpoint retention: {args.retain_checkpoints}",
        f"Analysis root: {args.analysis_root}",
        f"Eval root: {args.eval_root}",
        "",
    ]

    if missing_conditions:
        summary_lines.append("Conditions without aggregate JSON:")
        for condition in missing_conditions:
            summary_lines.append(f"- {condition}")
        summary_lines.append("")

    rows_by_task: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in comparison_rows:
        rows_by_task[str(row["task"])].append(row)

    for task in sorted(rows_by_task, key=lambda item: TASK_ORDER.index(item) if item in TASK_ORDER else len(TASK_ORDER)):
        summary_lines.append(f"Task: {task}")
        task_rows = rows_by_task[task]

        for spec in METRIC_SPECS:
            best_condition, best_value = choose_best(task_rows, spec["column"], spec["goal"])
            if best_condition is None:
                continue
            summary_lines.append(
                f"- Best {spec['label']}: {best_condition} ({best_value:.4f})"
            )

        completed = ", ".join(
            f"{row['condition']}={row.get('num_folds', 0)} folds"
            for row in sorted(task_rows, key=lambda item: str(item["condition"]))
        )
        summary_lines.append(f"- Completed folds: {completed}")
        summary_lines.append("")

    failures_by_condition: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in condition_status_rows:
        if row.get("status") != "completed":
            failures_by_condition[str(row.get("condition", "unknown"))].append(row)

    if failures_by_condition:
        summary_lines.append("Skipped / failed runs")
        summary_lines.append("---------------------")
        for condition in sorted(failures_by_condition):
            summary_lines.append(f"{condition}:")
            for row in failures_by_condition[condition]:
                summary_lines.append(
                    f"- {row.get('task')} {row.get('fold')} [{row.get('stage')}]: {row.get('status')}"
                )
            summary_lines.append("")

    summary_lines.extend(
        [
            "Scoped-project caveats",
            "----------------------",
            "- cVAE work is intentionally out of scope for this run package.",
            "- dVRK simulator validation is intentionally deferred.",
            "- Results should be written up from the exported CSV/TXT artifacts, not from console logs.",
        ]
    )
    (comparison_dir / "writeup_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    usage_prompt = build_usage_prompt(
        runner_script=args.runner_script,
        conditions=args.conditions,
        retain_checkpoints=args.retain_checkpoints,
        analysis_root=args.analysis_root,
        eval_root=args.eval_root,
    )
    (comparison_dir / "run_instructions_prompt.txt").write_text(usage_prompt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
