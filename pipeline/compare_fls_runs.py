#!/usr/bin/env python3
"""Side-by-side comparison of two FLS GW runs.

Loads two `results_fls.json` files and produces a markdown table of the
headline, companion, and per-task metrics so an ablation (e.g. with vs.
without per-subject EEG z-scoring) can be shown in one place.

Usage
-----
    python pipeline/compare_fls_runs.py \
        --left  reports/skill_manifold_fls/results_fls.json \
        --right reports/skill_manifold_fls_no_zscore/results_fls.json \
        --left_label "z-scored (default)" \
        --right_label "raw (--skip_eeg_zscore)" \
        --output reports/skill_manifold_fls/ablation_comparison.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


def _load(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v, kind: str = "f") -> str:
    if v is None:
        return "—"
    if kind == "p":
        return f"{v:.4f}"
    if kind == "z":
        return f"{v:+.3f}"
    if kind == "gw":
        return f"{v:.4f}"
    if kind == "tr":
        return f"{v:.4f}"
    if kind == "rate":
        return f"{v:.3f}"
    return str(v)


def _row(label: str, l_val: str, r_val: str) -> str:
    return f"| {label} | {l_val} | {r_val} |"


def _headline_rows(left: Dict, right: Dict) -> List[str]:
    L = left.get("headline", {}) or {}
    R = right.get("headline", {}) or {}
    return [
        _row("Headline GW distance", _fmt(L.get("gw_distance"), "gw"),
                                     _fmt(R.get("gw_distance"), "gw")),
        _row("Headline z", _fmt(L.get("z_score"), "z"),
                            _fmt(R.get("z_score"), "z")),
        _row("Headline p", _fmt(L.get("p_value"), "p"),
                            _fmt(R.get("p_value"), "p")),
        _row("Headline argmax",
             "`" + str(L.get("argmax_assignment", {})) + "`",
             "`" + str(R.get("argmax_assignment", {})) + "`"),
    ]


def _companion_rows(left: Dict, right: Dict) -> List[str]:
    L = left.get("companion") or {}
    R = right.get("companion") or {}
    return [
        _row("Companion N", str(L.get("n_subjects", "—")),
                             str(R.get("n_subjects", "—"))),
        _row("Companion trace(T)", _fmt(L.get("diag_mass_observed"), "tr"),
                                    _fmt(R.get("diag_mass_observed"), "tr")),
        _row("Companion z", _fmt(L.get("diag_mass_z_score"), "z"),
                             _fmt(R.get("diag_mass_z_score"), "z")),
        _row("Companion p", _fmt(L.get("diag_mass_p_value"), "p"),
                             _fmt(R.get("diag_mass_p_value"), "p")),
        _row("Companion argmax match-rate", _fmt(L.get("argmax_match_rate"), "rate"),
                                              _fmt(R.get("argmax_match_rate"), "rate")),
    ]


def _per_task_rows(left: Dict, right: Dict) -> List[str]:
    L = left.get("per_task") or {}
    R = right.get("per_task") or {}
    keys = sorted(set(int(k) for k in (*L.keys(), *R.keys())))
    rows: List[str] = []
    for tid in keys:
        ln = L.get(str(tid)) or L.get(tid) or {}
        rn = R.get(str(tid)) or R.get(tid) or {}
        tname = ln.get("task_name") or rn.get("task_name") or f"task_{tid}"
        rows.append(_row(f"{tname} GW",
                         _fmt(ln.get("gw_distance"), "gw"),
                         _fmt(rn.get("gw_distance"), "gw")))
        rows.append(_row(f"{tname} z",
                         _fmt(ln.get("z_score"), "z"),
                         _fmt(rn.get("z_score"), "z")))
        rows.append(_row(f"{tname} p",
                         _fmt(ln.get("p_value"), "p"),
                         _fmt(rn.get("p_value"), "p")))
        rows.append(_row(f"{tname} argmax",
                         "`" + str(ln.get("argmax_assignment", {})) + "`",
                         "`" + str(rn.get("argmax_assignment", {})) + "`"))
    return rows


def _config_rows(left: Dict, right: Dict) -> List[str]:
    Lc = left.get("config") or {}
    Rc = right.get("config") or {}
    return [
        _row("N (after degeneracy filter)",
             str(left.get("n_subjects", "—")),
             str(right.get("n_subjects", "—"))),
        _row("Tier counts",
             str(left.get("tier_counts", {})),
             str(right.get("tier_counts", {}))),
        _row("Dropped subjects",
             str(left.get("dropped_subjects") or "—"),
             str(right.get("dropped_subjects") or "—")),
        _row("residualize_task",
             "`" + str(Lc.get("residualize_task", "?")) + "`",
             "`" + str(Rc.get("residualize_task", "?")) + "`"),
        _row("eeg_zscore",
             "`" + str(Lc.get("eeg_zscore", "?")) + "`",
             "`" + str(Rc.get("eeg_zscore", "?")) + "`"),
    ]


def render_markdown(
    left: Dict, right: Dict,
    left_label: str, right_label: str,
) -> str:
    header = f"# FLS GW: {left_label} vs {right_label}\n"
    table_header = (
        f"| Metric | {left_label} | {right_label} |\n"
        f"|---|---|---|\n"
    )

    config = "## Configuration\n\n" + table_header + "\n".join(_config_rows(left, right)) + "\n"
    headline = "\n## Headline\n\n" + table_header + "\n".join(_headline_rows(left, right)) + "\n"
    companion = "\n## Companion\n\n" + table_header + "\n".join(_companion_rows(left, right)) + "\n"
    per_task_rows = _per_task_rows(left, right)
    per_task = (
        "\n## Per-task\n\n" + table_header + "\n".join(per_task_rows) + "\n"
        if per_task_rows else ""
    )

    return "\n".join([header, config, headline, companion, per_task])


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True,
                   help="First results_fls.json (e.g. the default run).")
    p.add_argument("--right", type=Path, required=True,
                   help="Second results_fls.json (e.g. the ablation run).")
    p.add_argument("--left_label", type=str, default="left",
                   help="Column label for the first run.")
    p.add_argument("--right_label", type=str, default="right",
                   help="Column label for the second run.")
    p.add_argument("--output", type=Path, default=None,
                   help="Where to write the markdown. Default = print to stdout.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    left = _load(args.left)
    right = _load(args.right)
    md = render_markdown(left, right, args.left_label, args.right_label)
    if args.output is None:
        print(md)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(md)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
