#!/usr/bin/env python3
"""
Aggregate 8-fold LOUO cross-validation results.

Computes mean ± std for all metrics across folds, which is the standard
way to report LOUO results in surgical gesture recognition papers.

Usage:
    python3 aggregate_louo_results.py
    python3 aggregate_louo_results.py --eval_dir eval_results --output louo_summary.txt
"""
import os
import re
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np


def parse_results_file(filepath):
    """Parse a single results .txt file and extract metrics."""
    metrics = {}

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse different sections
    lines = content.split('\n')
    current_section = None

    for line in lines:
        line = line.strip()

        # Detect sections
        if 'Loss Components' in line:
            current_section = 'loss'
        elif 'Kinematics Metrics' in line:
            current_section = 'kinematics'
        elif 'Gesture Metrics' in line:
            current_section = 'gesture'
        elif 'Skill Metrics' in line:
            current_section = 'skill'
        elif line.startswith('---') or line.startswith('==='):
            continue
        elif ':' in line and current_section:
            # Parse key: value pairs
            parts = line.split(':')
            if len(parts) >= 2:
                key = parts[0].strip()
                value_str = parts[1].strip()

                # Try to extract numeric value
                try:
                    # Handle percentages
                    if '%' in value_str:
                        value = float(value_str.replace('%', ''))
                    else:
                        # Extract first number from string
                        numbers = re.findall(r'[-+]?\d*\.?\d+', value_str)
                        if numbers:
                            value = float(numbers[0])
                        else:
                            continue

                    # Create unique metric name
                    metric_name = f"{current_section}_{key}".lower().replace(' ', '_')
                    metrics[metric_name] = value
                except (ValueError, IndexError):
                    continue

    return metrics


def aggregate_results(eval_dir, tasks=None):
    """Aggregate results across all folds for each task."""
    eval_dir = Path(eval_dir)

    if tasks is None:
        tasks = ['Knot_Tying', 'Needle_Passing', 'Suturing']

    all_results = {}

    for task in tasks:
        task_results = defaultdict(list)
        fold_count = 0

        # Find all result files for this task
        pattern = f"{task}_test_fold_*.txt"
        result_files = sorted(eval_dir.glob(pattern))

        if not result_files:
            # Try alternative naming pattern
            pattern = f"{task.lower()}_test_fold_*.txt"
            result_files = sorted(eval_dir.glob(pattern))

        for result_file in result_files:
            metrics = parse_results_file(result_file)
            fold_count += 1

            for metric_name, value in metrics.items():
                task_results[metric_name].append(value)

        if fold_count > 0:
            # Compute mean and std for each metric
            aggregated = {
                'num_folds': fold_count,
                'metrics': {}
            }

            for metric_name, values in task_results.items():
                aggregated['metrics'][metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }

            all_results[task] = aggregated

    return all_results


def print_summary(results, output_file=None):
    """Print a formatted summary of aggregated results."""
    lines = []

    lines.append("=" * 70)
    lines.append("8-FOLD LOUO CROSS-VALIDATION RESULTS")
    lines.append("=" * 70)
    lines.append("")

    for task, data in results.items():
        lines.append(f"Task: {task} ({data['num_folds']} folds)")
        lines.append("-" * 70)

        metrics = data['metrics']

        # Group metrics by category
        categories = ['loss', 'kinematics', 'gesture', 'skill']

        for category in categories:
            category_metrics = {k: v for k, v in metrics.items() if k.startswith(category)}

            if category_metrics:
                lines.append(f"\n  {category.upper()} METRICS:")

                for metric_name, stats in sorted(category_metrics.items()):
                    short_name = metric_name.replace(f"{category}_", "")
                    mean = stats['mean']
                    std = stats['std']

                    # Format based on magnitude
                    if abs(mean) < 0.01:
                        lines.append(f"    {short_name:30s}: {mean:.6f} ± {std:.6f}")
                    elif abs(mean) < 1:
                        lines.append(f"    {short_name:30s}: {mean:.4f} ± {std:.4f}")
                    else:
                        lines.append(f"    {short_name:30s}: {mean:.2f} ± {std:.2f}")

        lines.append("")
        lines.append("=" * 70)
        lines.append("")

    # Print key metrics summary table
    lines.append("SUMMARY TABLE (Key Metrics)")
    lines.append("-" * 70)
    lines.append(f"{'Task':<20} {'Gesture Acc':<18} {'Skill Acc':<18} {'Jaw Loss':<18}")
    lines.append("-" * 70)

    for task, data in results.items():
        metrics = data['metrics']

        # Get key metrics (adjust names based on what's available)
        gesture_acc = metrics.get('gesture_accuracy', metrics.get('gesture_acc', {}))
        skill_acc = metrics.get('skill_accuracy', metrics.get('skill_acc', {}))
        jaw_loss = metrics.get('loss_kin_jaw', metrics.get('kinematics_jaw', {}))

        g_str = f"{gesture_acc.get('mean', 0):.2f} ± {gesture_acc.get('std', 0):.2f}" if gesture_acc else "N/A"
        s_str = f"{skill_acc.get('mean', 0):.2f} ± {skill_acc.get('std', 0):.2f}" if skill_acc else "N/A"
        j_str = f"{jaw_loss.get('mean', 0):.4f} ± {jaw_loss.get('std', 0):.4f}" if jaw_loss else "N/A"

        lines.append(f"{task:<20} {g_str:<18} {s_str:<18} {j_str:<18}")

    lines.append("-" * 70)
    lines.append("")
    lines.append("Note: Results reported as mean ± std across folds")

    output = '\n'.join(lines)
    print(output)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(output)
        print(f"\nResults saved to: {output_file}")

    return output


def save_json_results(results, output_file):
    """Save detailed results as JSON for further analysis."""
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(output_file, 'w') as f:
        json.dump(convert(results), f, indent=2)

    print(f"Detailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Aggregate LOUO cross-validation results')
    parser.add_argument('--eval_dir', type=str, default='eval_results',
                       help='Directory containing evaluation result files')
    parser.add_argument('--output', type=str, default='louo_summary.txt',
                       help='Output file for summary')
    parser.add_argument('--json_output', type=str, default='louo_results.json',
                       help='Output file for detailed JSON results')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                       help='Tasks to aggregate (default: all)')

    args = parser.parse_args()

    # Aggregate results
    results = aggregate_results(args.eval_dir, args.tasks)

    if not results:
        print(f"No results found in {args.eval_dir}/")
        print("Make sure evaluation files exist with pattern: <Task>_test_fold_*.txt")
        return

    # Print and save summary
    print_summary(results, args.output)

    # Save detailed JSON
    save_json_results(results, args.json_output)


if __name__ == '__main__':
    main()
