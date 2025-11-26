"""
Report generation for benchmark evaluation results.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


class ReportGenerator:
    """Generator for benchmark evaluation reports."""

    def __init__(self, output_dir: Path):
        """
        Initialize report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = Path(output_dir)
        self.reports_dir = self.output_dir / 'reports'
        self.metrics_dir = self.output_dir / 'metrics'

        # Create directories
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_reports(
        self,
        metrics: Dict,
        predictions: Dict,
        targets: Dict,
        viz_paths: Dict[str, Path]
    ) -> None:
        """
        Generate all report formats.

        Args:
            metrics: Computed metrics dictionary
            predictions: Model predictions
            targets: Ground truth targets
            viz_paths: Paths to visualization files
        """
        print("\nGenerating reports...")

        # Text summary
        print("  - Text summary...")
        self.generate_text_report(metrics)

        # CSV exports
        print("  - CSV exports...")
        self.generate_csv_reports(metrics, predictions, targets)

        # Markdown report
        print("  - Markdown report...")
        self.generate_markdown_report(metrics, viz_paths)

        # JSON metrics
        print("  - JSON metrics...")
        self.save_metrics_json(metrics)

        print("Reports generated successfully")

    def generate_text_report(self, metrics: Dict) -> Path:
        """
        Generate text summary report.

        Args:
            metrics: Metrics dictionary

        Returns:
            Path to text report
        """
        output_path = self.metrics_dir / 'summary.txt'

        with open(output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BENCHMARK EVALUATION SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Kinematics metrics
            f.write("KINEMATICS METRICS\n")
            f.write("-" * 80 + "\n")
            kin = metrics['kinematics']
            f.write(f"  Position RMSE:           {kin['position_rmse']:.6f} m\n")
            f.write(f"    - X-axis RMSE:         {kin['position_rmse_x']:.6f} m\n")
            f.write(f"    - Y-axis RMSE:         {kin['position_rmse_y']:.6f} m\n")
            f.write(f"    - Z-axis RMSE:         {kin['position_rmse_z']:.6f} m\n")
            f.write(f"  End-Effector Error:      {kin['end_effector_error']:.6f} m (±{kin.get('end_effector_error_std', 0):.6f})\n")
            f.write(f"  Rotation RMSE (6D):      {kin.get('rotation_rmse', 0):.6f}\n")
            f.write(f"  Rotation Geodesic:       {kin.get('rotation_geodesic_deg', 0):.2f} degrees\n")
            f.write(f"  Gripper MAE:             {kin.get('gripper_mae', 0):.6f}\n")
            f.write(f"  Gripper RMSE:            {kin.get('gripper_rmse', 0):.6f}\n")
            f.write("\n")

            # Gesture metrics
            f.write("GESTURE CLASSIFICATION METRICS\n")
            f.write("-" * 80 + "\n")
            gest = metrics['gesture']
            f.write(f"  Accuracy:                {gest['accuracy']:.4f} ({gest['accuracy']*100:.2f}%)\n")
            f.write(f"  F1-Score (Macro):        {gest['f1_macro']:.4f}\n")
            f.write(f"  F1-Score (Micro):        {gest['f1_micro']:.4f}\n")
            f.write(f"  F1-Score (Weighted):     {gest['f1_weighted']:.4f}\n")
            f.write("\n")
            f.write("  Per-Class F1 Scores:\n")
            for cls, score in sorted(gest['f1_per_class'].items()):
                f.write(f"    {cls:6s}: {score:.4f}\n")
            f.write("\n")

            # Skill metrics
            f.write("SKILL CLASSIFICATION METRICS\n")
            f.write("-" * 80 + "\n")
            skill = metrics['skill']
            f.write(f"  Accuracy:                {skill['accuracy']:.4f} ({skill['accuracy']*100:.2f}%)\n")
            f.write(f"  F1-Score (Macro):        {skill['f1_macro']:.4f}\n")
            f.write(f"  F1-Score (Weighted):     {skill['f1_weighted']:.4f}\n")
            f.write("\n")
            f.write("  Per-Class F1 Scores:\n")
            for cls, score in sorted(skill['f1_per_class'].items()):
                f.write(f"    {cls:15s}: {score:.4f}\n")
            f.write("\n")

            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        return output_path

    def generate_csv_reports(
        self,
        metrics: Dict,
        predictions: Dict,
        targets: Dict
    ) -> Dict[str, Path]:
        """
        Generate CSV exports of metrics.

        Args:
            metrics: Metrics dictionary
            predictions: Predictions dictionary
            targets: Targets dictionary

        Returns:
            Dictionary mapping report names to paths
        """
        csv_paths = {}

        # Kinematics metrics CSV
        kin_data = {
            'Metric': [],
            'Value': []
        }
        for key, value in metrics['kinematics'].items():
            if not isinstance(value, (dict, list, np.ndarray)):
                kin_data['Metric'].append(key)
                kin_data['Value'].append(value)

        df_kin = pd.DataFrame(kin_data)
        kin_path = self.metrics_dir / 'kinematics_metrics.csv'
        df_kin.to_csv(kin_path, index=False)
        csv_paths['kinematics'] = kin_path

        # Gesture per-class metrics CSV
        gesture_data = {
            'Class': [],
            'Precision': [],
            'Recall': [],
            'F1-Score': []
        }
        for cls in metrics['gesture']['f1_per_class'].keys():
            gesture_data['Class'].append(cls)
            gesture_data['Precision'].append(metrics['gesture']['precision_per_class'][cls])
            gesture_data['Recall'].append(metrics['gesture']['recall_per_class'][cls])
            gesture_data['F1-Score'].append(metrics['gesture']['f1_per_class'][cls])

        df_gesture = pd.DataFrame(gesture_data)
        gesture_path = self.metrics_dir / 'gesture_per_class.csv'
        df_gesture.to_csv(gesture_path, index=False)
        csv_paths['gesture'] = gesture_path

        # Skill confusion matrix CSV
        skill_cm = metrics['skill']['confusion_matrix']
        skill_classes = ['Novice', 'Intermediate', 'Expert']
        df_skill_cm = pd.DataFrame(skill_cm, index=skill_classes, columns=skill_classes)
        skill_path = self.metrics_dir / 'skill_confusion.csv'
        df_skill_cm.to_csv(skill_path)
        csv_paths['skill'] = skill_path

        return csv_paths

    def generate_markdown_report(
        self,
        metrics: Dict,
        viz_paths: Dict[str, Path]
    ) -> Path:
        """
        Generate markdown report with embedded visualizations.

        Args:
            metrics: Metrics dictionary
            viz_paths: Paths to visualization files

        Returns:
            Path to markdown report
        """
        output_path = self.reports_dir / 'benchmark_report.md'

        with open(output_path, 'w') as f:
            # Header
            f.write("# Benchmark Evaluation Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write("| Task | Key Metric | Value |\n")
            f.write("|------|------------|-------|\n")
            f.write(f"| Kinematics | Position RMSE | {metrics['kinematics']['position_rmse']:.6f} m |\n")
            f.write(f"| Kinematics | Rotation Error | {metrics['kinematics'].get('rotation_geodesic_deg', 0):.2f}° |\n")
            f.write(f"| Gesture | Accuracy | {metrics['gesture']['accuracy']:.2%} |\n")
            f.write(f"| Gesture | F1-Score (Macro) | {metrics['gesture']['f1_macro']:.4f} |\n")
            f.write(f"| Skill | Accuracy | {metrics['skill']['accuracy']:.2%} |\n")
            f.write(f"| Skill | F1-Score (Macro) | {metrics['skill']['f1_macro']:.4f} |\n")
            f.write("\n---\n\n")

            # Kinematics Section
            f.write("## Kinematics Evaluation\n\n")
            f.write("### Summary Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            kin = metrics['kinematics']
            f.write(f"| Position RMSE | {kin['position_rmse']:.6f} m |\n")
            f.write(f"| Position RMSE (X) | {kin['position_rmse_x']:.6f} m |\n")
            f.write(f"| Position RMSE (Y) | {kin['position_rmse_y']:.6f} m |\n")
            f.write(f"| Position RMSE (Z) | {kin['position_rmse_z']:.6f} m |\n")
            f.write(f"| End-Effector Error | {kin['end_effector_error']:.6f} m |\n")
            f.write(f"| Rotation Geodesic Distance | {kin.get('rotation_geodesic_deg', 0):.2f}° |\n")
            f.write(f"| Gripper MAE | {kin.get('gripper_mae', 0):.6f} |\n")
            f.write("\n")

            # Add trajectory visualizations
            f.write("### Trajectory Visualizations\n\n")
            for key, path in viz_paths.items():
                if 'trajectory' in key:
                    rel_path = path.relative_to(self.output_dir)
                    f.write(f"![{key}](../{rel_path})\n\n")

            # Add error distribution
            if 'position_error_dist' in viz_paths:
                rel_path = viz_paths['position_error_dist'].relative_to(self.output_dir)
                f.write("### Error Distribution\n\n")
                f.write(f"![Position Error Distribution](../{rel_path})\n\n")

            # Add time series
            if 'kinematics_time_series' in viz_paths:
                rel_path = viz_paths['kinematics_time_series'].relative_to(self.output_dir)
                f.write("### Time Series Analysis\n\n")
                f.write(f"![Kinematics Time Series](../{rel_path})\n\n")

            f.write("---\n\n")

            # Gesture Classification Section
            f.write("## Gesture Classification\n\n")
            f.write("### Summary Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            gest = metrics['gesture']
            f.write(f"| Accuracy | {gest['accuracy']:.2%} |\n")
            f.write(f"| F1-Score (Macro) | {gest['f1_macro']:.4f} |\n")
            f.write(f"| F1-Score (Micro) | {gest['f1_micro']:.4f} |\n")
            f.write(f"| F1-Score (Weighted) | {gest['f1_weighted']:.4f} |\n")
            f.write("\n")

            # Confusion matrix
            if 'gesture_confusion' in viz_paths:
                rel_path = viz_paths['gesture_confusion'].relative_to(self.output_dir)
                f.write("### Confusion Matrix\n\n")
                f.write(f"![Gesture Confusion Matrix](../{rel_path})\n\n")

            # Per-class metrics
            if 'per_class_gesture' in viz_paths:
                rel_path = viz_paths['per_class_gesture'].relative_to(self.output_dir)
                f.write("### Per-Class Performance\n\n")
                f.write(f"![Per-Class Gesture Metrics](../{rel_path})\n\n")

            f.write("---\n\n")

            # Skill Classification Section
            f.write("## Skill Classification\n\n")
            f.write("### Summary Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            skill = metrics['skill']
            f.write(f"| Accuracy | {skill['accuracy']:.2%} |\n")
            f.write(f"| F1-Score (Macro) | {skill['f1_macro']:.4f} |\n")
            f.write(f"| F1-Score (Weighted) | {skill['f1_weighted']:.4f} |\n")
            f.write("\n")

            # Confusion matrix
            if 'skill_confusion' in viz_paths:
                rel_path = viz_paths['skill_confusion'].relative_to(self.output_dir)
                f.write("### Confusion Matrix\n\n")
                f.write(f"![Skill Confusion Matrix](../{rel_path})\n\n")

            # Per-class breakdown
            f.write("### Per-Class F1 Scores\n\n")
            f.write("| Skill Level | F1-Score |\n")
            f.write("|-------------|----------|\n")
            for cls, score in skill['f1_per_class'].items():
                f.write(f"| {cls} | {score:.4f} |\n")
            f.write("\n")

            f.write("---\n\n")

            # Footer
            f.write("## Additional Information\n\n")
            f.write(f"- Full metrics available in: `metrics/metrics.json`\n")
            f.write(f"- CSV exports available in: `metrics/`\n")
            f.write(f"- Visualizations available in: `visualizations/`\n")

        return output_path

    def save_metrics_json(self, metrics: Dict) -> Path:
        """
        Save metrics as JSON (already handled by evaluator, but can be called separately).

        Args:
            metrics: Metrics dictionary

        Returns:
            Path to JSON file
        """
        # Convert numpy arrays to lists for JSON serialization
        metrics_serializable = {}
        for task, task_metrics in metrics.items():
            metrics_serializable[task] = {}
            for key, value in task_metrics.items():
                if isinstance(value, dict):
                    # Nested dict (per-class metrics)
                    metrics_serializable[task][key] = value
                elif hasattr(value, 'tolist'):
                    # Numpy array
                    metrics_serializable[task][key] = value.tolist()
                else:
                    # Scalar
                    metrics_serializable[task][key] = value

        output_path = self.metrics_dir / 'metrics.json'
        with open(output_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        return output_path
