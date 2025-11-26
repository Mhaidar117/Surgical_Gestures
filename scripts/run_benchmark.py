#!/usr/bin/env python3
"""
CLI script for running benchmark evaluation.

Usage:
    python scripts/run_benchmark.py \
        --checkpoint checkpoints/baseline/checkpoint_epoch_10.pth \
        --data_root /path/to/Surgical_Gestures/repo \
        --task Knot_Tying \
        --mode val \
        --output_dir outputs/benchmarks/eval \
        --batch_size 8
"""
import argparse
import yaml
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from benchmark.evaluator import BenchmarkEvaluator
from benchmark.visualizations import VisualizationManager
from benchmark.report_generator import ReportGenerator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Run benchmark evaluation on a trained checkpoint',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file (.pth)'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Root directory containing Gestures/ folder'
    )

    # Optional arguments
    parser.add_argument(
        '--task',
        type=str,
        default='Knot_Tying',
        choices=['Knot_Tying', 'Needle_Passing', 'Suturing'],
        help='Task name'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='val',
        choices=['train', 'val', 'test'],
        help='Data split to evaluate on'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Output directory (default: outputs/benchmarks/<timestamp>)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='src/configs/benchmark_config.yaml',
        help='Path to benchmark configuration file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=16,
        help='Batch size for evaluation'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'mps', 'cpu'],
        help='Device to use for evaluation'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of dataloader workers'
    )

    return parser.parse_args()


def load_benchmark_config(config_path: str) -> dict:
    """Load benchmark configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return {
            'evaluation': {'batch_size': 16, 'num_workers': 4, 'device': 'auto'},
            'visualizations': {'enabled': True, 'format': 'png', 'dpi': 300},
            'reports': {'generate_text': True, 'generate_csv': True, 'generate_markdown': True}
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def print_header():
    """Print benchmark header."""
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUITE")
    print("=" * 80)


def print_summary(metrics: dict):
    """Print evaluation summary to console."""
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Kinematics
    print("\nKinematics:")
    kin = metrics['kinematics']
    print(f"  Position RMSE:     {kin['position_rmse']:.6f} m")
    print(f"  EE Error:          {kin['end_effector_error']:.6f} m")
    print(f"  Rotation Error:    {kin.get('rotation_geodesic_deg', 0):.2f}°")
    print(f"  Gripper MAE:       {kin.get('gripper_mae', 0):.6f}")

    # Gesture
    print("\nGesture Classification:")
    gest = metrics['gesture']
    print(f"  Accuracy:          {gest['accuracy']:.2%}")
    print(f"  F1 (Macro):        {gest['f1_macro']:.4f}")
    print(f"  F1 (Weighted):     {gest['f1_weighted']:.4f}")

    # Skill
    print("\nSkill Classification:")
    skill = metrics['skill']
    print(f"  Accuracy:          {skill['accuracy']:.2%}")
    print(f"  F1 (Macro):        {skill['f1_macro']:.4f}")
    print(f"  F1 (Weighted):     {skill['f1_weighted']:.4f}")

    print("\n" + "=" * 80)


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()

    # Print header
    print_header()

    # Set up output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('outputs') / 'benchmarks' / f'eval_{timestamp}'
    else:
        output_dir = Path(args.output_dir)

    print(f"\nConfiguration:")
    print(f"  Checkpoint:        {args.checkpoint}")
    print(f"  Data Root:         {args.data_root}")
    print(f"  Task:              {args.task}")
    print(f"  Mode:              {args.mode}")
    print(f"  Output Directory:  {output_dir}")
    print(f"  Batch Size:        {args.batch_size}")
    print(f"  Device:            {args.device}")
    print(f"  Config:            {args.config}")

    # Load benchmark config
    benchmark_config = load_benchmark_config(args.config)

    # Override batch size and device if specified
    if 'evaluation' not in benchmark_config:
        benchmark_config['evaluation'] = {}
    benchmark_config['evaluation']['batch_size'] = args.batch_size
    benchmark_config['evaluation']['device'] = args.device
    benchmark_config['evaluation']['num_workers'] = args.num_workers

    # Create evaluator
    print("\n" + "-" * 80)
    print("PHASE 1: EVALUATION")
    print("-" * 80)

    evaluator = BenchmarkEvaluator(
        checkpoint_path=args.checkpoint,
        data_root=args.data_root,
        task=args.task,
        mode=args.mode,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Run evaluation
    results = evaluator.run_full_evaluation(output_dir)

    metrics = results['metrics']
    predictions = results['predictions']
    targets = results['targets']

    # Generate visualizations
    if benchmark_config.get('visualizations', {}).get('enabled', True):
        print("\n" + "-" * 80)
        print("PHASE 2: VISUALIZATIONS")
        print("-" * 80)

        viz_manager = VisualizationManager(
            output_dir=output_dir,
            dpi=benchmark_config.get('visualizations', {}).get('dpi', 300),
            format=benchmark_config.get('visualizations', {}).get('format', 'png')
        )

        viz_paths = viz_manager.generate_all_plots(predictions, targets, metrics)
    else:
        viz_paths = {}

    # Generate reports
    if any(benchmark_config.get('reports', {}).values()):
        print("\n" + "-" * 80)
        print("PHASE 3: REPORT GENERATION")
        print("-" * 80)

        report_gen = ReportGenerator(output_dir=output_dir)
        report_gen.generate_all_reports(metrics, predictions, targets, viz_paths)

    # Print summary
    print_summary(metrics)

    # Print output locations
    print("\nOutput saved to:")
    print(f"  Root:              {output_dir}")
    print(f"  Metrics:           {output_dir / 'metrics'}")
    print(f"  Visualizations:    {output_dir / 'visualizations'}")
    print(f"  Reports:           {output_dir / 'reports'}")

    print("\nQuick access:")
    print(f"  Summary:           {output_dir / 'metrics' / 'summary.txt'}")
    print(f"  Markdown Report:   {output_dir / 'reports' / 'benchmark_report.md'}")
    print(f"  Metrics JSON:      {output_dir / 'metrics' / 'metrics.json'}")

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION COMPLETE")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
