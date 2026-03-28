#!/usr/bin/env python3
"""
Compare ground truth (76-col) vs generated (19-col) trajectories
"""

import numpy as np
import matplotlib.pyplot as plt

def load_ground_truth(filepath):
    """Load PSM2 from 76-column JIGSAW file"""
    data = np.loadtxt(filepath)
    pos = data[:, 57:60]      # PSM2 position
    rot = data[:, 60:69]      # PSM2 rotation
    gripper = data[:, 75]     # PSM2 gripper
    return pos, rot, gripper

def load_generated(filepath):
    """Load from 19-column generated file"""
    data = np.loadtxt(filepath)
    pos = data[:, 0:3]
    rot = data[:, 3:12]
    gripper = data[:, 18]
    return pos, rot, gripper

def compute_stats(pos_gt, pos_gen):
    """Compute trajectory statistics"""
    # Align lengths
    min_len = min(len(pos_gt), len(pos_gen))
    pos_gt = pos_gt[:min_len]
    pos_gen = pos_gen[:min_len]
    
    # Position error
    pos_error = np.linalg.norm(pos_gt - pos_gen, axis=1)
    
    # Jerk (smoothness)
    def compute_jerk(pos):
        vel = np.diff(pos, axis=0)
        acc = np.diff(vel, axis=0)
        jerk = np.diff(acc, axis=0)
        return np.mean(np.linalg.norm(jerk, axis=1))
    
    jerk_gt = compute_jerk(pos_gt)
    jerk_gen = compute_jerk(pos_gen)
    
    return {
        'mean_error_mm': pos_error.mean() * 1000,
        'max_error_mm': pos_error.max() * 1000,
        'jerk_gt': jerk_gt,
        'jerk_gen': jerk_gen,
        'jerk_ratio': jerk_gen / jerk_gt
    }

def plot_comparison(pos_gt, pos_gen, gripper_gt, gripper_gen, stats, skill, output_file='comparison'):
    """Create comparison plots"""
    min_len = min(len(pos_gt), len(pos_gen))
    pos_gt = pos_gt[:min_len]
    pos_gen = pos_gen[:min_len]
    gripper_gt = gripper_gt[:min_len]
    gripper_gen = gripper_gen[:min_len]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle(f'Trajectory Comparison: {skill}', fontsize=16, fontweight='bold')
    
    # Position XYZ comparison
    for i, label in enumerate(['X', 'Y', 'Z']):
        axes[i, 0].plot(pos_gt[:, i], 'b-', alpha=0.7, linewidth=2.5, label='Ground Truth')
        axes[i, 0].plot(pos_gen[:, i], 'r-', alpha=0.7, linewidth=1.5, label='Generated')
        axes[i, 0].set_ylabel(f'{label} Position (m)', fontsize=11)
        axes[i, 0].legend(fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].set_title('Position Trajectories', fontweight='bold', fontsize=12)
        if i == 2:
            axes[i, 0].set_xlabel('Frame', fontsize=11)
    
    # Position error
    error = np.linalg.norm(pos_gt - pos_gen, axis=1) * 1000
    axes[0, 1].plot(error, 'g-', linewidth=1.5)
    axes[0, 1].axhline(stats['mean_error_mm'], color='r', linestyle='--', linewidth=2,
                       label=f"Mean: {stats['mean_error_mm']:.2f} mm")
    axes[0, 1].set_ylabel('Error (mm)', fontsize=11)
    axes[0, 1].set_title('Trajectory Error', fontweight='bold', fontsize=12)
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Velocity comparison
    vel_gt = np.linalg.norm(np.diff(pos_gt, axis=0), axis=1) * 30
    vel_gen = np.linalg.norm(np.diff(pos_gen, axis=0), axis=1) * 30
    axes[1, 1].plot(vel_gt, 'b-', alpha=0.7, linewidth=2.5, label='Ground Truth')
    axes[1, 1].plot(vel_gen, 'r-', alpha=0.7, linewidth=1.5, label='Generated')
    axes[1, 1].set_ylabel('Velocity (m/s)', fontsize=11)
    axes[1, 1].set_title('Velocity Profile', fontweight='bold', fontsize=12)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim([0, max(vel_gt.max(), vel_gen.max()) * 1.1])
    
    # Gripper comparison
    axes[2, 1].plot(gripper_gt, 'b-', alpha=0.7, linewidth=2.5, label='Ground Truth')
    axes[2, 1].plot(gripper_gen, 'r-', alpha=0.7, linewidth=1.5, label='Generated')
    axes[2, 1].set_xlabel('Frame', fontsize=11)
    axes[2, 1].set_ylabel('Gripper Angle (rad)', fontsize=11)
    axes[2, 1].set_title('Gripper Trajectory', fontweight='bold', fontsize=12)
    axes[2, 1].legend(fontsize=9)
    axes[2, 1].grid(True, alpha=0.3)
    gripper_min = min(gripper_gt.min(), gripper_gen.min())
    gripper_max = max(gripper_gt.max(), gripper_gen.max())
    axes[2, 1].set_ylim([gripper_min * 1.1 if gripper_min < 0 else gripper_min * 0.9, 
                         gripper_max * 1.1])
    
    # Align y-axis labels in each column
    fig.align_ylabels(axes[:, 0])  # Align left column
    fig.align_ylabels(axes[:, 1])  # Align right column
    
    plt.tight_layout()
    plt.savefig(output_file + "_" + skill + ".png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {output_file}_{skill}.png")
    plt.show()
    
def main():
    # File paths
    gt_file = "/home/mai/JISSAW/Knot_Tying/kinematics/AllGestures/Knot_Tying_C002.txt"
    gen_file = "/home/mai/Programming/dvrk_gesture_generation/baby_knot_tying_psm2_final_smoothed_cmd.txt"
    
    print("Loading trajectories...")
    pos_gt, rot_gt, gripper_gt = load_ground_truth(gt_file)
    pos_gen, rot_gen, gripper_gen = load_generated(gen_file)
    
    print(f"Ground truth: {len(pos_gt)} frames")
    print(f"Generated: {len(pos_gen)} frames")
    
    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_stats(pos_gt, pos_gen)
    
    print("\n=== Trajectory Comparison ===")
    print(f"Mean position error: {stats['mean_error_mm']:.2f} mm")
    print(f"Max position error: {stats['max_error_mm']:.2f} mm")
    print(f"Ground truth jerk: {stats['jerk_gt']:.6f}")
    print(f"Generated jerk: {stats['jerk_gen']:.6f}")
    print(f"Jerk ratio (gen/gt): {stats['jerk_ratio']:.2f}x")
    
    if stats['jerk_ratio'] < 1:
        print("✓ Generated trajectory is SMOOTHER than ground truth")
    else:
        print("✗ Generated trajectory is MORE JITTERY than ground truth")
    
    # Create plots
    print("\nGenerating plots...")
    plot_comparison(pos_gt, pos_gen, gripper_gt, gripper_gen, stats, skill = "Knot Tying")

if __name__ == "__main__":
    main()
