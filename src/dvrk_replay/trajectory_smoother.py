import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import PyKDL

class TrajectorySmoother:
    def __init__(self,
            position_window = 11,
            rotation_window = 11,
            gripper_window = 11,
            max_vel = 0.05,
            max_acc = 0.2,
            max_gripper_vel = 0.5):
        self.position_window = position_window if position_window % 2 == 1 else position_window + 1
        self.rotation_window = rotation_window if rotation_window % 2 == 1 else rotation_window + 1
        self.max_vel = max_vel
        self.max_acc = max_acc
        self.max_gripper_vel = max_gripper_vel
        self.gripper_window = gripper_window if gripper_window % 2 == 1 else gripper_window + 1
 
    def smooth_position(self, positions):
        if len(positions) < self.position_window:
            return gaussian_filter1d(positions, sigma=2, axis=0)
        smoothed = np.zeros_like(positions)
        for i in range(3):
            smoothed[:,i] = savgol_filter(positions[:,i], self.position_window, polyorder=3)
        return smoothed

    def smooth_rotation(self, rotations):
        if len(rotations) < self.rotation_window:
            return gaussian_filter1d(rotations, sigma=2, axis=0)

        # Smooth each element of rotation matrix
        smoothed = np.zeros_like(rotations)
        for i in range(9):
            smoothed[:, i] = savgol_filter(rotations[:, i], self.rotation_window, polyorder=3)
        return self.project_to_SO3_batch(smoothed)
    
    def smooth_gripper(self, gripper):
        if len(gripper) < self.gripper_window:
            return gaussian_filter1d(gripper, sigma=2)
        
        return savgol_filter(gripper, self.gripper_window, polyorder=2)

    def smooth_gripper(self, gripper):
        if len(gripper) < self.gripper_window:
            return gaussian_filter1d(gripper, sigma=2)

        return savgol_filter(gripper, self.gripper_window, polyorder=2)

    def project_to_SO3_batch(self, matrices):
        valid_matrices = np.zeros_like(matrices)

        for i in range(len(matrices)):
            # Reshape to 3x3
            M = matrices[i].reshape(3, 3)

            # SVD projection to nearest rotation matrix
            U, _, Vt = np.linalg.svd(M)
            R = U @ Vt

            # Ensure determinant is +1 (proper rotation, not reflection)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

            valid_matrices[i] = R.flatten()

        return valid_matrices

    def _upsample_trajectory(self, trajectory, factor):
        N, D = trajectory.shape
        new_N = (N - 1) * factor + 1
        upsampled = np.zeros((new_N, D))

        for i in range(N - 1):
            start_idx = i * factor
            end_idx = (i + 1) * factor

            for d in range(D):
                upsampled[start_idx:end_idx, d] = np.linspace(
                    trajectory[i, d],
                    trajectory[i + 1, d],
                    factor,
                    endpoint=False
                )

        upsampled[-1] = trajectory[-1]
        return upsampled

    def remove_outliers(self, data, threshold=3.0):
        # Compute z-scores for each dimension
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        z_scores = np.abs((data - mean) / (std + 1e-8))

        # Find outliers
        outliers = np.any(z_scores > threshold, axis=1)

        if np.sum(outliers) > 0:
            print(f"Found {np.sum(outliers)} outlier frames, interpolating...")

            # Interpolate outlier values
            for d in range(data.shape[1]):
                valid_indices = np.where(~outliers)[0]
                outlier_indices = np.where(outliers)[0]

                if len(valid_indices) > 1:
                    data[outlier_indices, d] = np.interp(
                        outlier_indices,
                        valid_indices,
                        data[valid_indices, d]
                    )

        return data


def rotation_6d_to_matrix(rot_6d): 
    # First two columns
    a1 = rot_6d[0:3]
    a2 = rot_6d[3:6]
    
    # Normalize first column
    b1 = a1 / np.linalg.norm(a1)
    
    # Second column: orthogonalize and normalize
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    
    # Third column: cross product
    b3 = np.cross(b1, b2)
    
    # Stack into rotation matrix and flatten
    R = np.stack([b1, b2, b3], axis=1)  # 3x3
    return R.flatten()  # 9D

def main():
    input_file = "/home/mai/Programming/dvrk_gesture_generation/baby_suturing_psm2_final.txt"
    data = np.loadtxt(input_file)
    if data.shape[1] == 10:
        pos = data[:, :3]
        rot_6d = data[:, 3:9]
        gripper = data[:, 9]
        rot = np.array([rotation_6d_to_matrix(rot_6d[i]) for i in range(len(rot_6d))])
    
    elif data.shape[1] == 19:
        # Single PSM format: pos(3) + rot(9) + vel(3) + ang_vel(3) + gripper(1)
        print("Format: Single PSM (19 columns)")
        pos = data[:, 0:3]
        rot = data[:, 3:12]
        gripper = data[:, 18]

    elif data.shape[1] == 76:
        # JIGSAW format - use PSM1
        print("Detected JIGSAW format, using PSM1 data")
        pos = data[:, 38:41]
        rot = data[:, 41:50]
        gripper = data[:, 56]
    else:
        print(f"Error: Unexpected data shape {data.shape}")
        return
    
    # Print original statistics
    print("\n=== Original Trajectory ===")
    print(f"Position range:")
    print(f"  X: [{pos[:,0].min():.4f}, {pos[:,0].max():.4f}] m")
    print(f"  Y: [{pos[:,1].min():.4f}, {pos[:,1].max():.4f}] m")
    print(f"  Z: [{pos[:,2].min():.4f}, {pos[:,2].max():.4f}] m")
    print(f"Gripper range: [{gripper.min():.4f}, {gripper.max():.4f}] rad")
    
    # Compute original jerk
    vel_orig = np.diff(pos, axis=0)
    acc_orig = np.diff(vel_orig, axis=0)
    jerk_orig = np.diff(acc_orig, axis=0)
    jerk_mag_orig = np.mean(np.linalg.norm(jerk_orig, axis=1))
    print(f"Original jerk: {jerk_mag_orig:.6f}")
    
    # Create smoother
    print("\n=== Smoothing Trajectory ===")
    smoother = TrajectorySmoother(
        position_window=71,  # Larger window for more smoothing
        rotation_window=71,
        gripper_window=31,
        max_vel=0.03,
        max_acc=0.15,
        max_gripper_vel=0.5
    )
    
    # Remove outliers
    print("Removing outliers...")
    pos_clean = smoother.remove_outliers(pos, threshold=3.0)
    rot_clean = smoother.remove_outliers(rot, threshold=3.0)
    gripper_clean = smoother.remove_outliers(gripper.reshape(-1, 1), threshold=3.0).flatten()
    
    # Smooth trajectories
    print("Smoothing position...")
    pos_smooth = smoother.smooth_position(pos_clean)
    
    print("Smoothing rotation...")
    rot_smooth = smoother.smooth_rotation(rot_clean)
    
    print("Smoothing gripper...")
    gripper_smooth = smoother.smooth_gripper(gripper_clean)
    
    # Compute smoothed jerk
    vel_smooth = np.diff(pos_smooth, axis=0)
    acc_smooth = np.diff(vel_smooth, axis=0)
    jerk_smooth = np.diff(acc_smooth, axis=0)
    jerk_mag_smooth = np.mean(np.linalg.norm(jerk_smooth, axis=1))
    
    # Print results
    print("\n=== Smoothed Trajectory ===")
    print(f"Position range:")
    print(f"  X: [{pos_smooth[:,0].min():.4f}, {pos_smooth[:,0].max():.4f}] m")
    print(f"  Y: [{pos_smooth[:,1].min():.4f}, {pos_smooth[:,1].max():.4f}] m")
    print(f"  Z: [{pos_smooth[:,2].min():.4f}, {pos_smooth[:,2].max():.4f}] m")
    print(f"Gripper range: [{gripper_smooth.min():.4f}, {gripper_smooth.max():.4f}] rad")
    print(f"Smoothed jerk: {jerk_mag_smooth:.6f}")
    
    # Improvement
    improvement = (1 - jerk_mag_smooth / jerk_mag_orig) * 100
    print(f"\n✓ Jerk reduced by {improvement:.1f}%")
    
    # Mean deviation
    if len(pos) == len(pos_smooth):
        deviation = np.mean(np.linalg.norm(pos - pos_smooth, axis=1))
        print(f"✓ Mean position deviation: {deviation*1000:.2f} mm")
    
    # Save smoothed trajectory
    if data.shape[1] == 10:
        # Convert 9D back to 6D for saving
        rot_6d_smooth = rot_smooth[:, :6]  # Just take first 6 columns as approximation
        smoothed_data = np.hstack([pos_smooth, rot_6d_smooth, gripper_smooth.reshape(-1, 1)])
    elif data.shape[1] == 19:
        # Keep 19-column format, replace position, rotation, gripper
        smoothed_data = data.copy()
        smoothed_data[:, 0:3] = pos_smooth
        smoothed_data[:, 3:12] = rot_smooth
        smoothed_data[:, 18] = gripper_smooth
    elif data.shape[1] == 76:
        # Full JIGSAW format, replace PSM1 data
        smoothed_data = data.copy()
        smoothed_data[:, 38:41] = pos_smooth
        smoothed_data[:, 41:50] = rot_smooth
        smoothed_data[:, 56] = gripper_smooth
    else:
        print(f"Error: Cannot save format with {data.shape[1]} columns")
        return

    output_file = input_file.replace('.txt', '_smoothed_cmd.txt')
    np.savetxt(output_file, smoothed_data)
    print(f"\n✓ Saved smoothed trajectory to: {output_file}")
    
    # Optional: Plot comparison
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        
        # Position
        axes[0].plot(pos[:, 0], 'b-', alpha=0.5, label='Original X')
        axes[0].plot(pos_smooth[:, 0], 'r-', label='Smoothed X')
        axes[0].set_ylabel('X Position (m)')
        axes[0].legend()
        axes[0].grid(True)
        axes[0].set_title('Position Comparison')
        
        axes[1].plot(pos[:, 1], 'b-', alpha=0.5, label='Original Y')
        axes[1].plot(pos_smooth[:, 1], 'r-', label='Smoothed Y')
        axes[1].set_ylabel('Y Position (m)')
        axes[1].legend()
        axes[1].grid(True)
        
        axes[2].plot(pos[:, 2], 'b-', alpha=0.5, label='Original Z')
        axes[2].plot(pos_smooth[:, 2], 'r-', label='Smoothed Z')
        axes[2].set_ylabel('Z Position (m)')
        axes[2].set_xlabel('Frame')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file.replace('.txt', '_comparison.png'))
        print(f"✓ Saved comparison plot to: {output_file.replace('.txt', '_comparison.png')}")
        plt.show()
        
    except ImportError:
        print("(matplotlib not available, skipping plot)")


if __name__ == "__main__":
    main()
