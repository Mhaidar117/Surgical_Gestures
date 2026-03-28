"""
Replay JIGSAW needle passing trajectories on single dVRK PSM
"""

import rclpy
import numpy as np
import PyKDL
import time

# Import dVRK and CRTK Python clients
import dvrk
import crtk

class JIGSAWReplay:
    def __init__(self, arm_name='PSM1'):
        """
        Initialize replay for single PSM
        
        Args:
            arm_name: Name of the arm ('PSM1' or 'PSM2')
        """
        # Initialize ROS
        if not rclpy.ok():
            rclpy.init()
        
        self.arm_name = arm_name
        
        # Create ROS abstraction layer (from crtk)
        self.ral = crtk.ral('dvrk_replay')
        
        # Create dVRK arm interface
        print(f"Connecting to {arm_name}...")
        self.psm = dvrk.psm(self.ral, arm_name)
        
        # Wait for arm to be ready
        print(f"Waiting for {arm_name} to be ready...")
        time.sleep(2)
        
    def load_kinematics(self, filepath, source_arm='PSM1'):
        """
        Load JIGSAW kinematics file
        
        Args:
            filepath: Path to kinematics file
            source_arm: Which arm's data to use from file ('PSM1' or 'PSM2')
        """
        print(f"Loading kinematics from {filepath}...")
        self.data = np.loadtxt(filepath)
        
        # Parse based on source arm
        if source_arm == 'PSM1':
            self.pos = self.data[:, 38:41]      # xyz
            self.rot = self.data[:, 41:50]      # rotation matrix (flattened)
            self.gripper = self.data[:, 56]     # gripper angle
        elif source_arm == 'PSM2':
            self.pos = self.data[:, 57:60]
            self.rot = self.data[:, 60:69]
            self.gripper = self.data[:, 75]
        elif source_arm == "PSM2_generated":
            if self.data.shape[1] == 10:

                self.pos = self.data[:, :3]
                rot_6d = self.data[:, 3:9]
                self.gripper = self.data[:, 9]

                self.rot = np.array([self.rotation_6d_to_matrix(rot_6d[i]) for i in range(len(rot_6d))])
            elif self.data.shape[1] == 19:
                self.pos = self.data[:, 0:3]
                self.rot = self.data[:, 3:12]
                self.gripper = self.data[:, 18]
        else:
            raise ValueError(f"Unknown source arm: {source_arm}")
        
        self.num_samples = len(self.data)
        print(f"Loaded {self.num_samples} samples from {source_arm} data")
        
    def create_frame(self, pos, rot_flat):
        """Create PyKDL Frame from position and flattened rotation matrix"""
        # Reshape rotation matrix
        R = rot_flat.reshape(3, 3)
        
        # Create PyKDL rotation
        rot = PyKDL.Rotation(
            R[0, 0], R[0, 1], R[0, 2],
            R[1, 0], R[1, 1], R[1, 2],
            R[2, 0], R[2, 1], R[2, 2]
        )
        
        # Create PyKDL vector
        vec = PyKDL.Vector(pos[0], pos[1], pos[2])
        
        # Create frame
        return PyKDL.Frame(rot, vec)
    
    def replay_trajectory(self, rate_hz=30, start_idx=0, end_idx=None):
        """
        Replay trajectory on the PSM
        
        Args:
            rate_hz: Playback rate in Hz (JIGSAW is typically 30Hz)
            start_idx: Start sample index
            end_idx: End sample index (None = end of trajectory)
        """
        if end_idx is None:
            end_idx = self.num_samples
        
        print(f"\nReplaying trajectory on {self.arm_name}...")
        print("Enabling and homing arm...")
        
        # Enable and home
        self.psm.enable()
        self.psm.home()
        
        print("Arm ready!")
        time.sleep(1)
        
        dt = 1.0 / rate_hz
        
        print(f"Starting playback at {rate_hz} Hz...")
        print(f"Playing samples {start_idx} to {end_idx}")
        
        for i in range(start_idx, end_idx):
            start_time = time.time()
            
            # Create target frame
            target_frame = self.create_frame(self.pos[i], self.rot[i])
            
            # Move arm
            self.psm.move_cp(target_frame).wait()
            
            # Set gripper (jaw angle in radians)
            self.psm.jaw.move_jp(np.array([self.gripper[i]])).wait()
            
            # Progress indicator
            if i % 30 == 0:  # Print every second at 30Hz
                progress = 100 * (i - start_idx) / (end_idx - start_idx)
                print(f"Sample {i}/{end_idx} ({progress:.1f}%)")
            
            # Maintain playback rate
            elapsed = time.time() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)
        
        print(f"Replay complete!")
    
    def print_statistics(self):
        """Print statistics about the loaded trajectory"""
        print(f"\nTrajectory Statistics:")
        print(f"  Number of samples: {self.num_samples}")
        print(f"  Duration at 30Hz: {self.num_samples/30:.2f} seconds")
        print(f"  Position range:")
        print(f"    X: [{self.pos[:,0].min():.4f}, {self.pos[:,0].max():.4f}] m")
        print(f"    Y: [{self.pos[:,1].min():.4f}, {self.pos[:,1].max():.4f}] m")
        print(f"    Z: [{self.pos[:,2].min():.4f}, {self.pos[:,2].max():.4f}] m")
        print(f"  Gripper angle range: [{self.gripper.min():.4f}, {self.gripper.max():.4f}] rad")


def main():
    """Main function"""
    
    # Configuration
    ARM_NAME = 'PSM1'  # Change to 'PSM2' if needed
    SOURCE_ARM = 'PSM1'  # Which arm's data to use from the file

    # NOTE: Adjust FILEPATH to match your JIGSAW dataset location
    # Expected structure: Gestures/[Task]/kinematics/AllGestures/[Trial].txt
    FILEPATH = "Gestures/Needle_Passing/kinematics/AllGestures/Needle_Passing_B001.txt"
    
    # Create replay
    replay = JIGSAWReplay(arm_name=ARM_NAME)
    
    # Load kinematics data
    replay.load_kinematics(FILEPATH, source_arm=SOURCE_ARM)
    
    # Show statistics
    replay.print_statistics()
    
    # Confirm before starting
    input("\nPress Enter to start replay (or Ctrl+C to cancel)...")
    
    # Replay trajectory
    replay.replay_trajectory(rate_hz=30)
    
    # Cleanup
    rclpy.shutdown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nReplay cancelled by user")
        rclpy.shutdown()

