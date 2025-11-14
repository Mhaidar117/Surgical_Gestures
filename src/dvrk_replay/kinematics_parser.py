#!/usr/bin/env python3
"""
JIGSAW Needle Passing Kinematics Parser
Parses the 76-column kinematics data from JIGSAW dataset
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class JIGSAWKinematics:
    """Parse and process JIGSAW kinematics data"""
    
    def __init__(self, filepath):
        """
        Load kinematics file
        
        Args:
            filepath: Path to kinematics .txt file
        """
        self.data = np.loadtxt(filepath)
        self.parse_data()
        
    def parse_data(self):
        """Parse the 76 columns into structured format"""
        # Master Left (MTML) - columns 0-18
        self.mtml_pos = self.data[:, 0:3]           # xyz position
        self.mtml_rot = self.data[:, 3:12]          # 3x3 rotation matrix (flattened)
        self.mtml_vel = self.data[:, 12:15]         # linear velocity
        self.mtml_ang_vel = self.data[:, 15:18]     # angular velocity
        self.mtml_gripper = self.data[:, 18]        # gripper angle
        
        # Master Right (MTMR) - columns 19-37
        self.mtmr_pos = self.data[:, 19:22]
        self.mtmr_rot = self.data[:, 22:31]
        self.mtmr_vel = self.data[:, 31:34]
        self.mtmr_ang_vel = self.data[:, 34:37]
        self.mtmr_gripper = self.data[:, 37]
        
        # Slave Left (PSM1) - columns 38-56
        self.psm1_pos = self.data[:, 38:41]
        self.psm1_rot = self.data[:, 41:50]
        self.psm1_vel = self.data[:, 50:53]
        self.psm1_ang_vel = self.data[:, 53:56]
        self.psm1_gripper = self.data[:, 56]
        
        # Slave Right (PSM2) - columns 57-75
        self.psm2_pos = self.data[:, 57:60]
        self.psm2_rot = self.data[:, 60:69]
        self.psm2_vel = self.data[:, 69:72]
        self.psm2_ang_vel = self.data[:, 72:75]
        self.psm2_gripper = self.data[:, 75]
        
        self.num_samples = len(self.data)
        
    def get_rotation_matrix(self, rot_flat, idx):
        """Reshape flattened rotation matrix"""
        return rot_flat[idx].reshape(3, 3)
    
    def plot_trajectories(self, arm='PSM1'):
        """Plot 3D trajectory of specified arm"""
        fig = plt.figure(figsize=(12, 5))
        
        # Get data for specified arm
        if arm == 'PSM1':
            pos = self.psm1_pos
            gripper = self.psm1_gripper
        elif arm == 'PSM2':
            pos = self.psm2_pos
            gripper = self.psm2_gripper
        elif arm == 'MTML':
            pos = self.mtml_pos
            gripper = self.mtml_gripper
        elif arm == 'MTMR':
            pos = self.mtmr_pos
            gripper = self.mtmr_gripper
        else:
            raise ValueError(f"Unknown arm: {arm}")
        
        # 3D trajectory
        ax1 = fig.add_subplot(121, projection='3d')
        scatter = ax1.scatter(pos[:, 0], pos[:, 1], pos[:, 2], 
                             c=gripper, cmap='RdYlGn', s=10)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'{arm} 3D Trajectory (colored by gripper angle)')
        plt.colorbar(scatter, ax=ax1, label='Gripper Angle (rad)')
        
        # Gripper angle over time
        ax2 = fig.add_subplot(122)
        ax2.plot(gripper)
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Gripper Angle (rad)')
        ax2.set_title(f'{arm} Gripper Angle Over Time')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def plot_velocities(self, arm='PSM1'):
        """Plot velocity profiles"""
        if arm == 'PSM1':
            vel = self.psm1_vel
            ang_vel = self.psm1_ang_vel
        elif arm == 'PSM2':
            vel = self.psm2_vel
            ang_vel = self.psm2_ang_vel
        elif arm == 'MTML':
            vel = self.mtml_vel
            ang_vel = self.mtml_ang_vel
        elif arm == 'MTMR':
            vel = self.mtmr_vel
            ang_vel = self.mtmr_ang_vel
            
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Linear velocity
        axes[0].plot(vel[:, 0], label='vx')
        axes[0].plot(vel[:, 1], label='vy')
        axes[0].plot(vel[:, 2], label='vz')
        axes[0].set_ylabel('Linear Velocity (m/s)')
        axes[0].set_title(f'{arm} Velocities')
        axes[0].legend()
        axes[0].grid(True)
        
        # Angular velocity
        axes[1].plot(ang_vel[:, 0], label='ωx')
        axes[1].plot(ang_vel[:, 1], label='ωy')
        axes[1].plot(ang_vel[:, 2], label='ωz')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Angular Velocity (rad/s)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def get_statistics(self):
        """Print statistics about the trajectory"""
        print(f"Number of samples: {self.num_samples}")
        print(f"\nPSM1 Statistics:")
        print(f"  Position range: X=[{self.psm1_pos[:,0].min():.3f}, {self.psm1_pos[:,0].max():.3f}]")
        print(f"                  Y=[{self.psm1_pos[:,1].min():.3f}, {self.psm1_pos[:,1].max():.3f}]")
        print(f"                  Z=[{self.psm1_pos[:,2].min():.3f}, {self.psm1_pos[:,2].max():.3f}]")
        print(f"  Gripper angle range: [{self.psm1_gripper.min():.3f}, {self.psm1_gripper.max():.3f}] rad")
        
        print(f"\nPSM2 Statistics:")
        print(f"  Position range: X=[{self.psm2_pos[:,0].min():.3f}, {self.psm2_pos[:,0].max():.3f}]")
        print(f"                  Y=[{self.psm2_pos[:,1].min():.3f}, {self.psm2_pos[:,1].max():.3f}]")
        print(f"                  Z=[{self.psm2_pos[:,2].min():.3f}, {self.psm2_pos[:,2].max():.3f}]")
        print(f"  Gripper angle range: [{self.psm2_gripper.min():.3f}, {self.psm2_gripper.max():.3f}] rad")


# Example usage
if __name__ == "__main__":
    # Load the data
    filepath = "/home/mai/JISSAW/Needle_Passing/kinematics/AllGestures/Needle_Passing_B001.txt"
    
    print("Loading JIGSAW kinematics data...")
    kin = JIGSAWKinematics(filepath)
    
    # Print statistics
    kin.get_statistics()
    
    # Plot trajectories
    print("\nPlotting PSM1 trajectory...")
    kin.plot_trajectories('PSM1')
    
    print("\nPlotting PSM2 trajectory...")
    kin.plot_trajectories('PSM2')
    
    # Plot velocities
    print("\nPlotting PSM1 velocities...")
    kin.plot_velocities('PSM1')
