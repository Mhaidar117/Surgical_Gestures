"""
Offline script to precompute RAFT optical flow.
"""
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
import pickle
from tqdm import tqdm

try:
    from raft import RAFT
    from utils.utils import InputPadder
    RAFT_AVAILABLE = True
except ImportError:
    RAFT_AVAILABLE = False
    print("Warning: RAFT not available. Install from: https://github.com/princeton-vl/RAFT")


def get_device():
    """Get the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        return 'mps'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def load_raft_model(weights_path: str, device: str = None):
    """Load RAFT model."""
    if not RAFT_AVAILABLE:
        raise ImportError("RAFT not available")
    
    if device is None:
        device = get_device()
    
    model = RAFT(weights_path)
    model = model.to(device)
    model.eval()
    return model


def compute_raft_flow(video_path: str, model, device: str = None) -> np.ndarray:
    """Compute RAFT flow for a video."""
    if device is None:
        device = get_device()
    
    cap = cv2.VideoCapture(video_path)
    flows = []
    
    ret, frame1 = cap.read()
    if not ret:
        return np.array([])
    
    frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    frame1 = torch.from_numpy(frame1).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
    
    padder = InputPadder(frame1.shape)
    frame1 = padder.pad(frame1)[0]
    
    while True:
        ret, frame2 = cap.read()
        if not ret:
            break
        
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
        frame2 = torch.from_numpy(frame2).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        frame2 = padder.pad(frame2)[0]
        
        with torch.no_grad():
            flow_low, flow_up = model(frame1, frame2, iters=20, test_mode=True)
        
        flow = flow_up[0].cpu().permute(1, 2, 0).numpy()  # (H, W, 2)
        flows.append(flow)
        
        frame1 = frame2
    
    cap.release()
    return np.array(flows)


def main():
    parser = argparse.ArgumentParser(description='Precompute RAFT flow')
    parser.add_argument('--video_dir', type=str, required=True, help='Video directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--raft_weights', type=str, required=True, help='Path to RAFT weights')
    parser.add_argument('--device', type=str, default=None, help='Device (auto-detects if not specified)')
    
    args = parser.parse_args()
    
    if not RAFT_AVAILABLE:
        print("RAFT not available. Please install it first.")
        return
    
    # Get device
    device = args.device if args.device else get_device()
    print(f'Using device: {device}')
    
    # Load model
    model = load_raft_model(args.raft_weights, device)
    
    # Process videos
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    video_files = list(video_dir.glob('*.avi'))
    
    for video_file in tqdm(video_files, desc='Processing videos'):
        flows = compute_raft_flow(str(video_file), model, device)
        
        # Save as float16 to save space
        flows = flows.astype(np.float16)
        
        output_file = output_dir / f"{video_file.stem}_raft.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(flows, f)
    
    print(f"Processed {len(video_files)} videos")


if __name__ == '__main__':
    main()

