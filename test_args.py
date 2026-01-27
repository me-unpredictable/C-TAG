import os
import time
import pickle
import glob
import math
import sys
import torch
import numpy as np
import argparse
import networkx as nx
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributions.multivariate_normal as torchdist

# Import your modules
from model import CTAG
from utils import TrajectoryDataset, seq_to_graph
from metrics import ade, fde

def get_model_path(checkpoint_root='./checkpoint/'):
    """
    Interactive function to select a trained model checkpoint.
    """
    # 1. List experiment directories
    subdirs = [d for d in glob.glob(os.path.join(checkpoint_root, '*')) if os.path.isdir(d)]
    subdirs.sort()
    
    if not subdirs:
        print(f"No checkpoint directories found in {checkpoint_root}")
        sys.exit(1)
        
    print("\nAvailable Experiments:")
    for i, path in enumerate(subdirs):
        print(f"[{i}] {os.path.basename(path)}")
        
    try:
        exp_idx = int(input("Select experiment index: "))
        exp_dir = subdirs[exp_idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)
        
    # 2. List .pth files in selected directory
    pth_files = glob.glob(os.path.join(exp_dir, "*.pth"))
    pth_files.sort()
    
    # Sort such that best_model is first or last, or by epoch
    print(f"\nAvailable Checkpoints in {os.path.basename(exp_dir)}:")
    for i, path in enumerate(pth_files):
        print(f"[{i}] {os.path.basename(path)}")
        
    try:
        pth_idx = int(input("Select checkpoint index: "))
        model_path = pth_files[pth_idx]
    except (ValueError, IndexError):
        print("Invalid selection.")
        sys.exit(1)
        
    args_path = os.path.join(exp_dir, 'args.pkl')
    if not os.path.exists(args_path):
        print(f"args.pkl not found in {exp_dir}. Cannot load model configuration.")
        sys.exit(1)
        
    return model_path, args_path

def evaluate(model, loader, args, num_samples=20):
    model.eval()
    
    ade_list = []
    fde_list = []
    
    print(f"Starting evaluation (K={num_samples})...")
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            # Unpack batch (consistent with train.py)
            batch_tensors = batch[:-1]
            batch_metadata = batch[-1][0] # Tuple (file_path, ...)
            
            batch_tensors = [t.cuda() for t in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            
            # Prepare inputs
            V_obs_tmp = V_obs.permute(0, 3, 1, 2) 
            
            # Forward pass
            V_pred, _ = model(V_obs_tmp, A_obs, batch_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            # ------------------------------------------------------------------
            # Best-of-N Sampling Logic
            # ------------------------------------------------------------------
            
            batch_size = V_pred.size(0)
            seq_len = V_pred.size(1)
            num_ped = V_pred.size(2)
            
            # Extract parameters
            mu_x = V_pred[..., 0]
            mu_y = V_pred[..., 1]
            sx = torch.exp(torch.clamp(V_pred[..., 2], max=5.0)) # Clamp log-variance to avoid explosion
            sy = torch.exp(torch.clamp(V_pred[..., 3], max=5.0))
            corr = torch.tanh(V_pred[..., 4]) * 0.99 # Soft clamp correlation
            
            # Build Covariance Matrix
            cov = torch.zeros(batch_size, seq_len, num_ped, 2, 2).cuda()
            cov[..., 0, 0] = sx * sx
            cov[..., 0, 1] = corr * sx * sy
            cov[..., 1, 0] = corr * sx * sy
            cov[..., 1, 1] = sy * sy
            
            mean = V_pred[..., 0:2]
            
            # Safe Multivariate Normal
            epsilon = 1e-4
            cov[..., 0, 0] += epsilon
            cov[..., 1, 1] += epsilon
            
            try:
                mvnormal = torchdist.MultivariateNormal(mean, cov)
            except Exception as e:
                print(f"Distribution error for batch: {e}")
                # Fallback: diagonal only
                cov[..., 0, 1] = 0
                cov[..., 1, 0] = 0
                mvnormal = torchdist.MultivariateNormal(mean, cov)

            # Ground Truth (absolute coordinates)
            # obs_traj: [Batch, NumPed, 2, SeqLen] (Check dimension ordering, usually comes from Dataset like this)
            # In utils.py: seq_to_graph takes [NumNodes, 2, SeqLen]
            # But the batch tensor is collated.
            
            # Usually: 
            # obs_traj is [Batch, NumPed, 2, ObsLen]
            # pred_traj_gt is [Batch, NumPed, 2, PredLen]
            
            # We need last observed position to convert relative predictions to absolute.
            # last_obs: [Batch, NumPed, 2]
            last_obs = obs_traj[:, :, :, -1] 
            
            batch_ade = []
            batch_fde = []
            
            for _ in range(num_samples):
                # Sample relative offsets: [Batch, SeqLen, NumPed, 2]
                sample_rel = mvnormal.sample()
                
                # We need to change dimensions to match [Batch, NumPed, 2, SeqLen] or similar for easier math
                # sample_rel is [Batch, SeqLen, NumPed, 2] -> permute to [Batch, NumPed, 2, SeqLen]
                sample_rel_p = sample_rel.permute(0, 2, 3, 1) # [Batch, NumPed, 2, SeqLen]
                
                # Cumsum over time (last dim)
                sample_cumsum = torch.cumsum(sample_rel_p, dim=-1)
                
                # Add starting position
                # last_obs unsqueezed: [Batch, NumPed, 2, 1]
                sample_abs = sample_cumsum + last_obs.unsqueeze(-1)
                
                # Get Ground Truth Absolute
                # pred_traj_gt: [Batch, NumPed, 2, SeqLen]
                gt_abs = pred_traj_gt
                
                # Calculate Error
                diff = sample_abs - gt_abs
                dist = torch.norm(diff, dim=2) # [Batch, NumPed, SeqLen] (Norm over x,y)
                
                # ADE: Mean over Time
                ade_val = dist.mean(dim=-1) # [Batch, NumPed]
                # FDE: Value at last Time
                fde_val = dist[:, :, -1] # [Batch, NumPed]
                
                batch_ade.append(ade_val)
                batch_fde.append(fde_val)
            
            # Best of K
            batch_ade = torch.stack(batch_ade) # [K, Batch, NumPed]
            batch_fde = torch.stack(batch_fde)
            
            min_ade, _ = torch.min(batch_ade, dim=0) # [Batch, NumPed]
            min_fde, _ = torch.min(batch_fde, dim=0) # [Batch, NumPed]
            
            ade_list.extend(min_ade.cpu().numpy().flatten().tolist())
            fde_list.extend(min_fde.cpu().numpy().flatten().tolist())

    print(f"\nFinal Results (Best-of-{num_samples}):")
    print(f"ADE: {np.mean(ade_list):.4f}")
    print(f"FDE: {np.mean(fde_list):.4f}")

def main():
    # 1. Get Model
    model_path, args_path = get_model_path()
    
    # 2. Load Args
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        
    print(f"\nConfiguration Loaded from: {args_path}")
    
    # 3. Setup Data
    test_data_dir = './processed/test'
    if not os.path.exists(test_data_dir):
        print(f"Error: {test_data_dir} does not exist.")
        # fallback to args.dataset path if 'processed' doesn't exist?
        # Assuming user has data here as requested.
        
    print(f"Loading Test Data from {test_data_dir}...")
    
    dset_test = TrajectoryDataset(
        data_dir=test_data_dir,
        obs_len=args.obs_seq_len,
        pred_len=args.pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        delim=args.delim,
        dataset_name=args.dataset
    )
    
    loader_test = DataLoader(
        dset_test,
        batch_size=1, 
        shuffle=False,
        num_workers=4,
        collate_fn=TrajectoryDataset.collate_fn
    )
    
    # 4. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = CTAG(
        n_gcnn=args.n_gcnn,
        n_tcnn=args.n_tcnn,
        output_feat=args.output_size,
        seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,
        pred_seq_len=args.pred_seq_len,
        threshold=args.thres
    ).to(device)
    
    # 5. Load Weights
    print(f"Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # Try cleaning buffer/metadata keys if mismatch?
        # Usually exact match is best.
        print("Note: If size mismatch occurs, architecture args in args.pkl might not match the saved model.")
        sys.exit(1)
        
    # 6. Run Evaluation
    # Default 20 samples for standard evaluation
    evaluate(model, loader_test, args, num_samples=20)

if __name__ == '__main__':
    main()
