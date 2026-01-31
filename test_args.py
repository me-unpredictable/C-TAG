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
from utils_by_scene import TrajectoryDataset 
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
    
    print(f"Starting evaluation...")
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            # Unpack batch (consistent with train.py)
            batch_tensors = batch[:-1]
            batch_metadata = batch[-1] # Pass full list of metadata
            
            batch_tensors = [t.cuda() for t in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            
            # Prepare inputs
            V_obs_tmp = V_obs.permute(0, 3, 1, 2) 
            
            # Forward pass
            V_pred, _ = model(V_obs_tmp, A_obs, batch_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5]
            
            # ------------------------------------------------------------------
            # Evaluation Logic (Standard Absolute ADE/FDE)
            # ------------------------------------------------------------------
            
            # Convert to Absolute Coordinates
            V_pred_rel = V_pred[..., :2]
            V_pred_cumsum = torch.cumsum(V_pred_rel, dim=1)
            
            # obs_traj is [Batch, Nodes, 2, Time]
            last_obs = obs_traj[:, :, :, -1] # [Batch, Nodes, 2]
            last_obs = last_obs.unsqueeze(1) # [Batch, 1, Nodes, 2]
            
            V_pred_abs = V_pred_cumsum + last_obs # [Batch, Time, Nodes, 2]
            
            # GT Absolute
            # pred_traj_gt is [Batch, Nodes, 2, Time] -> Permute to [Batch, Time, Nodes, 2]
            V_tr_abs = pred_traj_gt.permute(0, 3, 1, 2)
            
            batch_size = V_pred.shape[0]
            V_pred_np = V_pred_abs.cpu().numpy()
            V_tr_np = V_tr_abs.cpu().numpy() # Absolute Target
            loss_mask_np = loss_mask.cpu().numpy()

            pred_list = []
            target_list = []
            count_list = []

            for i in range(batch_size):
                valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                num_valid = np.sum(valid_rows)
                if num_valid == 0: num_valid = 1 
                
                # Take Mean Prediction (mu_x, mu_y)
                p_i = V_pred_np[i, :, :num_valid, :2]
                t_i = V_tr_np[i, :, :num_valid, :2]

                pred_list.append(p_i)
                target_list.append(t_i)
                count_list.append(num_valid)

            ade_list.append(ade(pred_list, target_list, count_list))
            fde_list.append(fde(pred_list, target_list, count_list))

    print(f"\nFinal Results (Standard Absolute ADE/FDE):")
    print(f"ADE: {np.mean(ade_list):.4f}")
    print(f"FDE: {np.mean(fde_list):.4f}")

def main():
    # 1. Get Model
    model_path, args_path = get_model_path()
    
    # 2. Load Args (Base config)
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        
    print(f"\nConfiguration Loaded from: {args_path}")
    
    # 2.5 Peek at checkpoint for Metadata (Scene Name)
    print(f"Inspecting checkpoint metadata from {model_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set weights_only=False to allow loading argparse.Namespace embedded in checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scene_name = None
    state_dict = checkpoint
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        print("Detected new checkpoint format.")
        state_dict = checkpoint['state_dict']
        if 'scene_name' in checkpoint:
            scene_name = checkpoint['scene_name']
            print(f"Found embedded scene name: {scene_name}")
    
    # 3. Setup Data
    print(f"Targeting Scene: {scene_name}")
    
    test_data_dir = os.path.join('./processed/test', scene_name)
    
    if not os.path.exists(test_data_dir):
        print(f"Error: {test_data_dir} does not exist.")
        # fallback to just ./processed/test if specific scene not found? 
        # But user wants strict separation.
        print("Checking parent test dir...")
        if os.path.exists('./processed/test'):
             print("Warning: Specific scene dir not found, using generic ./processed/test")
             test_data_dir = './processed/test'
        
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
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        collate_fn=TrajectoryDataset.collate_fn
    )
    
    # 4. Initialize Model
    # device already set
    
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
    print(f"Applying weights...")
    try:
        model.load_state_dict(state_dict)
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
