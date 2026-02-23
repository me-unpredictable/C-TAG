import os
import time
import pickle
import glob
import math
import sys
import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Import your modules
from model import CTAG
from utils_by_scene import TrajectoryDataset 
from metrics import ade, fde

def get_model_path(checkpoint_root='./checkpoint/'):
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
        
    pth_files = glob.glob(os.path.join(exp_dir, "*.pth"))
    pth_files.sort()
    
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
    ped_trajectories = []
    
    print(f"Starting evaluation...")
    pbar = tqdm(loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in pbar:
            batch_tensors = batch[:-1]
            batch_metadata = batch[-1]
            
            batch_tensors = [t.cuda() for t in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            
            V_obs_tmp = V_obs.permute(0, 3, 1, 2) 
            # NEW: Prepare Absolute Coordinates
            abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
            model_metadata = [m[0] for m in batch_metadata]
            # NEW: Pass abs_coords to the model
            V_pred, _ = model(V_obs_tmp, A_obs, abs_coords, model_metadata)
            
            V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5] 
            
            V_pred_rel = V_pred[..., :2]
            V_pred_cumsum = torch.cumsum(V_pred_rel, dim=1)
            
            last_obs = obs_traj[:, :, :, -1].unsqueeze(1) 
            V_pred_abs = V_pred_cumsum + last_obs 
            
            V_tr_abs = pred_traj_gt.permute(0, 3, 1, 2)
            obs_traj_abs = obs_traj.permute(0, 3, 1, 2)
            
            batch_size = V_pred.shape[0]
            V_pred_np = V_pred_abs.cpu().numpy()
            V_tr_np = V_tr_abs.cpu().numpy()
            obs_traj_np = obs_traj_abs.cpu().numpy()
            loss_mask_np = loss_mask.cpu().numpy()

            pred_list = []
            target_list = []
            count_list = []

            for i in range(batch_size):
                meta_id, orig_w, orig_h = batch_metadata[i]
                unscale_x = orig_w / 512.0
                unscale_y = orig_h / 512.0
                
                valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                num_valid = np.sum(valid_rows)
                if num_valid == 0: num_valid = 1 
                
                p_i = V_pred_np[i, :, :num_valid, :2].copy()
                t_i = V_tr_np[i, :, :num_valid, :2].copy()
                o_i = obs_traj_np[i, :, :num_valid, :2].copy()
                
                p_i[..., 0] *= unscale_x
                p_i[..., 1] *= unscale_y
                t_i[..., 0] *= unscale_x
                t_i[..., 1] *= unscale_y
                o_i[..., 0] *= unscale_x
                o_i[..., 1] *= unscale_y

                pred_list.append(p_i)
                target_list.append(t_i)
                count_list.append(num_valid)
                
                for ped_idx in range(num_valid):
                    ped_pred = p_i[:, ped_idx, :]
                    ped_gt = t_i[:, ped_idx, :]
                    ped_obs = o_i[:, ped_idx, :]
                    
                    ped_ade = np.mean(np.linalg.norm(ped_pred - ped_gt, axis=-1))
                    
                    # Track physical movement distance to filter out stationary people
                    displacement = np.linalg.norm(ped_gt[-1] - ped_obs[0])
                    
                    ped_trajectories.append({
                        'ade': ped_ade,
                        'obs': ped_obs,
                        'pred': ped_pred,
                        'gt': ped_gt,
                        'displacement': displacement,
                        'meta_id': meta_id
                    })

            ade_list.append(ade(pred_list, target_list, count_list))
            fde_list.append(fde(pred_list, target_list, count_list))

    print(f"\nFinal Results (Standard Absolute ADE/FDE):")
    print(f"ADE: {np.mean(ade_list):.4f}")
    print(f"FDE: {np.mean(fde_list):.4f}")
    
    return ped_trajectories

def plot_top_5_trajectories(ped_trajectories, data_dir, no_map=False):
    # Filter out people who moved less than 15 pixels overall
    moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0]
    
    if len(moving_trajectories) >= 5:
        trajectories_to_plot = moving_trajectories
        print("\nFiltered out stationary agents (displacement < 15px) for clearer visualization.")
    else:
        trajectories_to_plot = ped_trajectories
        print("\nNot enough moving agents found. Showing best available agents.")

    trajectories_to_plot.sort(key=lambda x: x['ade'])
    top_5 = trajectories_to_plot[:5]

    print("\nVisualizing the Top 5 Moving Predictions (Lowest ADE)...")
    for idx, traj in enumerate(top_5):
        plt.figure(figsize=(10, 8))
        
        # Enforce True Aspect Ratio (1 pixel X = 1 pixel Y)
        plt.axis('equal')
        
        meta_id = traj.get('meta_id')
        map_img_path = None
        if meta_id:
            map_img_name = meta_id.replace('_map.pt', '.jpg')
            map_img_path = os.path.join('./processed/maps', map_img_name)
            if not os.path.exists(map_img_path):
                map_img_path = None
        
        obs = traj['obs']
        pred = traj['pred']
        gt = traj['gt']
        
        plt.plot(obs[:, 0], obs[:, 1], color='blue', marker='o', linestyle='-', linewidth=2, label='Observed History', markersize=4)
        plt.plot(gt[:, 0], gt[:, 1], color='green', marker='s', linestyle='-', linewidth=2, label='Ground Truth Future', markersize=4)
        plt.plot(pred[:, 0], pred[:, 1], color='red', marker='*', linestyle='--', linewidth=2, label='Predicted Future', markersize=5)
        
        plt.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]], color='green', linestyle='-', linewidth=2)
        plt.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], color='red', linestyle='--', linewidth=2)
        
        if map_img_path and not no_map:
            img = mpimg.imread(map_img_path)
            # Load Map and Align with standard image coordinates (Y=0 at Top)
            plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0) # Inverted Y for image
        else:
            plt.grid(True, linestyle='--', alpha=0.6)
            # Determine the center of the trajectory and create a fixed 150-pixel viewing window
            all_x = np.concatenate([obs[:, 0], gt[:, 0], pred[:, 0]])
            all_y = np.concatenate([obs[:, 1], gt[:, 1], pred[:, 1]])
            cx, cy = np.mean(all_x), np.mean(all_y)
            window = 100 
            plt.xlim(cx - window, cx + window)
            plt.ylim(cy - window, cy + window) 

        plt.title(f"Rank {idx+1} - Best Moving Prediction | ADE: {traj['ade']:.4f}")
        plt.legend(loc='best')
        plt.xlabel("X Coordinate (Pixels)")
        plt.ylabel("Y Coordinate (Pixels)")
        
        plt.show()

def plot_bottom_5_trajectories(ped_trajectories, data_dir, no_map=False):
    # Filter out people who moved less than 15 pixels overall
    moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0]
    
    if len(moving_trajectories) >= 5:
        trajectories_to_plot = moving_trajectories
        print("\nFiltered out stationary agents (displacement < 15px) for clearer visualization.")
    else:
        trajectories_to_plot = ped_trajectories
        print("\nNot enough moving agents found. Showing worst available agents.")

    trajectories_to_plot.sort(key=lambda x: x['ade'], reverse=True)
    bottom_5 = trajectories_to_plot[:5]

    print("\nVisualizing the Bottom 5 Moving Predictions (Highest ADE)...")
    for idx, traj in enumerate(bottom_5):
        plt.figure(figsize=(10, 8))
        
        # Enforce True Aspect Ratio (1 pixel X = 1 pixel Y)
        plt.axis('equal')
        
        meta_id = traj.get('meta_id')
        map_img_path = None
        if meta_id:
            map_img_name = meta_id.replace('_map.pt', '.jpg')
            map_img_path = os.path.join('./processed/maps', map_img_name)
            if not os.path.exists(map_img_path):
                map_img_path = None
        
        obs = traj['obs']
        pred = traj['pred']
        gt = traj['gt']
        
        plt.plot(obs[:, 0], obs[:, 1], color='blue', marker='o', linestyle='-', linewidth=2, label='Observed History', markersize=4)
        plt.plot(gt[:, 0], gt[:, 1], color='green', marker='s', linestyle='-', linewidth=2, label='Ground Truth Future', markersize=4)
        plt.plot(pred[:, 0], pred[:, 1], color='red', marker='*', linestyle='--', linewidth=2, label='Predicted Future', markersize=5)
        
        plt.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]], color='green', linestyle='-', linewidth=2)
        plt.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], color='red', linestyle='--', linewidth=2)
        
        if map_img_path and not no_map:
            img = mpimg.imread(map_img_path)
            # Load Map and Align with standard image coordinates (Y=0 at Top)
            plt.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
            plt.xlim(0, img.shape[1])
            plt.ylim(img.shape[0], 0) # Inverted Y for image
        else:
            plt.grid(True, linestyle='--', alpha=0.6)
            # Determine the center of the trajectory and create a fixed 150-pixel viewing window
            all_x = np.concatenate([obs[:, 0], gt[:, 0], pred[:, 0]])
            all_y = np.concatenate([obs[:, 1], gt[:, 1], pred[:, 1]])
            cx, cy = np.mean(all_x), np.mean(all_y)
            window = 100 
            plt.xlim(cx - window, cx + window)
            plt.ylim(cy - window, cy + window) 

        plt.title(f"Rank {idx+1} - Worst Moving Prediction | ADE: {traj['ade']:.4f}")
        plt.legend(loc='best')
        plt.xlabel("X Coordinate (Pixels)")
        plt.ylabel("Y Coordinate (Pixels)")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_map', action='store_true', help='Disable map and show only close up of trajectories')
    cmd_args = parser.parse_args()

    model_path, args_path = get_model_path()
    
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        
    print(f"\nConfiguration Loaded from: {args_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    scene_name = None
    state_dict = checkpoint
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        if 'scene_name' in checkpoint:
            scene_name = checkpoint['scene_name']
    
    test_data_dir = os.path.join('./processed/test', str(scene_name) if scene_name else '')
    
    if not os.path.exists(test_data_dir):
        if os.path.exists('./processed/test'):
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
    
    model = CTAG(
        n_gcnn=args.n_gcnn,
        n_tcnn=args.n_tcnn,
        output_feat=args.output_size,
        seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,
        pred_seq_len=args.pred_seq_len,
        threshold=args.thres
    ).to(device)
    
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        sys.exit(1)
        
    ped_trajectories = evaluate(model, loader_test, args, num_samples=20)
    
    if ped_trajectories:
        plot_top_5_trajectories(ped_trajectories, test_data_dir, no_map=cmd_args.no_map)
        plot_bottom_5_trajectories(ped_trajectories, test_data_dir, no_map=cmd_args.no_map)
if __name__ == '__main__':
    main()