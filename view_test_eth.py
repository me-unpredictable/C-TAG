import os
import pickle
import glob
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
from utils_by_scene_eth import TrajectoryDataset 
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
            
            if len(batch_tensors) == 11:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
                 theta = theta.cuda()
            else:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
                 theta = None

            batch_tensors = [t.cuda() for t in batch_tensors if torch.is_tensor(t)]
            if len(batch_tensors) == 11:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
            else:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            
            obs_traj = obs_traj.cuda()
            
            V_obs_tmp = V_obs.permute(0, 3, 1, 2) 
            abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
            model_metadata = [m[0] for m in batch_metadata]
            V_pred, _ = model(V_obs_tmp, A_obs, abs_coords, model_metadata)
            
            V_pred = V_pred.permute(0, 2, 3, 1) 
            V_pred_rel = V_pred[..., :2]
            
            if theta is not None:
                theta_exp = theta.unsqueeze(1).expand_as(V_pred_rel[..., 0])
                cos_th = torch.cos(theta_exp)
                sin_th = torch.sin(theta_exp)
                
                dx_local = V_pred_rel[..., 0]
                dy_local = V_pred_rel[..., 1]
                
                dx_global = dx_local * cos_th - dy_local * sin_th
                dy_global = dx_local * sin_th + dy_local * cos_th
                
                V_pred_rel = torch.stack([dx_global, dy_global], dim=-1)

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
                meta_tuple = batch_metadata[i]
                meta_id = meta_tuple[0] if isinstance(meta_tuple, tuple) else meta_tuple
                
                valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                num_valid = np.sum(valid_rows)
                if num_valid == 0: num_valid = 1 
                
                p_i = V_pred_np[i, :, :num_valid, :2].copy()
                t_i = V_tr_np[i, :, :num_valid, :2].copy()
                o_i = obs_traj_np[i, :, :num_valid, :2].copy()
                
                pred_list.append(p_i)
                target_list.append(t_i)
                count_list.append(num_valid)
                
                for ped_idx in range(num_valid):
                    ped_pred = p_i[:, ped_idx, :]
                    ped_gt = t_i[:, ped_idx, :]
                    ped_obs = o_i[:, ped_idx, :]
                    
                    ped_ade = np.mean(np.linalg.norm(ped_pred - ped_gt, axis=-1))
                    
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

    print(f"\nFinal Results (Meters ADE/FDE):")
    print(f"ADE: {np.mean(ade_list):.4f}")
    print(f"FDE: {np.mean(fde_list):.4f}")
    
    return ped_trajectories

def get_map_image(meta_id):
    if not meta_id:
        return None
    # Assuming meta_id format is "seq_eth_map.pt" and we want "seq_eth_map.png"
    # User specified map titles are like "seq_eth_map.png" in "maps" directory
    map_img_name = meta_id.replace('.pt', '.png')
    map_img_path = os.path.join('./maps', map_img_name)
    if not os.path.exists(map_img_path):
        # Fallback to older paths just in case
        map_img_path = os.path.join('./processed/maps', map_img_name)
    
    if os.path.exists(map_img_path):
        return map_img_path
    return None

def plot_trajectories(trajectories, title_prefix="Trajectory", no_map=False):
    for idx, traj in enumerate(trajectories):
        plt.figure(figsize=(10, 8))
        plt.axis('equal')
        
        meta_id = traj.get('meta_id')
        map_img_path = get_map_image(meta_id)
        
        obs = traj['obs']
        pred = traj['pred']
        gt = traj['gt']
        
        # Plot data 
        plt.plot(obs[:, 0], obs[:, 1], color='blue', marker='o', linestyle='-', linewidth=2, label='Observed', markersize=4)
        plt.plot(gt[:, 0], gt[:, 1], color='green', marker='s', linestyle='-', linewidth=2, label='Ground Truth', markersize=4)
        plt.plot(pred[:, 0], pred[:, 1], color='red', marker='*', linestyle='--', linewidth=2, label='Prediction', markersize=5)
        
        # Connections from history to future
        plt.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]], color='green', linestyle='-', linewidth=2)
        plt.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], color='red', linestyle='--', linewidth=2)
        
        if map_img_path and not no_map:
            img = mpimg.imread(map_img_path)
            # Displaying image with assumption that image corresponds to the real-world meter coordinates
            # Note: Explicit extent needs to be managed if pixels & meters mismatch
            plt.imshow(img)
            plt.grid(True, linestyle='--', alpha=0.6)
        else:
            plt.grid(True, linestyle='--', alpha=0.6)
            all_x = np.concatenate([obs[:, 0], gt[:, 0], pred[:, 0]])
            all_y = np.concatenate([obs[:, 1], gt[:, 1], pred[:, 1]])
            cx, cy = np.mean(all_x), np.mean(all_y)
            window = 10.0 # 10 meters window
            plt.xlim(cx - window, cx + window)
            plt.ylim(cy - window, cy + window) 

        plt.title(f"{title_prefix} {idx+1} | ADE: {traj['ade']:.4f}")
        plt.legend(loc='best')
        plt.xlabel("X Coordinate (Meters)")
        plt.ylabel("Y Coordinate (Meters)")
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_map', action='store_true', help='Disable map background')
    cmd_args = parser.parse_args()

    model_path, args_path = get_model_path()
    
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
        
    print(f"\nConfiguration Loaded from: {args_path}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Validation: Check if model trained on ETH
    train_dataset = getattr(args, 'dataset', '')
    if isinstance(checkpoint, dict) and not train_dataset:
         train_dataset = checkpoint.get('dataset', '')
         
    if str(train_dataset).lower() != 'eth':
         print(f"Error: Model was not trained on ETH. Dataset found: '{train_dataset}'")
         sys.exit(1)
    
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
        else:
             print(f"Test data directory {test_data_dir} not found.")
             sys.exit(1)
        
    print(f"Loading ETH Test Data from {test_data_dir}...")
    
    dset_test = TrajectoryDataset(
        data_dir=test_data_dir,
        obs_len=args.obs_seq_len,
        pred_len=args.pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        delim=args.delim,
        dataset_name='eth' # Explicitly ETH
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
        moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 1.0]
        
        if len(moving_trajectories) >= 5:
            plot_list = moving_trajectories
        else:
            plot_list = ped_trajectories
            
        plot_list.sort(key=lambda x: x['ade'])
        
        print("\nVisualizing Top 5 Best Moving Predictions...")
        plot_trajectories(plot_list[:5], "Rank Best", cmd_args.no_map)
        
        print("\nVisualizing Bottom 5 Worst Moving Predictions...")
        plot_list.sort(key=lambda x: x['ade'], reverse=True)
        plot_trajectories(plot_list[:5], "Rank Worst", cmd_args.no_map)

if __name__ == '__main__':
    main()
