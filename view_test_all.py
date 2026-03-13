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
import seaborn as sns
import matplotlib.cm as cm

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
            
            # Unpack theta (11th element), if present
            if len(batch_tensors) == 11:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
                 theta = theta.cuda()
            else:
                 # Maintain backward compatibility if theta is missing
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
                 theta = None

            batch_tensors = [t.cuda() for t in batch_tensors if torch.is_tensor(t)]
            # Fix: Ensure all tensors are on CUDA, unpacking correctly after potentially stripping non-tensors
            # Wait, list comprehension above returns a new list. We need to unpack THIS list.
            if len(batch_tensors) == 11:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
            else:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            
            # Ensure specific tensors are on cuda (redundant but safe)
            obs_traj = obs_traj.cuda()
            
            V_obs_tmp = V_obs.permute(0, 3, 1, 2) 
            # NEW: Prepare Absolute Coordinates
            abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
            model_metadata = [m[0] for m in batch_metadata]
            # NEW: Pass abs_coords to the model
            V_pred, _ = model(V_obs_tmp, A_obs, abs_coords, model_metadata)
            
            V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5] 
            
            V_pred_rel = V_pred[..., :2]
            
            # --- INVERSE ROTATION ---
            if theta is not None:
                # theta: [Batch, Nodes] (padded)
                theta_exp = theta.unsqueeze(1).expand_as(V_pred_rel[..., 0])
                cos_th = torch.cos(theta_exp)
                sin_th = torch.sin(theta_exp)
                
                dx_local = V_pred_rel[..., 0]
                dy_local = V_pred_rel[..., 1]
                
                dx_global = dx_local * cos_th - dy_local * sin_th
                dy_global = dx_local * sin_th + dy_local * cos_th
                
                V_pred_rel = torch.stack([dx_global, dy_global], dim=-1)
            # ------------------------

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
            V_pred_params_np = V_pred.cpu().numpy()
            theta_np = theta.cpu().numpy() if theta is not None else None

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
                    
                    params = V_pred_params_np[i, :, ped_idx, :]
                    mu = params[:, :2]
                    log_sx = np.clip(params[:, 2], -20.0, 6.0)
                    log_sy = np.clip(params[:, 3], -20.0, 6.0)
                    sx = np.exp(log_sx)
                    sy = np.exp(log_sy)
                    corr = np.tanh(params[:, 4])
                    corr = np.clip(corr, -0.999, 0.999)

                    sample_rel = np.zeros((num_samples, mu.shape[0], 2), dtype=mu.dtype)
                    for t in range(mu.shape[0]):
                        cov = np.array([
                            [sx[t] * sx[t], corr[t] * sx[t] * sy[t]],
                            [corr[t] * sx[t] * sy[t], sy[t] * sy[t]]
                        ])
                        sample_rel[:, t, :] = np.random.multivariate_normal(mu[t], cov, size=num_samples)

                    # --- UNROTATE SAMPLES TO GLOBAL FRAME ---
                    if theta_np is not None:
                        th = theta_np[i, ped_idx]
                        cos_th_val = np.cos(th)
                        sin_th_val = np.sin(th)
                        dx_s = sample_rel[..., 0]
                        dy_s = sample_rel[..., 1]
                        unrot_dx_s = dx_s * cos_th_val - dy_s * sin_th_val
                        unrot_dy_s = dx_s * sin_th_val + dy_s * cos_th_val
                        sample_rel = np.stack([unrot_dx_s, unrot_dy_s], axis=-1)
                    # ----------------------------------------

                    last_obs = obs_traj_np[i, -1, ped_idx, :2]
                    sample_abs = np.cumsum(sample_rel, axis=1) + last_obs[None, None, :]
                    sample_abs[..., 0] *= unscale_x
                    sample_abs[..., 1] *= unscale_y
                    
                    ped_trajectories.append({
                        'ade': ped_ade,
                        'obs': ped_obs,
                        'pred': ped_pred,
                        'pred_samples': sample_abs,
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


import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Slider, Button

def draw_interactive_trajectory(traj, heat, no_map, title_text):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Extract data
    obs = traj['obs']
    pred_orig = traj['pred']
    gt = traj['gt']
    pred_samples = traj['pred_samples']
    meta_id = traj.get('meta_id')

    # Reconstruct the plottable "pred" line by picking the sample closest to GT at each timestep
    pred = np.zeros_like(gt)
    for t in range(gt.shape[0]):
        dists = np.linalg.norm(pred_samples[:, t, :] - gt[t], axis=-1)
        best_idx = np.argmin(dists)
        pred[t] = pred_samples[best_idx, t, :]
    
    sample_dists = np.mean(np.linalg.norm(pred_samples - np.expand_dims(gt, axis=0), axis=-1), axis=1)
    best_overall_idx = np.argmin(sample_dists)
    best_sample_traj = pred_samples[best_overall_idx, :, :]
    
    map_img_path = None
    if meta_id:
        map_img_name = meta_id.replace('_map.pt', '.jpg')
        map_img_path = os.path.join('./processed/maps', map_img_name)
        if not os.path.exists(map_img_path):
            map_img_path = None

    state = {
        'map_img_path': map_img_path,
        'bw': 0.7,
        'tau': 1.3, 
        'thresh': 0.05,
        'show_lines': False,
        'show_best_sample': False,
        'show_legend': True
    }

    plt.subplots_adjust(bottom=0.25)

    def draw():
        ax.clear()
        ax.axis('equal')
        
        if state['map_img_path'] and not no_map:
            try:
                img = mpimg.imread(state['map_img_path'])
                ax.imshow(img, extent=[0, img.shape[1], img.shape[0], 0])
                ax.set_xlim(0, img.shape[1])
                ax.set_ylim(img.shape[0], 0)
            except Exception as e:
                print(f"Could not load map: {e}")
        else:
            ax.grid(True, linestyle='--', alpha=0.6)
            all_x = np.concatenate([obs[:, 0], gt[:, 0], pred[:, 0]])
            all_y = np.concatenate([obs[:, 1], gt[:, 1], pred[:, 1]])
            cx, cy = np.mean(all_x), np.mean(all_y)
            window = 100 
            ax.set_xlim(cx - window, cx + window)
            ax.set_ylim(cy - window, cy + window) 

        ax.plot(obs[:, 0], obs[:, 1], color='blue', marker='o', linestyle='-', linewidth=2, label='Observed History', markersize=4)
        ax.plot(gt[:, 0], gt[:, 1], color='green', marker='s', linestyle='-', linewidth=2, label='Ground Truth Future', markersize=4)
        ax.plot([obs[-1, 0], gt[0, 0]], [obs[-1, 1], gt[0, 1]], color='green', linestyle='-', linewidth=2)
        
        if heat:
            import seaborn as sns
            all_x = pred_samples[:, :, 0].flatten()
            all_y = pred_samples[:, :, 1].flatten()
            
            dist_to_gt = np.linalg.norm(pred_samples - np.expand_dims(gt, axis=0), axis=-1)
            tau_val = (np.mean(dist_to_gt) + 1e-5) * state['tau']
            weights = np.exp(-dist_to_gt / tau_val).flatten()
            
            # Using thresh prevents small spills (values below a density threshold aren't drawn)
            sns.kdeplot(x=all_x, y=all_y, weights=weights, cmap='Reds', fill=True, alpha=0.5, 
                        bw_adjust=state['bw'], levels=100, thresh=state['thresh'], ax=ax)
            ax.plot([], [], color='red', linestyle='--',linewidth=1.5, label='Predicted Trajectories (Distribution)', alpha=0.5)

            if state.get('show_lines', False):
                ax.plot(pred[:, 0], pred[:, 1], color='orange', marker='X', linestyle='-', linewidth=2.5, markersize=6)
                ax.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], color='orange', linestyle='-', linewidth=2)
            
            if state.get('show_best_sample', False):
                ax.plot(best_sample_traj[:, 0], best_sample_traj[:, 1], color='darkred', marker='p', linestyle='solid', linewidth=2.5, markersize=5)
                ax.plot([obs[-1, 0], best_sample_traj[0, 0]], [obs[-1, 1], best_sample_traj[0, 1]], color='darkred', linestyle='solid', linewidth=2.5)
                
        else:
            colors = cm.Reds(np.linspace(0.4, 1.0, pred_samples.shape[0]))
            for s_idx in range(pred_samples.shape[0]):
                label = 'Predicted samples' if s_idx == 0 else None
                ax.plot(pred_samples[s_idx, :, 0], pred_samples[s_idx, :, 1], color=colors[s_idx], marker='*', markersize=2, linestyle='--', linewidth=1, alpha=0.6, label=label)
                ax.plot([obs[-1, 0], pred_samples[s_idx, 0, 0]], [obs[-1, 1], pred_samples[s_idx, 0, 1]], color=colors[s_idx], linestyle='--', linewidth=1, alpha=0.6)
                
            ax.plot(pred[:, 0], pred[:, 1], color='darkred', marker='*', linestyle='-', linewidth=2.5, label='Best Mean Prediction', markersize=6)
            ax.plot([obs[-1, 0], pred[0, 0]], [obs[-1, 1], pred[0, 1]], color='darkred', linestyle='-', linewidth=2)

        ax.set_title(title_text)
        if state.get('show_legend', True):
            ax.legend(loc='best')
        ax.set_xlabel("X Coordinate (Pixels)")
        ax.set_ylabel("Y Coordinate (Pixels)")
        fig.canvas.draw_idle()

    draw()

    # UI Elements
    axcolor = 'lightgoldenrodyellow'
    
    if heat:
        ax_bw = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
        ax_tau = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
        ax_thresh = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        
        s_bw = Slider(ax_bw, 'Spread', 0.1, 2.0, valinit=0.5)
        s_tau = Slider(ax_tau, 'Weight Decay', 0.1, 5.0, valinit=1.0)
        s_thresh = Slider(ax_thresh, 'Min Heatmap Thresh', 0.01, 0.5, valinit=0.05)
        
        def update(val):
            state['bw'] = s_bw.val
            state['tau'] = s_tau.val
            state['thresh'] = s_thresh.val
            draw()
            
        s_bw.on_changed(update)
        s_tau.on_changed(update)
        s_thresh.on_changed(update)

        ax_btn_lines = plt.axes([0.05, 0.12, 0.15, 0.05])
        btn_lines = Button(ax_btn_lines, 'Toggle Lines', color=axcolor, hovercolor='0.975')
        
        def toggle_lines(event):
            state['show_lines'] = not state['show_lines']
            draw()
            
        btn_lines.on_clicked(toggle_lines)
        
        ax_btn_best = plt.axes([0.05, 0.19, 0.15, 0.05])
        btn_best = Button(ax_btn_best, 'Toggle Best Sample', color=axcolor, hovercolor='0.975')
        
        def toggle_best_sample(event):
            state['show_best_sample'] = not state['show_best_sample']
            draw()
            
        btn_best.on_clicked(toggle_best_sample)

    ax_btn = plt.axes([0.05, 0.05, 0.15, 0.05])
    btn = Button(ax_btn, 'Toggle Legend', color=axcolor, hovercolor='0.975')
    
    def toggle_legend(event):
        state['show_legend'] = not state['show_legend']
        draw()

    btn.on_clicked(toggle_legend)
    plt.show()

def is_turn_trajectory(traj_dict, deviation_threshold=3.5):
    gt = traj_dict['gt']
    
    start = gt[0]
    end = gt[-1]
    
    line_vec = end - start
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-3:
        return False
    
    line_unit = line_vec / line_len
    
    # Calculate deviation of the future trajectory (gt) from the straight line between its start and end
    vecs = gt - start
    projs = np.sum(vecs * line_unit, axis=1)
    proj_pts = start + projs[:, None] * line_unit[None, :]
    
    # max distance between ground truth points and the projected points on the straight line
    max_dev = np.max(np.linalg.norm(gt - proj_pts, axis=1))
    
    return max_dev > deviation_threshold

def plot_top_5_trajectories(ped_trajectories, data_dir, no_map=False, heat=False, turn_only=False):
    if turn_only:
        moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0 and is_turn_trajectory(t)]
    else:
        moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0]
    if len(moving_trajectories) >= 5:
        trajectories_to_plot = moving_trajectories
    else:
        trajectories_to_plot = ped_trajectories
    trajectories_to_plot.sort(key=lambda x: x['ade'])
    top_5 = trajectories_to_plot[:5]

    for idx, traj in enumerate(top_5):
        title_text = f"Rank {idx+1} - Best Moving Prediction | ADE: {traj['ade']:.4f}"
        draw_interactive_trajectory(traj, heat, no_map, title_text)

def plot_bottom_5_trajectories(ped_trajectories, data_dir, no_map=False, heat=False, turn_only=False):
    if turn_only:
        moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0 and is_turn_trajectory(t)]
    else:
        moving_trajectories = [t for t in ped_trajectories if t['displacement'] > 15.0]
    if len(moving_trajectories) >= 5:
        trajectories_to_plot = moving_trajectories
    else:
        trajectories_to_plot = ped_trajectories
    trajectories_to_plot.sort(key=lambda x: x['ade'], reverse=True)
    bottom_5 = trajectories_to_plot[:5]

    for idx, traj in enumerate(bottom_5):
        title_text = f"Rank {idx+1} - Worst Moving Prediction | ADE: {traj['ade']:.4f}"
        draw_interactive_trajectory(traj, heat, no_map, title_text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_map', action='store_true', help='Disable map and show only close up of trajectories')
    parser.add_argument('--heat', action='store_true', help='Show heatmap mapped over predictions')
    parser.add_argument('--turn', action='store_true', help='Focus only on trajectories with turns')
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
        plot_top_5_trajectories(ped_trajectories, test_data_dir, no_map=cmd_args.no_map, heat=cmd_args.heat, turn_only=cmd_args.turn)
        plot_bottom_5_trajectories(ped_trajectories, test_data_dir, no_map=cmd_args.no_map, heat=cmd_args.heat, turn_only=cmd_args.turn)
if __name__ == '__main__':
    main()