# exclusive trainer for ETH dataset, with the new rotation handling and absolute coordinate integration logic.
import os
import math
import sys
import time
import pickle
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm  # [ADDED] For progress bars

# FORCE GPU 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from model import CTAG
from utils_by_scene import TrajectoryDataset
from utils_by_scene_eth import TrajectoryDataset as TrajectoryDatasetETH
from metrics import *#ade_loss, fde_loss, bivariate_loss

def extract_model_metadata(batch_metadata_list):
    model_metadata = []
    for meta_item in batch_metadata_list:
        if isinstance(meta_item, (list, tuple)) and len(meta_item) > 0:
            model_metadata.append(meta_item[0])
        else:
            model_metadata.append(meta_item)
    return model_metadata

def convert_meters_to_pixels(abs_coords, batch_metadata_list):
    """Projects meter coordinates to pixel coordinates using Homography."""
    abs_coords_px = abs_coords.clone()
    batch_size = abs_coords.shape[0]
    
    for b_idx in range(batch_size):
        meta_tuple = batch_metadata_list[b_idx]
        if isinstance(meta_tuple, tuple) and len(meta_tuple) >= 4:
            H = meta_tuple[3] 
            H_tensor = torch.tensor(H, dtype=torch.float32, device=abs_coords.device)
            
            x = abs_coords[b_idx, 0, :, :]
            y = abs_coords[b_idx, 1, :, :]
            
            x_prime = H_tensor[0,0]*x + H_tensor[0,1]*y + H_tensor[0,2]
            y_prime = H_tensor[1,0]*x + H_tensor[1,1]*y + H_tensor[1,2]
            z_prime = H_tensor[2,0]*x + H_tensor[2,1]*y + H_tensor[2,2]
            
            # Avoid division by zero
            z_prime = torch.clamp(z_prime, min=1e-6)
            
            u = x_prime / z_prime
            v = y_prime / z_prime
            
            abs_coords_px[b_idx, 0, :, :] = u
            abs_coords_px[b_idx, 1, :, :] = v
            
    return abs_coords_px

# [FIX] Define masked_mse_loss locally if import fails or for clarity
def masked_mse_loss(V_pred, V_trgt, mask=None):
    """
    Replaced with Smooth L1 (Huber) Loss for warm-up.
    This is much more forgiving to large deviations (sharp turns).
    V_pred: [Batch, Time, Nodes, 5] (mu_x, mu_y, ...)
    V_trgt: [Batch, Time, Nodes, 2] (gt_x, gt_y)
    """
    # Extract coordinates
    mu = V_pred[..., :2]    # Shape: [Batch, Time, Nodes, 2]
    target = V_trgt[..., :2] # Shape: [Batch, Time, Nodes, 2]
    
    # Calculate Smooth L1 Loss (beta=1.0 is default PyTorch behavior)
    # Reduces explosion of gradients on sharp curve misses
    loss = torch.nn.functional.smooth_l1_loss(mu, target, reduction='none')
    
    # Sum the loss over the X and Y coordinates
    loss = torch.sum(loss, dim=-1)
    
    if mask is not None:
        loss = loss.masked_fill(~mask.bool(), 0.0)
        num_valid = torch.sum(mask)
        if num_valid > 0:
            return torch.sum(loss) / num_valid
        return torch.tensor(0.0, device=loss.device)
    
    return torch.mean(loss)

def graph_loss(V_pred, V_target, mask=None, use_mse=False):
    if mask is not None:
         # Move mask to same device
         mask = mask.to(V_pred.device)
         
         if use_mse:
            return masked_mse_loss(V_pred, V_target, mask)
         else:
            return bivariate_loss(V_pred, V_target, mask)
    else:
        if use_mse:
            return masked_mse_loss(V_pred, V_target)
        else:
            return bivariate_loss(V_pred, V_target)

# Global Args Parsing
parser = argparse.ArgumentParser()

# Model specific parameters
parser.add_argument('--input_size', type=int, default=4) # [dx, dy, speed, heading angle]
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_gcnn', type=int, default=2, help='Number of GCN layers')
parser.add_argument('--n_tcnn', type=int, default=6, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--thres', type=float, default=0.3, help='Threshold to make connections between agents')

# Data specific parameters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='Dataset to train on (eth or ucy)')
parser.add_argument('--scene_name', default='eth', help='Scene name to train on (eth, hotel, univ, zara1, zara2)')

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=64, help='minibatch size (Virtual Batch Size for Gradient Accumulation)')
parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=3, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=75, help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='personal tag for the model')
parser.add_argument('--delim', default='\t', help='Delimiter used in the dataset file')
parser.add_argument('--shuffle', action="store_true", default=False, help='Whether to shuffle the sequences')
parser.add_argument('--reload_data', action="store_true", default=False, help='Whether to reload the data from all files')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the raw dataset directory')
parser.add_argument('--skip_val', action="store_true", default=False, help='Whether to skip validation during training')
parser.add_argument('--save_all', action="store_true", default=False, help='Whether to save all models during training')
parser.add_argument('--log_dir', type=str, default="./logs", help='Directory to save logs')
parser.add_argument('--n_splits', type=int, default=1, help='(Deprecated) Number of splits')

args = parser.parse_args()


def get_expected_split_pkl_files(split_dir, split_name):
    pkl_files = sorted(glob.glob(os.path.join(split_dir, "*.pkl")))
    if not pkl_files:
        return []

    expected_suffix = f"_{split_name}.pkl"
    invalid_files = [
        path for path in pkl_files
        if not os.path.basename(path).endswith(expected_suffix)
    ]

    if invalid_files:
        raise RuntimeError(
            f"Unexpected files found in {split_dir} for split '{split_name}': {invalid_files}"
        )

    return pkl_files


# -----------------------------------------------------------------------------
# TRAINING FUNCTIONS
# -----------------------------------------------------------------------------

def train(epoch, model, optimizer, loader_train, metrics):
    model.train()
    loss_batch = 0 
    
    # [ADDED] Progress Bar
    use_mse = (epoch < 30)
    desc_str = f"Epoch {epoch} [Train MSE]" if use_mse else f"Epoch {epoch} [Train NLL]"
    pbar = tqdm(loader_train, desc=desc_str, unit="batch")

    for cnt, batch in enumerate(pbar): 
        # 1. Unpack 
        batch_tensors = batch[:-1] 
        batch_metadata_list = batch[-1]
        
        # [FIX] Robust Unpacking for Theta
        if len(batch_tensors) == 11:
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
            theta = theta.to(next(model.parameters()).device)
        else:
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch_tensors
            theta = None
        
        # Move to GPU
        obs_traj = obs_traj.to(next(model.parameters()).device)
        pred_traj_gt = pred_traj_gt.to(next(model.parameters()).device)
        obs_traj_rel = obs_traj_rel.to(next(model.parameters()).device)
        pred_traj_gt_rel = pred_traj_gt_rel.to(next(model.parameters()).device)
        non_linear_ped = non_linear_ped.to(next(model.parameters()).device)
        loss_mask = loss_mask.to(next(model.parameters()).device)
        V_obs = V_obs.to(next(model.parameters()).device)
        A_obs = A_obs.to(next(model.parameters()).device)
        V_tr = V_tr.to(next(model.parameters()).device)
        A_tr = A_tr.to(next(model.parameters()).device)

        optimizer.zero_grad() 

        # 3. Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        # Prepare Absolute Coordinates for VSIE Map Sampling
        # obs_traj is [Batch, Nodes, 2, Time] -> Convert to [Batch, 2, Time, Nodes]
        abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
        abs_coords_px = convert_meters_to_pixels(abs_coords, batch_metadata_list)
        model_metadata = extract_model_metadata(batch_metadata_list)

        V_pred, _ = model(V_obs_tmp, A_obs, abs_coords_px, model_metadata)
        V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5]
        
        # --- CANONICAL UN-ROTATION ---
        # The model predicts in the agent-centric (rotated) frame. We must rotate the
        # predicted relative steps back to the global map frame before integration.
        V_pred_rel = V_pred[..., :2]
        
        # theta is shape [Batch, Nodes]. Reshape for broadcasting [Batch, 1, Nodes, 1]
        cos_th = torch.cos(theta).unsqueeze(1).unsqueeze(-1)
        sin_th = torch.sin(theta).unsqueeze(1).unsqueeze(-1)

        dx = V_pred_rel[..., 0:1]
        dy = V_pred_rel[..., 1:2]

        # Inverse rotation matrix (rotate by +theta)
        unrot_dx = dx * cos_th - dy * sin_th
        unrot_dy = dx * sin_th + dy * cos_th
        V_pred_rel_global = torch.cat([unrot_dx, unrot_dy], dim=-1)

        # --- INTEGRATE TO ABSOLUTE FOR LOSS ---
        V_pred_cumsum = torch.cumsum(V_pred_rel_global, dim=1)
        last_obs = obs_traj[:, :, :, -1].unsqueeze(1) # [Batch, 1, Nodes, 2]
        V_pred_abs_mu = V_pred_cumsum + last_obs
        
        # Reattach sigmas 
        V_pred_abs = torch.cat([V_pred_abs_mu, V_pred[..., 2:]], dim=-1)
        
        # Target is absolute global
        V_tr_abs = pred_traj_gt.permute(0, 3, 1, 2)
        
        mask_perm = loss_mask.permute(0, 2, 1)
        mask_perm = mask_perm[:, -args.pred_seq_len:, :]
        
        loss = graph_loss(V_pred_abs, V_tr_abs, mask_perm, use_mse=use_mse)

        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss) or (loss.item() == 0 and epoch > 0):
            continue 

        # 5. Backprop
        loss.backward()
        
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        optimizer.step()
        
        loss_batch += loss.item()
        pbar.set_postfix({'Loss': loss_batch / (cnt + 1)})
            
    metrics['train_loss'].append(loss_batch / len(loader_train))


def calculate_ade_fde(model, loader_val, metrics, record_metrics=True):
    model.eval()
    ade_total = 0.0
    fde_total = 0.0
    total_sequences = 0
    
    with torch.no_grad():
        for batch in loader_val: 
             batch_tensors = batch[:-1]
             # Unpack theta from the batch tensors (now the 11th tensor)
             obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch_tensors
             
             batch_metadata_list = batch[-1]
             
             # Move tensors to GPU
             obs_traj = obs_traj.to(next(model.parameters()).device)
             pred_traj_gt = pred_traj_gt.to(next(model.parameters()).device)
             obs_traj_rel = obs_traj_rel.to(next(model.parameters()).device)
             pred_traj_gt_rel = pred_traj_gt_rel.to(next(model.parameters()).device)
             non_linear_ped = non_linear_ped.to(next(model.parameters()).device)
             loss_mask = loss_mask.to(next(model.parameters()).device)
             V_obs = V_obs.to(next(model.parameters()).device)
             A_obs = A_obs.to(next(model.parameters()).device)
             V_tr = V_tr.to(next(model.parameters()).device)
             A_tr = A_tr.to(next(model.parameters()).device)
             theta = theta.to(next(model.parameters()).device)

             V_obs_tmp = V_obs.permute(0, 3, 1, 2)
             model_metadata = extract_model_metadata(batch_metadata_list)
             
             # NEW: Prepare Absolute Coordinates
             abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
             abs_coords_px = convert_meters_to_pixels(abs_coords, batch_metadata_list)
             
             # NEW: Pass abs_coords to the model
             V_pred, _ = model(V_obs_tmp, A_obs, abs_coords_px, model_metadata)
             
             V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5]

             # --- INTEGRATION LOGIC (Make Absolute) ---
             # 1. Get Relative predictions (dx, dy)
             V_pred_rel = V_pred[..., :2]
             
             # Un-rotate the predictions before integration
             # theta is [Batch, Max_Agents], expand to [Batch, Time, Max_Agents] for broadcasting
             theta_exp = theta.unsqueeze(1).expand_as(V_pred_rel[..., 0])
             
             cos_th = torch.cos(theta_exp)
             sin_th = torch.sin(theta_exp)
             
             # Rotated Back:
             # x' = x cos(th) - y sin(th)  (To rotate back by +theta? No, we rotated by -theta to align)
             # To reverse -theta rotation, we rotate by +theta. 
             # x_global = x_local * cos(theta) - y_local * sin(theta) 
             # y_global = x_local * sin(theta) + y_local * cos(theta)
             
             dx_local = V_pred_rel[..., 0]
             dy_local = V_pred_rel[..., 1]
             
             dx_global = dx_local * cos_th - dy_local * sin_th
             dy_global = dx_local * sin_th + dy_local * cos_th
             
             # Re-stack
             V_pred_rel_global = torch.stack([dx_global, dy_global], dim=-1)
             
             # 2. Integrate offsets
             V_pred_cumsum = torch.cumsum(V_pred_rel_global, dim=1)
             
             # 3. Add Last Observed Position
             # obs_traj is [Batch, Nodes, 2, Time] -> Get last time step
             last_obs = obs_traj[:, :, :, -1] # [Batch, Nodes, 2]
             last_obs = last_obs.unsqueeze(1) # [Batch, 1, Nodes, 2]
             
             # 4. Final Absolute Prediction (Scaled 0-512)
             V_pred_abs = V_pred_cumsum + last_obs # [Batch, Time, Nodes, 2]
             
             # 5. Prepare Target (Absolute)
             # pred_traj_gt is [Batch, Nodes, 2, Time] -> Permute to [Batch, Time, Nodes, 2]
             V_tr_abs = pred_traj_gt.permute(0, 3, 1, 2)

             batch_size = V_pred.shape[0]
             V_pred_np = V_pred_abs.cpu().numpy()
             V_tr_np = V_tr_abs.cpu().numpy()
             loss_mask_np = loss_mask.cpu().numpy()

             pred_list = []
             target_list = []
             count_list = []

             for i in range(batch_size):
                 valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                 num_valid = np.sum(valid_rows)
                 if num_valid == 0: num_valid = 1 

                 p_i = V_pred_np[i, :, :num_valid, :2].copy()
                 t_i = V_tr_np[i, :, :num_valid, :2].copy()

                 pred_list.append(p_i)
                 target_list.append(t_i)
                 count_list.append(num_valid)

             if pred_list:
                 batch_sequence_count = len(pred_list)
                 ade_total += ade(pred_list, target_list, count_list) * batch_sequence_count
                 fde_total += fde(pred_list, target_list, count_list) * batch_sequence_count
                 total_sequences += batch_sequence_count

    if total_sequences == 0:
        final_ade = 0.0
        final_fde = 0.0
    else:
        final_ade = ade_total / total_sequences
        final_fde = fde_total / total_sequences
    
    if record_metrics:
        metrics['ade'].append(final_ade)
        metrics['fde'].append(final_fde)
    
    return final_ade, final_fde

def vald(epoch, model, loader_val, metrics, constant_metrics):
    model.eval()
    loss_batch = 0 
    
    use_mse = (epoch < 30)
    desc_str = f"Epoch {epoch} [Val MSE]" if use_mse else f"Epoch {epoch} [Val NLL]"
    pbar = tqdm(loader_val, desc=desc_str, unit="batch")
    
    with torch.no_grad():
        for cnt, batch in enumerate(pbar): 
            batch_tensors = batch[:-1] 
            batch_metadata_list = batch[-1]
            
            batch = [tensor.to(next(model.parameters()).device) for tensor in batch_tensors]
            
            # [FIX] Unpack theta (11th element), if present. Handle backward compatibility.
            if len(batch) == 11:
                obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr, theta = batch
            else:
                 obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch
                 theta = None

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            # Prepare Absolute Coordinates for VSIE
            abs_coords = obs_traj.permute(0, 2, 3, 1).contiguous()
            abs_coords_px = convert_meters_to_pixels(abs_coords, batch_metadata_list)
            
            model_metadata = extract_model_metadata(batch_metadata_list)
            V_pred, _ = model(V_obs_tmp, A_obs, abs_coords_px, model_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1)
            # --- CANONICAL UN-ROTATION ---
            V_pred_rel = V_pred[..., :2]
            cos_th = torch.cos(theta).unsqueeze(1).unsqueeze(-1)
            sin_th = torch.sin(theta).unsqueeze(1).unsqueeze(-1)
            dx = V_pred_rel[..., 0:1]
            dy = V_pred_rel[..., 1:2]
            
            unrot_dx = dx * cos_th - dy * sin_th
            unrot_dy = dx * sin_th + dy * cos_th
            V_pred_rel_global = torch.cat([unrot_dx, unrot_dy], dim=-1)

            V_pred_cumsum = torch.cumsum(V_pred_rel_global, dim=1)
            last_obs = obs_traj[:, :, :, -1].unsqueeze(1) 
            V_pred_abs_mu = V_pred_cumsum + last_obs
            
            V_pred_abs = torch.cat([V_pred_abs_mu, V_pred[..., 2:]], dim=-1)
            V_tr_abs = pred_traj_gt.permute(0, 3, 1, 2)
            
            mask_perm = loss_mask.permute(0, 2, 1)
            mask_perm = mask_perm[:, -args.pred_seq_len:, :]
            
            loss = graph_loss(V_pred_abs, V_tr_abs, mask_perm, use_mse=use_mse)

            loss_batch += loss.item()
            pbar.set_postfix({'Loss': loss_batch / (cnt + 1)})

    avg_val_loss = loss_batch / len(loader_val)
    metrics['val_loss'].append(avg_val_loss)
    
    if avg_val_loss < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = avg_val_loss
        constant_metrics['min_val_epoch'] = epoch
    
    # Calculate ADE/FDE using the helper function
    # We call it here to record metrics during training loop if desired, 
    # but the original code structure had it separate or returned.
    # We will assume we want to track it.
    ade_, fde_ = calculate_ade_fde(model, loader_val, metrics)
    
    print(f"\tEpoch {epoch} Val Stats - Loss: {avg_val_loss:.4f} | ADE: {ade_:.4f} | FDE: {fde_:.4f}")

    # Note: calculate_ade_fde appends to metrics, so we don't need to do it here.
    return ade_, fde_

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print('*'*30)
    print("Training initiating....")
    print(args)

    # Create log directories
    os.makedirs(args.log_dir, exist_ok=True)
    checkpoint_dir = './checkpoint/'+'ETH_'+args.tag+'/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Log file setup
    log_file = open(os.path.join(args.log_dir, time.ctime()+'_log.txt'), 'w')
    log_file.write(str(args)+'\n')
    log_file.write('Epoch,Train_loss,Val_loss,Val_ADE,Val_FDE\n')

    # Save args
    with open(checkpoint_dir+'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    # -----------------------------------------------------------------------------
    # DATASET INITIALIZATION
    # -----------------------------------------------------------------------------
    from torch.utils.data import ConcatDataset

    # Scene selection logic for ETH/UCY datasets
    scene_arg = args.scene_name.lower()
    if scene_arg == 'eth':
        scenes = ['eth']
    elif scene_arg == 'hotel':
        scenes = ['hotel']
    elif scene_arg == 'univ':
        scenes = ['univ']
    elif scene_arg in ['zara1', 'zara01']:
        scenes = ['zara01']
    elif scene_arg in ['zara2', 'zara02']:
        scenes = ['zara02']
    else:
        print(f"Invalid scene name '{args.scene_name}' provided. Defaulting to all scenes.")
        scenes = ['eth', 'hotel', 'univ', 'zara01', 'zara02']
    
    train_datasets = []
    val_datasets = []

    # Select correct dataset class
    DatasetClass = TrajectoryDatasetETH
    
    # Pre-check if all data exists
    needs_generation = args.reload_data
    if not needs_generation:
        for scene in scenes:
            processed_train_dir = os.path.join('./processed/train', scene)
            processed_val_dir = os.path.join('./processed/val', scene)
            processed_test_dir = os.path.join('./processed/test', scene)
            
            t_ex = os.path.exists(processed_train_dir) and len(get_expected_split_pkl_files(processed_train_dir, 'train')) > 0
            v_ex = os.path.exists(processed_val_dir) and len(get_expected_split_pkl_files(processed_val_dir, 'val')) > 0
            ts_ex = os.path.exists(processed_test_dir) and len(get_expected_split_pkl_files(processed_test_dir, 'test')) > 0
            
            if not (t_ex and v_ex and ts_ex):
                needs_generation = True
                break

    if needs_generation:
        print("Missing processed splits or reload requested. Generating splits from RAW data...")
        # This generates ALL scenes into ./processed
        _ = DatasetClass(
            data_dir=args.dataset_path,
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            delim=args.delim,
            dataset_name=args.dataset,
            reload_data=True
        )
        print("Data generation complete.")

    for scene in scenes:
        processed_train_dir = os.path.join('./processed/train', scene)
        processed_val_dir = os.path.join('./processed/val', scene)
        train_pkl_files = get_expected_split_pkl_files(processed_train_dir, 'train')

        print(f"Initializing Datasets for Scene: {scene}...")
        print(f"Using train split files: {[os.path.basename(path) for path in train_pkl_files]}")

        dset_train = DatasetClass(
            data_dir=processed_train_dir,
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            delim=args.delim,
            dataset_name=args.dataset
        )
        train_datasets.append(dset_train)

        if not args.skip_val:
            val_pkl_files = get_expected_split_pkl_files(processed_val_dir, 'val')
            print(f"Using val split files: {[os.path.basename(path) for path in val_pkl_files]}")
            dset_val = DatasetClass(
                data_dir=processed_val_dir,
                obs_len=args.obs_seq_len,
                pred_len=args.pred_seq_len,
                skip=1,
                norm_lap_matr=True,
                delim=args.delim,
                dataset_name=args.dataset
            )
            val_datasets.append(dset_val)

    # Combine datasets
    combined_train_dset = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]

    loader_train = DataLoader(
        combined_train_dset,
        batch_size=args.batch_size, # Use actual batch size for training
        shuffle=False, # can't shuffle scene based data
        num_workers=4,
        collate_fn=DatasetClass.collate_fn 
    )

    loader_val = None
    if not args.skip_val:
        combined_val_dset = ConcatDataset(val_datasets) if len(val_datasets) > 1 else val_datasets[0]
        loader_val = DataLoader(
            combined_val_dset,
            batch_size=args.batch_size, # Use actual batch size for validation
            shuffle=False,
            num_workers=4,
            collate_fn=DatasetClass.collate_fn
        )

    print('Data loaded.')

    # -----------------------------------------------------------------------------
    # MODEL SETUP
    # -----------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CTAG(
        threshold=args.thres,
        n_gcnn=args.n_gcnn,
        n_tcnn=args.n_tcnn,
        input_feat=args.input_size,
        output_feat=args.output_size,
        seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,
        pred_seq_len=args.pred_seq_len
    ).to(device)
    # Optimizer and Scheduler
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4) # Lower LR for Adam
    if args.use_lrschd:
        # Changed to ReduceLROnPlateau for better convergence checking
      scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.1,       # Reduce by 10% instead of 90%
    patience=30,      # Wait 30 epochs before reducing (was 10)
    threshold=1e-2, 
    threshold_mode='abs',
    min_lr=1e-5
)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    else:
        scheduler = None

    # -----------------------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------------------
    
    metrics = {'train_loss': [], 'val_loss': [], 'ade': [], 'fde': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}

    best_val_loss = float('inf')
    best_model_state = None

    # Early Stopping Variables
    early_stop_counter = 0
    early_stop_patience = 20
    early_stop_tolerance = 0.001  # Tolerance as discussed
    min_lr_threshold = 1.1e-5     # Slightly above 1e-5 to safely catch it if float precision varies
    
    print("Starting Training Loop...")

    for epoch in range(args.num_epochs): 
        train(epoch, model, optimizer, loader_train, metrics)
        
        if not args.skip_val:
            vald(epoch, model, loader_val, metrics, constant_metrics)
        
        current_lr = optimizer.param_groups[0]['lr']
        current_val_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')

        # Scheduler
        if args.use_lrschd:
            if scheduler is not None:
                # Use Validation Loss as it's the standard for ReduceLROnPlateau
                # and aligns with our Early Stopping metric
                current_monitor_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
                scheduler.step(current_monitor_loss)
                
                print(f"Learning Rate after Epoch {epoch}: {optimizer.param_groups[0]['lr']}")
                if len(metrics['train_loss']) > 0 and np.isnan(metrics['train_loss'][-1]):
                    print("NaN loss detected.")

        # --- Early Stopping/NaN Logic ---
        # NaN Handling: If train_loss or val_loss is NaN/Inf, reduce LR
        # If already at minimum LR, stop training.

        has_nan = False
        if len(metrics['train_loss']) > 0 and (np.isnan(metrics['train_loss'][-1]) or np.isinf(metrics['train_loss'][-1])):
            has_nan = True
        if len(metrics['val_loss']) > 0 and (np.isnan(metrics['val_loss'][-1]) or np.isinf(metrics['val_loss'][-1])):
            has_nan = True

        if has_nan:
            print("*" * 50)
            print(" [WARNING] NaN or Inf detected in loss.")
            if optimizer.param_groups[0]['lr'] <= min_lr_threshold:
                 print(f" [EarlyStopping] LR is at min ({optimizer.param_groups[0]['lr']:.6f}) and NaN detected. Terminating training.")
                 break
            else:
                 new_lr = max(1e-5, optimizer.param_groups[0]['lr'] * 0.1)
                 for param_group in optimizer.param_groups:
                     param_group['lr'] = new_lr
                 print(f" [EarlyStopping] Dropping LR to {new_lr:.6f} due to NaN.")
                 print("*" * 50)
                 continue # Proceed to next epoch without evaluating early stopping counts

        # 1. Check if we are at the lowest LR
        if optimizer.param_groups[0]['lr'] <= min_lr_threshold:
             # Check improvement against the best observed loss so far
             # constant_metrics['min_val_loss'] holds the absolute best validation loss seen
             
            print(f" [EarlyStopping] LR is at minimum ({optimizer.param_groups[0]['lr']:.6f})...")

            # Check if current epoch's loss improved significantly over the 'best'
            if metrics['val_loss'][-1] > (constant_metrics['min_val_loss'] - early_stop_tolerance):
                 early_stop_counter += 1
                 print(f" [EarlyStopping] No significant improvement (curr: {metrics['val_loss'][-1]:.4f} vs best: {constant_metrics['min_val_loss']:.4f}). Counter: {early_stop_counter}/{early_stop_patience}")
            else:
                 early_stop_counter = 0
                 print(f" [EarlyStopping] Improvement detected! Counter reset.")

            if early_stop_counter >= early_stop_patience:
                 print("*"*50)
                 print(f" [EarlyStopping] Triggered! LR is min and no improvement for {early_stop_patience} epochs.")
                 print("*"*50)
                 break
        else:
             early_stop_counter = 0

        # Console Log
        print(f'Epoch: {epoch} | Train Loss: {metrics["train_loss"][-1]:.4f} | Val Loss: {metrics["val_loss"][-1]:.4f}')

        # Checkpoints
        checkpoint = {
            'state_dict': model.state_dict(),
            'dataset_name': 'eth',
            'scene_name': args.scene_name,
            'args': args,
            'epoch': epoch,
            'metrics': metrics
        }
        
        if args.save_all:
            torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth'))
        
        curr_val_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
        
        # Always check and save the best model at every epoch
        # Using ADE as the primary metric for saving the best model
        save_metric = metrics['ade'][-1] if len(metrics['ade']) > 0 else curr_val_loss
        
        if save_metric < best_val_loss:
            best_val_loss = save_metric
            best_model_state = model.state_dict() 
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New Best Model Saved! Metric: {save_metric:.4f}")

        with open(os.path.join(checkpoint_dir, 'metrics.pkl'), 'wb') as fp:
            pickle.dump(metrics, fp)
        
        t_loss = metrics['train_loss'][-1] if metrics['train_loss'] else 0
        v_loss = metrics['val_loss'][-1] if metrics['val_loss'] else 0
        curr_ade = metrics['ade'][-1] if metrics['ade'] else 0
        curr_fde = metrics['fde'][-1] if metrics['fde'] else 0
        log_file.write(f"{epoch},{t_loss},{v_loss},{curr_ade},{curr_fde}\n")

    if not args.skip_val:
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        elif os.path.exists(os.path.join(checkpoint_dir, 'best_model.pth')):
            checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), weights_only=False)
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        ade_calc, fde_calc = calculate_ade_fde(model, loader_val, metrics, record_metrics=False)
        print(f"Final Best Model - ADE: {ade_calc:.4f}, FDE: {fde_calc:.4f}")
        log_file.write(f"FINAL,,,{ade_calc},{fde_calc}\n")

    log_file.close()

