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
from metrics import *#ade_loss, fde_loss, bivariate_loss

# [FIX] Define masked_mse_loss locally if import fails or for clarity
def masked_mse_loss(V_pred, V_trgt, mask=None):
    """
    Masked MSE Loss for warm-up.
    V_pred: [Batch, Time, Nodes, 5] (mu_x, mu_y, ...)
    V_trgt: [Batch, Time, Nodes, 2] (gt_x, gt_y)
    """
    mu_x = V_pred[..., 0]
    mu_y = V_pred[..., 1]
    
    x = V_trgt[..., 0]
    y = V_trgt[..., 1]
    
    loss = (x - mu_x)**2 + (y - mu_y)**2
    
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
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_gcnn', type=int, default=2, help='Number of GCN layers')
parser.add_argument('--n_tcnn', type=int, default=6, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--thres', type=float, default=0.3, help='Threshold to make connections between agents')

# Data specific parameters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth', help='Apolloscape,eth,hotel,univ,zara1,zara2,SDD')
parser.add_argument('--scene_name', default='bookstore', help='Scene name to train on [quad,nexus,little,hyang,gates,deathCircle,coupa,bookstore]')

# Training specific parameters
parser.add_argument('--batch_size', type=int, default=64, help='minibatch size (Virtual Batch Size for Gradient Accumulation)')
parser.add_argument('--num_epochs', type=int, default=150, help='number of epochs')
parser.add_argument('--clip_grad', type=float, default=None, help='gradient clipping')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=75, help='number of steps to drop the lr')
parser.add_argument('--use_lrschd', action="store_true", default=True, help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', help='personal tag for the model')
parser.add_argument('--delim', default='\t', help='Delimiter used in the dataset file')
parser.add_argument('--shuffle', action="store_true", default=False, help='Whether to shuffle the sequences')
parser.add_argument('--reload_data', action="store_true", default=False, help='Whether to reload the data from all files')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the raw dataset directory')
parser.add_argument('--skip_val', action="store_true", default=False, help='Whether to skip validation during training')
parser.add_argument('--log_dir', type=str, default="./logs", help='Directory to save logs')
parser.add_argument('--n_splits', type=int, default=1, help='(Deprecated) Number of splits')

args = parser.parse_args()


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
        
        batch = [tensor.cuda() for tensor in batch_tensors]
        
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

        optimizer.zero_grad() 

        # 3. Forward
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        
        # Extract map filenames from metadata tuples
        model_metadata = [m[0] for m in batch_metadata_list]
        
        V_pred, _ = model(V_obs_tmp, A_obs, model_metadata) 
        V_pred = V_pred.permute(0, 2, 3, 1)
        
        # 4. Loss
        mask_perm = loss_mask.permute(0, 2, 1)
        mask_perm = mask_perm[:, -args.pred_seq_len:, :]
        
        V_tr_perm = V_tr 
        loss = graph_loss(V_pred, V_tr_perm, mask_perm, use_mse=use_mse)
        
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


def calculate_ade_fde(model, loader_val, metrics):
    model.eval()
    ade_batch_list = []
    fde_batch_list = []
    
    with torch.no_grad():
        for batch in loader_val: 
             batch_tensors = batch[:-1]
             batch_metadata_list = batch[-1]
             
             batch = [tensor.cuda() for tensor in batch_tensors]
             # Note: We need obs_traj (for last pos) and pred_traj_gt (for absolute target)
             obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

             V_obs_tmp = V_obs.permute(0, 3, 1, 2)
             model_metadata = [m[0] for m in batch_metadata_list] 
             
             V_pred, _ = model(V_obs_tmp, A_obs, model_metadata)
             V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5]

             # --- INTEGRATION LOGIC (Make Absolute) ---
             # 1. Get Relative predictions (dx, dy)
             V_pred_rel = V_pred[..., :2]
             
             # 2. Integrate offsets
             V_pred_cumsum = torch.cumsum(V_pred_rel, dim=1)
             
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
                 _, orig_w, orig_h = batch_metadata_list[i]
                 
                 unscale_x = orig_w / 512.0
                 unscale_y = orig_h / 512.0
                 
                 valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                 num_valid = np.sum(valid_rows)
                 if num_valid == 0: num_valid = 1 

                 p_i = V_pred_np[i, :, :num_valid, :2].copy()
                 t_i = V_tr_np[i, :, :num_valid, :2].copy()

                 # --- APPLY UN-SCALING ---
                 p_i[..., 0] *= unscale_x
                 p_i[..., 1] *= unscale_y
                 
                 t_i[..., 0] *= unscale_x
                 t_i[..., 1] *= unscale_y
                 # ------------------------

                 pred_list.append(p_i)
                 target_list.append(t_i)
                 count_list.append(num_valid)

             ade_batch_list.append(ade(pred_list, target_list, count_list))
             fde_batch_list.append(fde(pred_list, target_list, count_list))

    final_ade = np.mean(ade_batch_list)
    final_fde = np.mean(fde_batch_list)
    
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
            
            batch = [tensor.cuda() for tensor in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            
            # Extract map filenames
            model_metadata = [m[0] for m in batch_metadata_list]
            
            V_pred, _ = model(V_obs_tmp, A_obs, model_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            mask_perm = loss_mask.permute(0, 2, 1)
            mask_perm = mask_perm[:, -args.pred_seq_len:, :]
            
            V_tr_perm = V_tr
            loss = graph_loss(V_pred, V_tr_perm, mask_perm, use_mse=use_mse)

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
    checkpoint_dir = './checkpoint/'+args.tag+'/'
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
    processed_train_dir = os.path.join('./processed/train', args.scene_name)
    processed_val_dir = os.path.join('./processed/val', args.scene_name)

    # Check if processed data exists for this scene
    files_exist = os.path.exists(processed_train_dir) and len(glob.glob(os.path.join(processed_train_dir, "*.pkl"))) > 0

    if args.reload_data or not files_exist:
        print(f"Processed data for {args.scene_name} missing or reload requested. Generating splits from RAW data...")
        # This generates ALL scenes into ./processed
        _ = TrajectoryDataset(
            data_dir=args.dataset_path,
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            delim=args.delim,
            dataset_name=args.dataset
        )
        print("Data generation complete.")

    print(f"Initializing Datasets for Scene: {args.scene_name}...")

    dset_train = TrajectoryDataset(
        data_dir=processed_train_dir,
        obs_len=args.obs_seq_len,
        pred_len=args.pred_seq_len,
        skip=1,
        norm_lap_matr=True,
        delim=args.delim,
        dataset_name=args.dataset
    )

    loader_train = DataLoader(
        dset_train,
        batch_size=args.batch_size, # Use actual batch size for training
        shuffle=args.shuffle,
        num_workers=4,
        collate_fn=TrajectoryDataset.collate_fn 
    )

    loader_val = None
    if not args.skip_val:
        dset_val = TrajectoryDataset(
            data_dir=processed_val_dir,
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            skip=1,
            norm_lap_matr=True,
            delim=args.delim,
            dataset_name=args.dataset
        )
        loader_val = DataLoader(
            dset_val,
            batch_size=args.batch_size, # Use actual batch size for validation
            shuffle=False,
            num_workers=4,
            collate_fn=TrajectoryDataset.collate_fn
        )

    print('Data loaded.')

    # -----------------------------------------------------------------------------
    # MODEL SETUP
    # -----------------------------------------------------------------------------
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
    # Optimizer and Scheduler
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=0.001) # Lower LR for Adam
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

    print("Starting Training Loop...")

    for epoch in range(args.num_epochs): 
        train(epoch, model, optimizer, loader_train, metrics)
        
        if not args.skip_val:
            vald(epoch, model, loader_val, metrics, constant_metrics)
        
        # Scheduler
        if args.use_lrschd:
            if scheduler is not None:
                current_train_loss = metrics['train_loss'][-1] if len(metrics['train_loss']) > 0 else float('inf')
                scheduler.step(current_train_loss)
                print(f"Learning Rate after Epoch {epoch}: {optimizer.param_groups[0]['lr']}")
                if len(metrics['train_loss']) > 0 and np.isnan(metrics['train_loss'][-1]):
                    print("NaN loss detected.")

        # Console Log
        print(f'Epoch: {epoch} | Train Loss: {metrics["train_loss"][-1]:.4f} | Val Loss: {metrics["val_loss"][-1]:.4f}')

        # Checkpoints
        checkpoint = {
            'state_dict': model.state_dict(),
            'scene_name': args.scene_name,
            'args': args,
            'epoch': epoch,
            'metrics': metrics
        }
        
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth'))
        
        curr_val_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
        
        # Use Loss as primary metric for best model unless you want ADE
        # save_metric = curr_val_loss 
        save_metric = metrics['ade'][-1] # Uncomment to save based on ADE
        
        if save_metric < best_val_loss:
            best_val_loss = save_metric
            best_model_state = model.state_dict() 
            torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"New Best Model Saved! ADE: {save_metric:.4f}")

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
        
        ade_calc, fde_calc = calculate_ade_fde(model, loader_val, metrics)
        print(f"Final Best Model - ADE: {ade_calc:.4f}, FDE: {fde_calc:.4f}")
        log_file.write(f"FINAL,,,{ade_calc},{fde_calc}\n")

    log_file.close()

