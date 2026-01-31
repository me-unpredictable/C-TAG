# here we are training with gradient accumulation
# so we load one batch at a time and accumulate gradients
# until we reach the desired virtual batch size
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
parser.add_argument('--batch_size', type=int, default=512, help='minibatch size (Virtual Batch Size for Gradient Accumulation)')
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
    batch_size=1, # Must be 1 for variable sequence length
    shuffle=args.shuffle,
    num_workers=0,
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
        batch_size=1,
        shuffle=False,
        num_workers=0,
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
        factor=0.1, 
        patience=10, 
        threshold=1e-2, 
        threshold_mode='abs',
        min_lr=1e-4
    )
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
else:
    scheduler = None

# -----------------------------------------------------------------------------
# TRAINING FUNCTIONS
# -----------------------------------------------------------------------------

def train(epoch, model, optimizer, loader_train, metrics):
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    
    # Track actual accumulated samples for correct normalization
    accumulation_count = 0
    
    # [ADDED] Progress Bar
    use_mse = (epoch < 30)
    desc_str = f"Epoch {epoch} [Train MSE]" if use_mse else f"Epoch {epoch} [Train NLL]"
    pbar = tqdm(loader_train, desc=desc_str, unit="seq")

    for cnt, batch in enumerate(pbar): 
        batch_count += 1
        accumulation_count += 1

        # 1. Unpack 
        batch_tensors = batch[:-1] # Slice off metadata
        # print("Batch tensors length:", len(batch_tensors))
        # print('Batch metadata:', batch[-1]) # batch[-1] shows size of each sequence in the batch
        batch_metadata = batch[-1][0] # it returns a tubple with feature map pt file name hence selecting [0]
        
        batch = [tensor.cuda() for tensor in batch_tensors]
        
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch


        # 3. Forward
        # optimizer.zero_grad() # MOVED to step 5
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        
        V_pred, _ = model(V_obs_tmp, A_obs,batch_metadata) 
        
        V_pred = V_pred.permute(0, 2, 3, 1)
        

        # 4. Loss
        # Ensure mask is [Batch, Time, Nodes] to match V_pred structure
        # loss_mask is [Batch, Nodes, Total_Time] -> Permute to [Batch, Total_Time, Nodes]
        mask_perm = loss_mask.permute(0, 2, 1)
        
        # SLICE MASK: The loss is calculated only on prediction steps
        # loss_mask covers obs + pred. We take the last pred_seq_len steps.
        mask_perm = mask_perm[:, -args.pred_seq_len:, :]
        
        # [FIX] V_tr from dataset is (B, T, V, C). 
        # Loss expects (B, T, V, C).
        # V_tr_perm = V_tr.permute(0, 3, 2, 1) # OLD BUG
        
        # It seems V_tr is ALREADY (B, T, V, C) based on utils_by_scene.py pad_V returning (T, N, D)
        # So no permutation needed, or identity.
        V_tr_perm = V_tr # (B, T, V, C)
        
        l = graph_loss(V_pred, V_tr_perm, mask_perm, use_mse=use_mse)
        
        # [ADDED] Check for NaN/Inf in individual batch loss
        if torch.isnan(l) or torch.isinf(l) or (l.item() == 0 and epoch > 0):
            # print(f"[WARNING] Invalid loss detected at Epoch {epoch}, Batch {cnt}. Value: {l.item()}")
            # print("Skipping backward pass for this batch to avoid poisoning model weights.")
            is_fst_loss = True
            loss = 0
            continue 

        if is_fst_loss:
            loss = l
            is_fst_loss = False
        else:
            loss = loss + l

        # 5. Backprop (Gradient Accumulation)
        turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
        
        if batch_count % args.batch_size == 0 or cnt == turn_point:
            optimizer.zero_grad() # Correct placement for Gradient Accumulation
            
            # FIX: Normalize by actual number of accumulated samples (handles last batch correctly)
            loss = loss / accumulation_count
            
            loss.backward()
            
            # [MODIFIED] Only clip gradients if explicitly requested
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            
            optimizer.step()
            
            # Record metrics
            current_loss = loss.item()
            loss_batch += current_loss
            
            # Update Progress Bar with current loss
            pbar.set_postfix({'Loss': loss_batch / (cnt + 1) * args.batch_size})
            
            is_fst_loss = True
            loss = 0 
            accumulation_count = 0 

    metrics['train_loss'].append(loss_batch / (loader_len / args.batch_size))

def vald(epoch, model, loader_val, metrics, constant_metrics):
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    
    # [ADDED] Progress Bar
    use_mse = (epoch < 30)
    desc_str = f"Epoch {epoch} [Val MSE]" if use_mse else f"Epoch {epoch} [Val NLL]"
    pbar = tqdm(loader_val, desc=desc_str, unit="seq")
    
    with torch.no_grad():
        for cnt, batch in enumerate(pbar): 
            batch_count += 1

            batch_tensors = batch[:-1] # Slice off metadata
            # print("Batch tensors length:", len(batch_tensors))
            # print('Batch metadata:', batch[-1]) # batch[-1] shows size of each sequence in the batch
            batch_metadata = batch[-1][0] # it returns a tubple with feature map pt file name hence selecting [0]
            
            batch = [tensor.cuda() for tensor in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            
            V_pred, _ = model(V_obs_tmp, A_obs,batch_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1)
            
            mask_perm = loss_mask.permute(0, 2, 1)
            # SLICE MASK (Validation too)
            mask_perm = mask_perm[:, -args.pred_seq_len:, :]
            
            # [FIX] V_tr from dataset is (B, T, V, C). 
            V_tr_perm = V_tr
            
            l = graph_loss(V_pred, V_tr_perm, mask_perm, use_mse=use_mse)

            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss = loss + l
            
            turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
            
            if batch_count % args.batch_size == 0 or cnt == turn_point:
                loss = loss / args.batch_size
                loss_batch += loss.item()
                is_fst_loss = True
                loss = 0
                pbar.set_postfix({'Loss': loss_batch / (cnt + 1) * args.batch_size})

    avg_val_loss = loss_batch / (loader_len / args.batch_size)
    metrics['val_loss'].append(avg_val_loss)
    
    if avg_val_loss < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = avg_val_loss
        constant_metrics['min_val_epoch'] = epoch

def calculate_ade_fde(model, loader_val):
    model.eval()
    ade_ls = []
    fde_ls = []
    
    print("Calculating ADE/FDE...")
    pbar = tqdm(loader_val, desc="ADE/FDE", unit="seq")

    with torch.no_grad():
        for batch in pbar:
            batch_tensors = batch[:-1]
            batch_metadata = batch[-1][0]
            batch = [tensor.cuda() for tensor in batch_tensors]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            
            V_pred, _ = model(V_obs_tmp, A_obs,batch_metadata)
            V_pred = V_pred.permute(0, 2, 3, 1)

            # --- FIX: Unpack Batch for ADE/FDE calculation ---
            batch_size = V_pred.shape[0]
            pred_list = []
            target_list = []
            count_list = []
            
            V_pred_np = V_pred.cpu().numpy()
            # [FIX] V_tr also needs permutation here to match V_pred structure for ADE/FDE logic
            V_tr_perm = V_tr
            V_tr_np = V_tr_perm.cpu().numpy()
            
            loss_mask_np = loss_mask.cpu().numpy() # (B, N, T)
            
            for i in range(batch_size):
                # Determine number of valid agents in this sequence
                # loss_mask[i] is (N, T). Valid agents have non-zero entries (or we assume count from data loading)
                # Simple heuristic: Count rows that are not all zeros (or consistent with valid mask)
                # Since pad_mask appends zeros, we can just find the split point.
                # However, loss_mask might be all 1s for valid and 0s for pad.
                # Let's count agents with at least one valid time step.
                valid_rows = np.any(loss_mask_np[i] > 0, axis=1)
                num_valid = np.sum(valid_rows)
                
                if num_valid == 0: num_valid = 1 # Fallback
                
                # Extract valid part and only (x, y) coords
                # V_pred is (B, T, N, 5). We take first 2 channels.
                p_i = V_pred_np[i, :, :num_valid, :2] # (T, num_valid, 2)
                t_i = V_tr_np[i, :, :num_valid, :2]   # (T, num_valid, 2)
                
                pred_list.append(p_i)
                target_list.append(t_i)
                count_list.append(num_valid)

            ade_ls.append(ade(pred_list, target_list, count_list))
            fde_ls.append(fde(pred_list, target_list, count_list))

    return np.mean(ade_ls), np.mean(fde_ls)

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
             # ReduceLROnPlateau requires a metric (validation loss usually)
            #current_val_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
            # base on train loss lr scheduler must work
            current_train_loss = metrics['train_loss'][-1] if len(metrics['train_loss']) > 0 else float('inf')
    

            # Step with validation loss
            # scheduler.step(current_val_loss)

            # Step with training loss
            scheduler.step(current_train_loss)

            print(f"Learning Rate after Epoch {epoch}: {optimizer.param_groups[0]['lr']}")
            # Old manual check (ReduceOnPlateau handles NaN internally usually, but safe to keep eye on logs)
            if len(metrics['train_loss']) > 0 and np.isnan(metrics['train_loss'][-1]):
                print("NaN loss detected.")

    # Console Log
    print(f'Epoch: {epoch} | Train Loss: {metrics["train_loss"][-1]:.4f} | Val Loss: {metrics["val_loss"][-1]:.4f}')

    # Checkpoints
    # Create checkpoint dict with metadata
    checkpoint = {
        'state_dict': model.state_dict(),
        'scene_name': args.scene_name,
        'args': args,
        'epoch': epoch,
        'metrics': metrics
    }
    
    torch.save(checkpoint, os.path.join(checkpoint_dir, f'model_epoch{epoch}.pth'))
    
    curr_val_loss = metrics['val_loss'][-1] if len(metrics['val_loss']) > 0 else float('inf')
    if curr_val_loss < best_val_loss:
        best_val_loss = curr_val_loss
        best_model_state = model.state_dict() # Keep distinct logical copy if needed, or just use checkpoint
        # Save best model with same metadata structure
        torch.save(checkpoint, os.path.join(checkpoint_dir, 'best_model.pth'))

    with open(os.path.join(checkpoint_dir, 'metrics.pkl'), 'wb') as fp:
        pickle.dump(metrics, fp)
    
    t_loss = metrics['train_loss'][-1] if metrics['train_loss'] else 0
    v_loss = metrics['val_loss'][-1] if metrics['val_loss'] else 0
    log_file.write(f"{epoch},{t_loss},{v_loss},0,0\n")

if not args.skip_val:
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    elif os.path.exists(os.path.join(checkpoint_dir, 'best_model.pth')):
        # Load checkpoint with weights_only=False because it contains args/Namespace
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), weights_only=False)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    ade_calc, fde_calc = calculate_ade_fde(model, loader_val)
    print(f"Final Best Model - ADE: {ade_calc:.4f}, FDE: {fde_calc:.4f}")
    log_file.write(f"FINAL,,,{ade_calc},{fde_calc}\n")

log_file.close()
