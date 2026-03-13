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

def evaluate(model, loader, args, map_type='default'):
    model.eval()
    
    ade_list = []
    fde_list = []
    
    # Setup ablation settings logic
    use_no_map = (map_type == 'no_map')
    # Save the original extract_local_context to restore later
    original_extract = model.vsie.extract_local_context

    if use_no_map:
        # Monkey patch extract_local_context to return zero tensor
        def zero_extract(feature_map, agent_coords, img_w=512.0, img_h=512.0):
            batch_size, _, _, _ = feature_map.shape
            _, _, time_steps, num_nodes = agent_coords.shape
            # The channels output from compressor is 256
            channels = 256
            zeros = torch.zeros((batch_size, time_steps, num_nodes, channels), device=feature_map.device)
            return zeros
        
        model.vsie.extract_local_context = zero_extract

    print(f"\nStarting evaluation (Ablation mode: {map_type})...")
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
            
            # Here we rewrite batch_metadata to direct to bad_map or white_map
            # The model internally uses os.path.join('./processed/maps', pt_filename)
            # By passing '../bad_maps/filename', it resolves to './processed/bad_maps/filename'
            model_metadata = []
            for m in batch_metadata:
                orig_name = m[0]
                if map_type == 'bad_map':
                    model_metadata.append(os.path.join('..', 'bad_maps', orig_name))
                elif map_type == 'white_map':
                    model_metadata.append(os.path.join('..', 'white_maps', orig_name))
                else:
                    model_metadata.append(orig_name)
                    
            V_pred, _ = model(V_obs_tmp, A_obs, abs_coords, model_metadata)
            
            V_pred = V_pred.permute(0, 2, 3, 1) # [Batch, Time, Nodes, 5] 
            
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

            ade_list.append(ade(pred_list, target_list, count_list))
            fde_list.append(fde(pred_list, target_list, count_list))

    final_ade = np.mean(ade_list)
    final_fde = np.mean(fde_list)
    
    # Restore monkey patch
    model.vsie.extract_local_context = original_extract
    
    return final_ade, final_fde


def run_evaluation_for_model(model_path, args_path):
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
        return
        
    results_dir = os.path.join('test_bed')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    map_types = ['normal_map', 'bad_map', 'white_map', 'no_map']
    results = {}
    
    for mt in map_types:
        ade_val, fde_val = evaluate(model, loader_test, args, map_type=mt)
        results[mt] = (ade_val, fde_val)
    
    model_name = os.path.basename(os.path.dirname(model_path))
    scene_str = scene_name if scene_name else "All"
    
    print("\n" + "="*50)
    print(f"RESULTS FOR MODEL: {model_name}")
    print(f"SCENE: {scene_str}")
    print("="*50)
    print(f"| {'Map Type':<15} | {'ADE':<10} | {'FDE':<10} |")
    print("-" * 50)
    for mt in map_types:
        print(f"| {mt:<15} | {results[mt][0]:<10.4f} | {results[mt][1]:<10.4f} |")
    print("="*50 + "\n")
    
    res_file_md = os.path.join(results_dir, "ablation_results.md")
    write_header_md = not os.path.exists(res_file_md)
    
    with open(res_file_md, "a") as f:
        if write_header_md:
            f.write("# Ablation Study Results\n\n")
            f.write("| Scene | Model Name | Normal ADE | Normal FDE | Bad Map ADE | Bad Map FDE | White Map ADE | White Map FDE | Zero Map ADE | Zero Map FDE |\n")
            f.write("|-------|------------|------------|------------|-------------|-------------|---------------|---------------|--------------|--------------|\n")
        f.write(f"| {scene_str} | {model_name} | {results['normal_map'][0]:.4f} | {results['normal_map'][1]:.4f} | {results['bad_map'][0]:.4f} | {results['bad_map'][1]:.4f} | {results['white_map'][0]:.4f} | {results['white_map'][1]:.4f} | {results['no_map'][0]:.4f} | {results['no_map'][1]:.4f} |\n")
        
    res_file_csv = os.path.join(results_dir, "ablation_results.csv")
    write_header_csv = not os.path.exists(res_file_csv)
    
    with open(res_file_csv, "a") as f:
        if write_header_csv:
            f.write("Scene,Model Name,Normal_map ADE,Normal_map FDE,bad_map ADE,bad_map FDE,white_map ADE,white_map FDE,zero_map ADE,zero_map FDE\n")
        f.write(f"{scene_str},{model_name},{results['normal_map'][0]:.4f},{results['normal_map'][1]:.4f},{results['bad_map'][0]:.4f},{results['bad_map'][1]:.4f},{results['white_map'][0]:.4f},{results['white_map'][1]:.4f},{results['no_map'][0]:.4f},{results['no_map'][1]:.4f}\n")
        
    print(f"Results appended to {res_file_md} and {res_file_csv}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate C-TAG with Ablation settings.")
    parser.add_argument('--auto', action='store_true', help='Automatically run best models for all checkpoint dirs on all 4 map types')
    cmd_args = parser.parse_args()

    if cmd_args.auto:
        checkpoint_root = './checkpoint/'
        subdirs = [d for d in glob.glob(os.path.join(checkpoint_root, '*')) if os.path.isdir(d)]
        subdirs.sort()
        
        for exp_dir in subdirs:
            model_path = os.path.join(exp_dir, 'best_model.pth')
            args_path = os.path.join(exp_dir, 'args.pkl')
            
            if not os.path.exists(model_path):
                print(f"Skipping {exp_dir} (best_model.pth not found)")
                continue
            if not os.path.exists(args_path):
                print(f"Skipping {exp_dir} (args.pkl not found)")
                continue
                
            print(f"\n===========================")
            print(f"Running auto evaluation: Model={os.path.basename(exp_dir)}")
            print(f"===========================")
            run_evaluation_for_model(model_path, args_path)
                
    else:
        model_path, args_path = get_model_path()
        run_evaluation_for_model(model_path, args_path)

if __name__ == '__main__':
    main()
