"""
utils_by_scene_eth.py

Description: 
    Utilities for trajectory dataset loading, preprocessing, and batching.
    Tailored for ETH dataset (seq_eth, seq_hotel) processing.

"""

import os
import math
import sys
import pickle
import glob

from PIL import Image
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from tqdm import tqdm

def anorm(p1, p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    if NORM == 0:
        return 0
    return 1 / (NORM)

def normalize_adj_dense(mx):
    mx = mx + np.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def poly_fit(traj, pred_len, threshold):
    t = np.arange(traj.shape[1])
    res_x = np.polyfit(t, traj[0, :], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, :], 2, full=True)[1]
    if len(res_x) == 0:
        res_x = 0.0
    else:
        res_x = res_x[0]
    if len(res_y) == 0:
        res_y = 0.0
    else:
        res_y = res_y[0]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def read_file(file_path, delim=None):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line_data = []
            for i in line:
                try:
                    line_data.append(float(i))
                except ValueError:
                    pass
            data.append(line_data)
    return np.asarray(data)

def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    if torch.is_tensor(seq_):
        seq_np = seq_.detach().cpu().numpy()
        seq_rel_np = seq_rel.detach().cpu().numpy()
    else:
        seq_np = seq_
        seq_rel_np = seq_rel

    if seq_np.ndim == 2:
        seq_np = seq_np[np.newaxis, :, :]
        seq_rel_np = seq_rel_np[np.newaxis, :, :]

    num_nodes = seq_np.shape[0]
    seq_len = seq_np.shape[2]
    num_features = seq_rel_np.shape[1]

    V = np.zeros((seq_len, num_nodes, num_features))
    A = np.zeros((seq_len, num_nodes, num_nodes))

    for s in range(seq_len):
        V[s, :, :] = seq_rel_np[:, :, s]
        pos_s = seq_np[:, :, s]
        
        diff = pos_s[:, np.newaxis, :] - pos_s[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)

        with np.errstate(divide='ignore', invalid='ignore'):
            adj_mat = np.zeros_like(dists)
            mask = dists != 0
            adj_mat[mask] = 1.0 / dists[mask]
        
        np.fill_diagonal(adj_mat, 0)

        if norm_lap_matr:
            A[s, :, :] = normalize_adj_dense(adj_mat)
        else:
            A[s, :, :] = adj_mat

    return torch.from_numpy(V).type(torch.float), torch.from_numpy(A).type(torch.float)

class TrajectoryDataset(Dataset):
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.2,
        min_ped=1, delim=None, norm_lap_matr=True, fill_missing=False, 
        shuffle=False, n_splits=5, dataset_name='eth', processed_dir='./processed',
        min_displacement=0.5, reload_data=False, target_set='all'): 
        
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir 
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len 
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr
        self.fill_missing = fill_missing
        self.shuffle = shuffle
        self.n_splits = n_splits
        self.dataset_name = dataset_name
        self.processed_dir = processed_dir
        self.min_ped = min_ped
        self.threshold = threshold
        self.min_displacement = min_displacement
        self.reload_data = reload_data
        self.target_set = target_set

        pkl_files = glob.glob(os.path.join(self.data_dir, "**", "*.pkl"), recursive=True)
        
        if len(pkl_files) > 0 and not self.reload_data:
            self._init_lazy_loading()
        else:
            print(f"No .pkl files found in {self.data_dir} or reload requested. Scanning for raw data to process...")
            self._process_raw_data()
            self._init_lazy_loading()

    def _process_raw_data(self):
        datasets = ['ETH', 'UCY']
        print(f"Processing scenes in ETH and UCY.")

        for dataset_name in datasets:
            dataset_path = os.path.join(self.data_dir, dataset_name)
            if not os.path.exists(dataset_path):
                continue
            
            scenes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
            
            for scene_name in scenes:
                current_scene_path = os.path.join(dataset_path, scene_name)
                txt_files = glob.glob(os.path.join(current_scene_path, '*.txt'))
                
                if not txt_files:
                    print(f"Warning: No .txt files found in {current_scene_path}.")
                    continue
                
                img_path = os.path.join(current_scene_path, 'bg.jpg')
                if not os.path.exists(img_path):
                    img_path = os.path.join(current_scene_path, 'bg.png')
                
                splits = ['train', 'val', 'test']
                for s_name in splits:
                    if self.target_set != 'all' and s_name != self.target_set:
                        continue
                    
                    split_out_dir = os.path.join(self.processed_dir, s_name, scene_name)
                    os.makedirs(split_out_dir, exist_ok=True)
                    
                    meta_id = f"{scene_name}_map.pt" 

                    print(f"Processing: {s_name} | {dataset_name} | {scene_name}")
                    # Process all txt files in the scene dir
                    for txt_file in txt_files:
                        base_txt = os.path.splitext(os.path.basename(txt_file))[0]
                        save_name = f"{scene_name}_{base_txt}_{s_name}.pkl"
                        save_path = os.path.join(split_out_dir, save_name)

                        if os.path.exists(save_path) and not self.reload_data:
                            continue

                        self._process_single_video(txt_file, meta_id, save_path, img_path, s_name)

    def _process_single_video(self, file_path, meta_id, save_path, img_path, split_name):
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
        else:
            print(f"Warning: No image found for {meta_id}, assuming 1.0 scale")
            orig_w, orig_h = 512, 512

        raw_data = read_file(file_path, self.delim)
        # Format: frame_id, agent_id, pos_x, pos_y
        data = raw_data[:, [0, 1, 2, 3]]
        data = data[data[:, 0].argsort()]
        frames = np.unique(data[:, 0]).tolist()
        
        num_frames = len(frames)
        n_train = int(num_frames * 0.7)
        n_test = int(num_frames * 0.2)
        
        train_frames = set(frames[:n_train])
        test_frames = set(frames[n_train:n_train+n_test])
        val_frames = set(frames[n_train+n_test:])
        
        valid_track_ids = []
        track_ids = np.unique(data[:, 1])
        for tid in track_ids:
            t_frames = data[data[:, 1] == tid, 0]
            if split_name == 'train' and all(f in train_frames for f in t_frames):
                valid_track_ids.append(tid)
            elif split_name == 'test' and all(f in test_frames for f in t_frames):
                valid_track_ids.append(tid)
            elif split_name == 'val' and all(f in val_frames for f in t_frames):
                valid_track_ids.append(tid)
                
        data = data[np.isin(data[:, 1], valid_track_ids)]
        
        if len(data) == 0:
            print(f"Skipping {split_name} for {file_path}: No full trajectories found.")
            return

        # Re-verify frames after filtering
        frames = np.unique(data[:, 0]).tolist()
        
        frame_data = []
        for frame in frames:
            frame_data.append(data[frame == data[:, 0], :])

        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped_list = []
        num_peds_in_seq = []
        seq_meta_list = []
        rot_angle_list = []
        
        graph_v_obs = []
        graph_a_obs = []
        graph_v_pred = []
        graph_a_pred = []

        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / self.skip))
        iterator = tqdm(range(0, num_sequences * self.skip + 1, self.skip), 
                       total=num_sequences, desc=f"Seqs", leave=False)

        for idx in iterator:
            if idx + self.seq_len > len(frame_data): break
            curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
            peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
            
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 4, self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
            curr_theta = np.zeros(len(peds_in_curr_seq)) 
            
            num_peds_considered = 0
            _non_linear_ped = []
            
            for _, obj_id in enumerate(peds_in_curr_seq):
                curr_obj_seq = curr_seq_data[curr_seq_data[:, 1] == obj_id, :]
                curr_obj_seq = np.around(curr_obj_seq, decimals=4)
                
                obj_front = frames.index(curr_obj_seq[0, 0]) - idx
                obj_end = frames.index(curr_obj_seq[-1, 0]) - idx + 1
                
                if obj_end - obj_front != self.seq_len: continue 
                if len(curr_obj_seq) != self.seq_len: continue

                curr_obj_seq = np.transpose(curr_obj_seq[:, 2:4]) 
                
                # --- STATIONARY AGENT FILTER ---
                # Changed threshold for ETH metric scale
                start_pos = curr_obj_seq[:, 0:1] 
                dists_from_start = np.linalg.norm(curr_obj_seq - start_pos, axis=0)
                max_displacement = np.max(dists_from_start)
                
                if max_displacement < self.min_displacement: 
                    continue 
                # -------------------------------

                # --- LINEAR AGENT FILTER ---
                t_steps = np.arange(self.seq_len)
                res_x = np.polyfit(t_steps, curr_obj_seq[0, :], 1, full=True)[1]
                res_y = np.polyfit(t_steps, curr_obj_seq[1, :], 1, full=True)[1]
                
                err_x = res_x[0] if len(res_x) > 0 else 0.0
                err_y = res_y[0] if len(res_y) > 0 else 0.0
                total_linear_error = err_x + err_y
                
                if total_linear_error < 0.05:  # Scaled down for meters
                    continue 
                # ---------------------------------
                
                # --- PHYSICS FILTER ---
                vel_vectors = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1]
                v1 = vel_vectors[:, :-1]
                v2 = vel_vectors[:, 1:]
                
                dot_products = np.sum(v1 * v2, axis=0)
                mag1 = np.linalg.norm(v1, axis=0)
                mag2 = np.linalg.norm(v2, axis=0)
                
                valid_mask = (mag1 > 1e-4) & (mag2 > 1e-4)
                
                if np.any(valid_mask):
                    cos_angles = dot_products[valid_mask] / (mag1[valid_mask] * mag2[valid_mask])
                    cos_angles = np.clip(cos_angles, -1.0, 1.0)
                    angles_deg = np.degrees(np.arccos(cos_angles))
                    
                    if np.max(angles_deg) > 80.0: 
                        continue 
                # -------------------------------------------

                # No Boundary filter since ETH isn't inherently bounded to 512x512 coordinate limits
               
                dx = curr_obj_seq[0, 1:] - curr_obj_seq[0, :-1]
                dy = curr_obj_seq[1, 1:] - curr_obj_seq[1, :-1]
                
                last_obs_idx = self.obs_len - 2
                if last_obs_idx < 0: last_obs_idx = 0
                
                last_dx = dx[last_obs_idx]
                last_dy = dy[last_obs_idx]
                
                theta = np.arctan2(last_dy, last_dx)
                
                cos_th = np.cos(-theta)
                sin_th = np.sin(-theta)
                
                rot_dx = dx * cos_th - dy * sin_th
                rot_dy = dx * sin_th + dy * cos_th
                
                rel_curr_obj_seq = np.zeros((4, self.seq_len))
                rel_curr_obj_seq[0, 1:] = rot_dx
                rel_curr_obj_seq[1, 1:] = rot_dy
                rel_curr_obj_seq[2, :] = np.cos(theta)
                rel_curr_obj_seq[3, :] = np.sin(theta)

                _idx = num_peds_considered
                curr_seq[_idx, :, obj_front:obj_end] = curr_obj_seq
                curr_seq_rel[_idx, :, obj_front:obj_end] = rel_curr_obj_seq
                curr_theta[_idx] = theta 
                
                _non_linear_ped.append(poly_fit(curr_obj_seq, self.pred_len, self.threshold))
                curr_loss_mask[_idx, obj_front:obj_end] = 1
                num_peds_considered += 1

            if num_peds_considered >= self.min_ped:
                non_linear_ped_list.append(np.array(_non_linear_ped))
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_meta_list.append((meta_id, orig_w, orig_h))
                rot_angle_list.append(curr_theta[:num_peds_considered]) 
                
                s_ = curr_seq[:num_peds_considered]
                s_rel_ = curr_seq_rel[:num_peds_considered]
                seq_list.append(s_)
                seq_list_rel.append(s_rel_)

                v_o, a_o = seq_to_graph(s_[:, :, :self.obs_len], s_rel_[:, :, :self.obs_len], self.norm_lap_matr)
                graph_v_obs.append(v_o.clone())
                graph_a_obs.append(a_o.clone())
                
                v_p, a_p = seq_to_graph(s_[:, :, self.obs_len:], s_rel_[:, :, self.obs_len:], self.norm_lap_matr)
                graph_v_pred.append(v_p.clone())
                graph_a_pred.append(a_p.clone())

        if len(seq_list) > 0:
            data_dict = {
                'obs_traj': torch.from_numpy(np.concatenate(seq_list, axis=0)[:, :, :self.obs_len]).type(torch.float),
                'pred_traj': torch.from_numpy(np.concatenate(seq_list, axis=0)[:, :, self.obs_len:]).type(torch.float),
                'obs_traj_rel': torch.from_numpy(np.concatenate(seq_list_rel, axis=0)[:, :, :self.obs_len]).type(torch.float),
                'pred_traj_rel': torch.from_numpy(np.concatenate(seq_list_rel, axis=0)[:, :, self.obs_len:]).type(torch.float),
                'loss_mask': torch.from_numpy(np.concatenate(loss_mask_list, axis=0)).type(torch.float),
                'non_linear_ped': torch.from_numpy(np.concatenate(non_linear_ped_list, axis=0)).type(torch.float),
                'num_peds_in_seq': num_peds_in_seq,
                'seq_meta': seq_meta_list,
                'v_obs': graph_v_obs,
                'A_obs': graph_a_obs,
                'v_pred': graph_v_pred,
                'A_pred': graph_a_pred,
                'theta': torch.from_numpy(np.concatenate(rot_angle_list, axis=0)).type(torch.float)
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Saved {save_path} with {len(seq_list)} sequences.")

    def _init_lazy_loading(self):
        search_path = os.path.join(self.data_dir, "**", "*.pkl")
        self.shard_paths = sorted(glob.glob(search_path, recursive=True))
        
        print(f"Found {len(self.shard_paths)} shards. Building index...")
        
        self.index_map = [] 
        self.num_seq = 0
        
        for file_idx, p_path in enumerate(tqdm(self.shard_paths, desc="Indexing")):
            with open(p_path, 'rb') as f:
                d = pickle.load(f)
                count = len(d['num_peds_in_seq'])
                for i in range(count):
                    self.index_map.append((file_idx, i))
                self.num_seq += count
        
        print(f"Total sequences indexed: {self.num_seq}")
        self.current_file_idx = -1
        self.current_data = None

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        file_idx, local_idx = self.index_map[index]
        
        if self.current_file_idx != file_idx:
            with open(self.shard_paths[file_idx], 'rb') as f:
                self.current_data = pickle.load(f)
            self.current_file_idx = file_idx
            self.cum_start_idx = [0] + np.cumsum(self.current_data['num_peds_in_seq']).tolist()

        d = self.current_data
        start = self.cum_start_idx[local_idx]
        end = self.cum_start_idx[local_idx+1]
        
        if 'theta' in d:
            theta = d['theta'][start:end].clone()
        else:
            num_peds = end - start
            theta = torch.zeros(num_peds) 

        out = [
            d['obs_traj'][start:end, :].clone(),
            d['pred_traj'][start:end, :].clone(),
            d['obs_traj_rel'][start:end, :].clone(),
            d['pred_traj_rel'][start:end, :].clone(),
            d['non_linear_ped'][start:end].clone(),
            d['loss_mask'][start:end, :].clone(),
            d['v_obs'][local_idx].clone(),
            d['A_obs'][local_idx].clone(),
            d['v_pred'][local_idx].clone(),
            d['A_pred'][local_idx].clone(),
            d['seq_meta'][local_idx],
            theta 
        ]
        return out

    @staticmethod
    def collate_fn(batch):
        batch_list = list(zip(*batch))
        
        num_peds_list = [item[0].shape[0] for item in batch]
        max_peds = max(num_peds_list)
        
        new_batch = []
        for i, (obs, pred, obs_rel, pred_rel, nl, mask, v_o, a_o, v_p, a_p, meta, th) in enumerate(batch):
            num_peds = obs.shape[0]
            pad_peds = max_peds - num_peds
            
            if pad_peds > 0:
                obs = torch.cat([obs, torch.zeros(pad_peds, obs.shape[1], obs.shape[2]).type_as(obs)], dim=0)
                pred = torch.cat([pred, torch.zeros(pad_peds, pred.shape[1], pred.shape[2]).type_as(pred)], dim=0)
                obs_rel = torch.cat([obs_rel, torch.zeros(pad_peds, obs_rel.shape[1], obs_rel.shape[2]).type_as(obs_rel)], dim=0)
                pred_rel = torch.cat([pred_rel, torch.zeros(pad_peds, pred_rel.shape[1], pred_rel.shape[2]).type_as(pred_rel)], dim=0)
                nl = torch.cat([nl, torch.zeros(pad_peds).type_as(nl)], dim=0)
                mask = torch.cat([mask, torch.zeros(pad_peds, mask.shape[1]).type_as(mask)], dim=0)
                th = torch.cat([th, torch.zeros(pad_peds).type_as(th)], dim=0)
                
                v_o = torch.cat([v_o, torch.zeros(v_o.shape[0], pad_peds, v_o.shape[2]).type_as(v_o)], dim=1)
                v_p = torch.cat([v_p, torch.zeros(v_p.shape[0], pad_peds, v_p.shape[2]).type_as(v_p)], dim=1)
                
                a_o = torch.cat([a_o, torch.zeros(a_o.shape[0], pad_peds, a_o.shape[2]).type_as(a_o)], dim=1)
                a_o = torch.cat([a_o, torch.zeros(a_o.shape[0], a_o.shape[1], pad_peds).type_as(a_o)], dim=2)
                
                a_p = torch.cat([a_p, torch.zeros(a_p.shape[0], pad_peds, a_p.shape[2]).type_as(a_p)], dim=1)
                a_p = torch.cat([a_p, torch.zeros(a_p.shape[0], a_p.shape[1], pad_peds).type_as(a_p)], dim=2)
            
            new_batch.append((obs, pred, obs_rel, pred_rel, nl, mask, v_o, a_o, v_p, a_p, meta, th))
            
        batch_list = list(zip(*new_batch))
        
        return [
            torch.stack(batch_list[0]), 
            torch.stack(batch_list[1]), 
            torch.stack(batch_list[2]), 
            torch.stack(batch_list[3]), 
            torch.stack(batch_list[4]), 
            torch.stack(batch_list[5]), 
            torch.stack(batch_list[6]), 
            torch.stack(batch_list[7]), 
            torch.stack(batch_list[8]), 
            torch.stack(batch_list[9]), 
            torch.stack(batch_list[11]),
            list(batch_list[10])        
        ]

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Process ETH trajectory dataset and generate .pkl files.")
    parser.add_argument('--dataset_root', type=str, default='./eth', help="Root directory of the eth dataset (e.g. where ETH and UCY folders are)")
    parser.add_argument('--processed_dir', type=str, default='./processed', help="Directory to save the processed files")
    parser.add_argument('--set', type=str, default='all', choices=['all', 'train', 'val', 'test'], help="Which split name to append (though eth processing isn't split by video)")
    parser.add_argument('--obs_len', type=int, default=8, help="Observation length")
    parser.add_argument('--pred_len', type=int, default=12, help="Prediction length")
    parser.add_argument('--reload_data', action='store_true', help="Force re-generation of files")
    
    args = parser.parse_args()
    
    dataset = TrajectoryDataset(
        data_dir=args.dataset_root,
        processed_dir=args.processed_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        target_set=args.set,
        reload_data=args.reload_data,
        min_displacement=0.5 # Suitable for meter-scale data
    )
    print("Done!")
