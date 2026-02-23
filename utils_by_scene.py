"""
utils.py

Description: 
    Utilities for trajectory dataset loading, preprocessing, and batching.
    Features:
    - Vectorized Graph Generation (Fast)
    - Padding Collation (Enables Batch_Size > 1)
    - Correct Metadata Naming (Matches map_utils.py)
    - Lazy Loading (Handles large datasets like SDD)
    - Dynamic Filtering of Stationary Agents (min_displacement)

Author: me__unpredictable (vishal patel) https://vishalresearch.com
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
    """
    Row-normalize dense matrix (Symmetrical Normalization for GCN).
    Formula: D^{-0.5} * A * D^{-0.5}
    """
    mx = mx + np.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx
                
def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    """
    Vectorized version of graph construction.
    seq_: [Num_Nodes, 2, Seq_Len]
    seq_rel: [Num_Nodes, 2, Seq_Len]
    """
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

    V = np.zeros((seq_len, num_nodes, 2))
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

def poly_fit(traj, traj_len, threshold):
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1] 
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1] 
    if res_x + res_y >= threshold: 
        return 1.0
    else:
        return 0.0

def read_file(_path, delim='\t'):
    data = []
    if delim == 'space':
        delim = ' ' 
    
    class_map = {
        '"Pedestrian"': 1, 'Pedestrian': 1,
        '"Biker"': 2, 'Biker': 2,
        '"Skater"': 3, 'Skater': 3,
        '"Car"': 4, 'Car': 4,
        '"Bus"': 5, 'Bus': 5,
        '"Cart"': 6, 'Cart': 6
    }

    def parse_and_append(file_obj, delimiter, data_list):
        for line in file_obj:
            line = line.strip()
            if not line: continue
            raw_tokens = line.split(delimiter)
            if delimiter == ' ':
                raw_tokens = [x for x in raw_tokens if x]
            parsed_line = []
            for i in raw_tokens:
                try:
                    parsed_line.append(float(i))
                except ValueError:
                    if i in class_map:
                        parsed_line.append(float(class_map[i]))
                    else:
                        parsed_line.append(0.0)
            if len(parsed_line) > 0:
                data_list.append(parsed_line)

    try:
        with open(_path, 'r', encoding='latin-1') as f:
            parse_and_append(f, delim, data)
        if len(data) > 0 and len(data[0]) < 2:
            raise ValueError("Likely wrong delimiter")    
    except:
        data = []
        if delim == '\t': delim = ' '
        else: delim = '\t'
        with open(_path, 'r', encoding='latin-1') as f:
            parse_and_append(f, delim, data)
            
    return np.asarray(data)

def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, pred_seq_rel_list,
     non_linear_ped_list, loss_mask_list, v_obs_list, A_obs_list,
     v_pred_list, A_pred_list, seq_meta_list) = zip(*data)

    max_agents = max([x.shape[1] for x in v_obs_list])
    
    def pad_N(tensor, N_target):
        N, D, T = tensor.shape
        pad_amt = N_target - N
        if pad_amt == 0: return tensor
        return torch.cat([tensor, torch.zeros(pad_amt, D, T)], dim=0)

    def pad_V(tensor, N_target):
        T, N, D = tensor.shape
        pad_amt = N_target - N
        if pad_amt == 0: return tensor
        return torch.cat([tensor, torch.zeros(T, pad_amt, D)], dim=1)

    def pad_A(tensor, N_target):
        T, N, _ = tensor.shape
        pad = N_target - N
        if pad == 0: return tensor
        tensor = torch.cat([tensor, torch.zeros(T, N, pad)], dim=2)
        tensor = torch.cat([tensor, torch.zeros(T, pad, N + pad)], dim=1)
        return tensor

    def pad_mask(tensor, N_target):
        N, T = tensor.shape
        pad = N_target - N
        if pad == 0: return tensor
        return torch.cat([tensor, torch.zeros(pad, T)], dim=0)
    
    obs_traj = torch.stack([pad_N(x, max_agents) for x in obs_seq_list])
    pred_traj = torch.stack([pad_N(x, max_agents) for x in pred_seq_list])
    obs_traj_rel = torch.stack([pad_N(x, max_agents) for x in obs_seq_rel_list])
    pred_traj_rel = torch.stack([pad_N(x, max_agents) for x in pred_seq_rel_list])
    
    v_obs = torch.stack([pad_V(x, max_agents) for x in v_obs_list])
    v_pred = torch.stack([pad_V(x, max_agents) for x in v_pred_list])
    A_obs = torch.stack([pad_A(x, max_agents) for x in A_obs_list])
    A_pred = torch.stack([pad_A(x, max_agents) for x in A_pred_list])
    
    loss_mask = torch.stack([pad_mask(x, max_agents) for x in loss_mask_list])
    non_linear_ped = torch.cat(non_linear_ped_list, dim=0)

    return tuple([
        obs_traj, pred_traj, obs_traj_rel, pred_traj_rel, non_linear_ped,
        loss_mask, v_obs, A_obs, v_pred, A_pred, seq_meta_list
    ])


class TrajectoryDataset(Dataset):
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.2,
        min_ped=1, delim='\t', norm_lap_matr=True, fill_missing=False, 
        shuffle=False, n_splits=5, dataset_name='', processed_dir='./processed',
        min_displacement=30.0): # ADDED THRESHOLD HERE
        
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
        self.min_displacement = min_displacement # STORED FOR USE

        pkl_files = glob.glob(os.path.join(self.data_dir, "**", "*.pkl"), recursive=True)
        
        if len(pkl_files) > 0:
            self._init_lazy_loading()
        else:
            print(f"No .pkl files found in {data_dir}. Scanning for raw data to process...")
            self._process_raw_data()
            self.num_seq = 0

    def _process_raw_data(self):
        if self.dataset_name.lower() in ['eth','hotel','univ','zara1','zara2']:
            self.delim = '\t'
        elif 'sdd' in self.dataset_name.lower():
            self.delim = ' '
        
        if self.dataset_name.lower() == 'sdd':
            if not os.path.exists(os.path.join(self.data_dir,'annotations')):
                raise ValueError("For SDD dataset, data_dir must be the root directory containing 'annotations' folder.")
            
            scenes = os.listdir(os.path.join(self.data_dir, 'annotations'))
            scenes = [s for s in scenes if os.path.isdir(os.path.join(self.data_dir, 'annotations', s))]
            scenes.sort() 
            
            print(f"Processing {len(scenes)} scenes with Intra-Scene Splitting (Videos split per scene).")

            for scene_name in scenes:
                current_scene_path = os.path.join(self.data_dir, 'annotations', scene_name)
                videos = os.listdir(current_scene_path)
                videos = [v for v in videos if os.path.isdir(os.path.join(current_scene_path, v))]
                videos.sort()
                
                num_videos = len(videos)
                
                if num_videos >= 3:
                    n_val = max(1, int(num_videos * 0.15))
                    n_test = max(1, int(num_videos * 0.15))
                    n_train = num_videos - n_val - n_test
                elif num_videos == 2:
                    n_train = 1
                    n_val = 0
                    n_test = 1
                else:
                    n_train = 1
                    n_val = 0
                    n_test = 0
                
                train_v = videos[:n_train]
                val_v = videos[n_train:n_train+n_val]
                test_v = videos[n_train+n_val:]
                
                splits = {'train': train_v, 'val': val_v, 'test': test_v}
                print(f"Scene: {scene_name} ({num_videos} videos) | Train: {len(train_v)}, Val: {len(val_v)}, Test: {len(test_v)}")

                for s_name in splits:
                    videos_in_split = splits[s_name]
                    split_out_dir = os.path.join(self.processed_dir, s_name, scene_name)
                    os.makedirs(split_out_dir, exist_ok=True)

                    for v in videos_in_split:
                        path = os.path.join(current_scene_path, v, 'annotations.txt')
                        if not os.path.exists(path): continue
                        
                        meta_id = f"{scene_name}_{v}_map.pt" 
                        save_name = f"{scene_name}_{v}.pkl"
                        save_path = os.path.join(split_out_dir, save_name)

                        img_path = os.path.join(current_scene_path, v, 'reference.jpg')
                        if not os.path.exists(img_path):
                            jpgs = glob.glob(os.path.join(current_scene_path, v, "*.jpg"))
                            if len(jpgs) > 0: img_path = jpgs[0]
                        
                        print(f"Processing: {s_name} | {scene_name} | {v}")
                        self._process_single_video(path, meta_id, save_path, img_path)

    def _process_single_video(self, file_path, meta_id, save_path, img_path):
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                orig_w, orig_h = img.size
        else:
            print(f"Warning: No image found for {meta_id}, assuming 1.0 scale")
            orig_w, orig_h = 512, 512

        scale_x = 512.0 / orig_w
        scale_y = 512.0 / orig_h

        if 'sdd' in self.dataset_name.lower():
            raw_data = read_file(file_path, self.delim)
            if raw_data.shape[1] >= 6: 
                center_x = (raw_data[:, 1] + raw_data[:, 3]) / 2.0
                center_y = (raw_data[:, 2] + raw_data[:, 4]) / 2.0
                track_id = raw_data[:, 0]
                frame_id = raw_data[:, 5]
                data = np.stack((frame_id, track_id, center_x, center_y), axis=1)
            else:
                data = raw_data[:, :4]
            
            data[:, 2] = data[:, 2] * scale_x
            data[:, 3] = data[:, 3] * scale_y
        else:
            data = read_file(file_path, self.delim)
        
        data = data[data[:, 0].argsort()]
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
            
            curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
            curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
            
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
                # Calculate maximum distance traveled from their starting point
                start_pos = curr_obj_seq[:, 0:1] # Shape (2, 1)
                dists_from_start = np.linalg.norm(curr_obj_seq - start_pos, axis=0)
                max_displacement = np.max(dists_from_start)
                
                # REVISED: Reduced from 30 to 15
                if max_displacement < 15.0:
                    continue # Skip this agent, they barely moved
                # -------------------------------

                # --- LINEAR AGENT FILTER ---
                # Fit a 1st-degree polynomial (straight line) to the entire trajectory
                t_steps = np.arange(self.seq_len)
                
                res_x = np.polyfit(t_steps, curr_obj_seq[0, :], 1, full=True)[1]
                res_y = np.polyfit(t_steps, curr_obj_seq[1, :], 1, full=True)[1]
                
                err_x = res_x[0] if len(res_x) > 0 else 0.0
                err_y = res_y[0] if len(res_y) > 0 else 0.0
                total_linear_error = err_x + err_y
                
                # REVISED: Reduced from 20 to 5
                # Only drop agents with LESS than 5 pixel deviation from a perfect line
                if total_linear_error < 5.0:  
                    continue # Skip this agent, they are walking in a perfect straight line
                # ---------------------------------
                 # --- PHYSICS FILTER (Tracking Artifacts) ---
                # Calculate step-by-step velocity vectors
                vel_vectors = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1] # Shape: (2, seq_len-1)
                
                v1 = vel_vectors[:, :-1]
                v2 = vel_vectors[:, 1:]
                
                # Calculate dot product and magnitudes
                dot_products = np.sum(v1 * v2, axis=0)
                mag1 = np.linalg.norm(v1, axis=0)
                mag2 = np.linalg.norm(v2, axis=0)
                
                # Avoid division by zero (ignore frames where agent is temporarily stopped)
                valid_mask = (mag1 > 1e-4) & (mag2 > 1e-4)
                
                if np.any(valid_mask):
                    cos_angles = dot_products[valid_mask] / (mag1[valid_mask] * mag2[valid_mask])
                    # Clip to [-1, 1] to avoid arccos nan due to float precision
                    cos_angles = np.clip(cos_angles, -1.0, 1.0)
                    angles_deg = np.degrees(np.arccos(cos_angles))
                    
                    # If an agent instantly turns more than 120 degrees, it's an ID swap
                    if np.max(angles_deg) > 120.0:
                        continue # Skip this agent
                # -------------------------------------------

                # --- BOUNDARY FILTER (Edge Noise) ---
                # Remove agents that get too close to the image edges (e.g., 30 original pixels)
                margin_x = 30.0 * scale_x
                margin_y = 30.0 * scale_y
                
                min_x, max_x = np.min(curr_obj_seq[0, :]), np.max(curr_obj_seq[0, :])
                min_y, max_y = np.min(curr_obj_seq[1, :]), np.max(curr_obj_seq[1, :])
                
                if (min_x < margin_x) or (max_x > 512.0 - margin_x) or \
                   (min_y < margin_y) or (max_y > 512.0 - margin_y):
                    continue # Skip this agent, they are touching the edge noise zone
                # ------------------------------------
               
                rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1]
                
                _idx = num_peds_considered
                curr_seq[_idx, :, obj_front:obj_end] = curr_obj_seq
                curr_seq_rel[_idx, :, obj_front:obj_end] = rel_curr_obj_seq
                
                _non_linear_ped.append(poly_fit(curr_obj_seq, self.pred_len, self.threshold))
                curr_loss_mask[_idx, obj_front:obj_end] = 1
                num_peds_considered += 1

            if num_peds_considered >= self.min_ped:
                non_linear_ped_list.append(np.array(_non_linear_ped))
                num_peds_in_seq.append(num_peds_considered)
                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                seq_meta_list.append((meta_id, orig_w, orig_h))
                
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
                'A_pred': graph_a_pred
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
            d['seq_meta'][local_idx]
        ]
        return out

    @staticmethod
    def collate_fn(batch):
        return seq_collate(batch)