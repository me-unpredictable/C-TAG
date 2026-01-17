import os
import math
import sys
import pickle

import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from torch.utils.data import Dataset
from tqdm import tqdm

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_, seq_rel, norm_lap_matr=True):
    '''
    Vectorized version of graph construction.
    '''
    # Ensure inputs are on CPU and numpy for vectorization
    if torch.is_tensor(seq_):
        seq_np = seq_.detach().cpu().numpy()
        seq_rel_np = seq_rel.detach().cpu().numpy()
    else:
        seq_np = seq_
        seq_rel_np = seq_rel

    # Handle shape consistency
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
        
        np.fill_diagonal(adj_mat, 1)

        if norm_lap_matr:
            G = nx.from_numpy_array(adj_mat)
            lap = nx.normalized_laplacian_matrix(G).toarray()
            A[s, :, :] = lap
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


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.2,
        min_ped=1, delim='\t', norm_lap_matr=True, fill_missing=False, 
        shuffle=False, n_splits=5, mk_splits=False, dataset_name='', processed_dir='./processed'):
        
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

        # -------------------------------------------------------------------------
        # 1. Loading Path: Try loading PKL from processed_dir
        # -------------------------------------------------------------------------
        # We try to load the specific split provided by 'dataset_name' logic or just generic load
        # Since the class is initialized usually with a specific intent, we can check 
        # if we are in training mode (not mk_splits) and if pkl exists.
        
        # However, typically this class is called with mk_splits=True once to generate data,
        # and then called with mk_splits=False to load it. 
        # When loading, we might need to know WHICH split to load, but usually the dataloader 
        # points to the specific split folder (e.g. data_dir/train).
        
        # Checking if data_dir contains a .pkl file directly (Standard C-TAG behavior)
        pkl_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        if not mk_splits and len(pkl_files) > 0:
            pkl_path = os.path.join(self.data_dir, pkl_files[0])
            print(f"Loading data from Pickle file: {pkl_path}")
            with open(pkl_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            self.obs_traj = saved_data['obs_traj'].type(torch.float)
            self.pred_traj = saved_data['pred_traj'].type(torch.float)
            self.obs_traj_rel = saved_data['obs_traj_rel'].type(torch.float)
            self.pred_traj_rel = saved_data['pred_traj_rel'].type(torch.float)
            self.loss_mask = saved_data['loss_mask'].type(torch.float)
            self.non_linear_ped = saved_data['non_linear_ped'].type(torch.float)
            self.num_peds_in_seq = saved_data['num_peds_in_seq']
            self.seq_meta = saved_data['seq_meta'] # Load Metadata
            
            self.v_obs = saved_data['v_obs']
            self.A_obs = saved_data['A_obs']
            self.v_pred = saved_data['v_pred']
            self.A_pred = saved_data['A_pred']
            
            cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
            self.seq_start_end = [
                (start, end)
                for start, end in zip(cum_start_idx, cum_start_idx[1:])
            ]
            self.num_seq = len(self.seq_start_end)
            return # Exit init after loading

        # -------------------------------------------------------------------------
        # 2. Processing Path: Read Raw Files -> Split -> Save
        # -------------------------------------------------------------------------
        
        if dataset_name.lower() in ['eth','hotel','univ','zara1','zara2']:
            self.delim = '\t'
        elif 'sdd' in dataset_name.lower():
            self.delim = ' '
        else:
            self.delim = delim

        if dataset_name.lower() == 'sdd':
            if not os.path.exists(os.path.join(data_dir,'annotations')):
                raise ValueError("For SDD dataset, data_dir must be the root directory containing 'annotations' folder.")
            
            scenes = os.listdir(os.path.join(data_dir, 'annotations'))
            scenes = [s for s in scenes if os.path.isdir(os.path.join(data_dir, 'annotations', s))]
            scenes.sort() 

            # Divide scenes
            val_scene = [scenes[-1]]
            scenes = scenes[:-1]
            
            if len(scenes) > 4:
                test_scene = scenes[-2:] # Returns list, no brackets needed
                train_scene = scenes[:-2]
            else:
                test_scene = [scenes[-1]] 
                train_scene = scenes[:-1]
            
            splits = {
                'train': train_scene,
                'val': val_scene,
                'test': test_scene
            }

            for s in splits:
                # Containers for current split
                seq_list = []
                seq_list_rel = []
                loss_mask_list = []
                non_linear_ped_list = []
                num_peds_in_seq = []
                seq_meta_list = [] # Metadata container
                
                graph_v_obs = []
                graph_a_obs = []
                graph_v_pred = []
                graph_a_pred = []

                scenes_in_split = splits[s]
                print(f"Processing {s} split with {len(scenes_in_split)} scenes: {scenes_in_split}")

                # Loop over ALL scenes in the split
                for scene_name in scenes_in_split:
                    current_scene_path = os.path.join(data_dir, 'annotations', scene_name)
                    if not os.path.isdir(current_scene_path): continue
                    
                    videos = os.listdir(current_scene_path)
                    
                    for v in videos:
                        path = os.path.join(current_scene_path, v, 'annotations.txt')
                        if not os.path.exists(path): continue
                        
                        # Metadata string for this video
                        meta_id = f"{scene_name}_{v}.pt"

                        print(f"Processing Split: {s} | Scene: {scene_name} | Video: {v}")

                        # Read File
                        if 'sdd' in self.dataset_name.lower():
                            raw_data = read_file(path, delim)
                            if raw_data.shape[1] >= 6: 
                                center_x = (raw_data[:, 1] + raw_data[:, 3]) / 2.0
                                center_y = (raw_data[:, 2] + raw_data[:, 4]) / 2.0
                                track_id = raw_data[:, 0]
                                frame_id = raw_data[:, 5]
                                data = np.stack((frame_id, track_id, center_x, center_y), axis=1)
                            else:
                                data = raw_data[:, :4]
                        else:
                            data = read_file(path, delim)
                        
                        data = data[data[:, 0].argsort()]
                        frames = np.unique(data[:, 0]).tolist()
                        frame_data = []
                        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
                        
                        for frame in frames:
                            frame_data.append(data[frame == data[:, 0], :])

                        seq_iterator = tqdm(range(0, num_sequences * self.skip + 1, skip), total=num_sequences)

                        for idx in seq_iterator:
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
                                
                                rel_curr_obj_seq = np.zeros(curr_obj_seq.shape)
                                rel_curr_obj_seq[:, 1:] = curr_obj_seq[:, 1:] - curr_obj_seq[:, :-1]
                                
                                _idx = num_peds_considered
                                curr_seq[_idx, :, obj_front:obj_end] = curr_obj_seq
                                curr_seq_rel[_idx, :, obj_front:obj_end] = rel_curr_obj_seq
                                
                                _non_linear_ped.append(poly_fit(curr_obj_seq, pred_len, threshold))
                                curr_loss_mask[_idx, obj_front:obj_end] = 1
                                num_peds_considered += 1

                            if num_peds_considered > min_ped:
                                non_linear_ped_list.append(np.array(_non_linear_ped))
                                num_peds_in_seq.append(num_peds_considered)
                                loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                                
                                # Store Metadata (1 per sequence)
                                seq_meta_list.append(meta_id)
                                
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

                # Concatenate and Save
                if len(seq_list) > 0:
                    seq_list = np.concatenate(seq_list, axis=0)
                    seq_list_rel = np.concatenate(seq_list_rel, axis=0)
                    loss_mask_list = np.concatenate(loss_mask_list, axis=0)
                    non_linear_ped = np.concatenate(non_linear_ped_list, axis=0)
                    
                    self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).type(torch.float)
                    self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).type(torch.float)
                    self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).type(torch.float)
                    self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).type(torch.float)
                    self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
                    self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
                    self.num_peds_in_seq = num_peds_in_seq
                    self.seq_meta = seq_meta_list # Store list of strings
                    
                    self.v_obs = graph_v_obs
                    self.A_obs = graph_a_obs
                    self.v_pred = graph_v_pred
                    self.A_pred = graph_a_pred
                    
                    cum_start_idx = [0] + np.cumsum(self.num_peds_in_seq).tolist()
                    self.seq_start_end = [
                        (start, end)
                        for start, end in zip(cum_start_idx, cum_start_idx[1:])
                    ]
                    self.num_seq = len(self.seq_start_end)

                    # Save PKL
                    split_path = os.path.join(self.processed_dir, s)
                    if not os.path.exists(split_path):
                        os.makedirs(split_path)
                    pkl_path = os.path.join(split_path, f"{s}.pkl")
                    
                    print(f"Saving processed data for {s} split to {pkl_path}...")
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(self.__dict__, f)
                else:
                    print(f"Warning: No sequences found for split {s}")

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index],
            self.seq_meta[index] # Return metadata
        ]
        return out

    def get_fold(self, fold_index):
        train_idx, val_idx = self.kfolds[fold_index]
        train_data = [self.__getitem__(i) for i in train_idx]
        val_data = [self.__getitem__(i) for i in val_idx]
        return train_data, val_data