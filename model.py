from curses import meta
import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
import time
from matplotlib import pyplot as plt

class VSIE(nn.Module):
    # Visual Spatio-Temporal Interaction Encoder
    def __init__(self, in_feat, output_dim, th):
        super(VSIE, self).__init__()
        self.th = th
        self.in_feat = in_feat # NEW: Store in_feat dynamically
        self.encoder = nn.LSTM(in_feat, in_feat*2, batch_first=True)
        # --- STAGE 1: Social Attention (Agent to Agent) ---
        self.fc_q1 = nn.Linear(in_feat*2, in_feat*4)    # Q uses LSTM output
        self.fc_k1 = nn.Linear(in_feat, in_feat*4)      # K uses original input x
        self.fc_v1 = nn.Linear(in_feat, in_feat*4)      # V uses original input x

        # --- STAGE 2: Environmental Attention (Agent to Map) ---
        self.fc_q2 = nn.Linear(in_feat*4, in_feat*4)
        self.fc_k2 = nn.Linear(256, in_feat*4)

        self.fc_out = nn.Linear(in_feat*4, output_dim)

        # --- C-TAG CAPACITY FIX ---
        # 1. Widen Compressor: 2048 -> 256 (Was 32)
        #    This preserves 8x more visual detail from the ResNet map.
        self.compressor = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
    def extract_local_context(self, feature_map, agent_coords, img_w=512.0, img_h=512.0):
        batch_size, channels, h_dim, w_dim = feature_map.shape
        _, _, time_steps, num_nodes = agent_coords.shape
        
        # Permute to [Batch, Time, Nodes, Features]
        coords = agent_coords.permute(0, 2, 3, 1)
        
        # Extract only x and y coordinates (first 2 features)
        coords_xy = coords[..., :2]
        
        # Flatten for grid_sample: [Batch, Time*Nodes, 1, 2]
        flat_coords = coords_xy.reshape(batch_size, -1, 1, 2)
        
        # Normalize to [-1, 1] for grid_sample
        # (Assumes coordinates are already scaled to 0-512 in utils_by_scene.py)
        norm_coords = torch.zeros_like(flat_coords)
        norm_coords[..., 0] = 2 * (flat_coords[..., 0] / img_w) - 1 # X
        norm_coords[..., 1] = 2 * (flat_coords[..., 1] / img_h) - 1 # Y
        
        # Grid sample
        sampled = Func.grid_sample(feature_map, norm_coords, align_corners=False)
        
        # Reshape to [Batch, Time, Nodes, Channels]
        # Channels will automatically match the compressor output (256)
        local_context = sampled.squeeze(-1).permute(0, 2, 1).view(batch_size, time_steps, num_nodes, channels)
        return local_context

    def viz_threshold(self,x):
        fig,ax=plt.subplots(1,2,figsize=(6,6))
        x_range,y_range = list(range(x.shape[1])),list(range(x.shape[0]))
        ax[0].imshow(x.cpu().detach().numpy())
        ax[0].set_title('Original')
        ax[1].imshow(torch.where(x>self.th,x,torch.zeros_like(x)).cpu().detach().numpy())
        ax[1].set_title('After Threshold')
        plt.set_cmap('inferno')
        plt.colorbar(ax[0].imshow(x.cpu().detach().numpy()),ax=ax[0])
        plt.colorbar(ax[1].imshow(torch.where(x>self.th,x,torch.zeros_like(x)).cpu().detach().numpy()),ax=ax[1])
        plt.suptitle('Threshold: {}'.format(self.th))
        plt.show()

    def threshold_relu(self, x, threshold, num_nodes):
        # Masking logic to preserve graph structure
        mask = (x > threshold).float()
        return x * mask

    def positional_encoding(self, x):
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, in_feat = x.size()
        elif x.dim() == 3:
            seq_len, num_nodes, in_feat = x.size()
        else:
            return x

        pos_enc = torch.zeros((seq_len, in_feat), device=x.device)
        div_term = torch.exp(torch.arange(0, in_feat, 2, dtype=torch.float, device=x.device) *
                             -(math.log(10000.0) / in_feat))

        position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
        
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        if in_feat > 1:
            pos_enc[:, 1::2] = torch.cos(position * div_term)

        if x.dim() == 4:
            pos_enc = pos_enc.unsqueeze(0).unsqueeze(2)
        else:
             pos_enc = pos_enc.unsqueeze(1)
             
        return x + pos_enc

    def forward(self, x, abs_coords, metadata):
        # Input x is [Batch, Channel, Time, Nodes]
        x_input_coords = x.clone() 
        x_original = x.shape 

        # Always ensure [B, T, V, C] structure regardless of input channel count
        if x.dim() == 4:
            # Check if likely in [B, C, T, V] format
            # We assume C is smaller than T or V typically, or match self.in_feat logic
            # If x.size(1) is small (channels), permute.
            if x.size(1) == self.in_feat or x.size(1) in [2, 4]: 
                 x = x.permute(0, 2, 3, 1) # [B, T, V, C]

        # Auto-pad features if input is 2D (dx, dy) but model expects 4D (dx, dy, v, theta)
        if x.size(-1) == 2 and self.in_feat == 4:
            zeros = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 2).to(x.device)
            x = torch.cat([x, zeros], dim=-1)
        
        x = self.positional_encoding(x)
        
        b, t, n, c_in = x.size()
        x_reshaped = x.contiguous().view(-1, c_in) 

        
        X_lstm, _ = self.encoder(x_reshaped.unsqueeze(1)) 
        X = X_lstm.squeeze(1) # [B*T*V, C*2]

        # ==========================================
        # STAGE 1: SOCIAL SELF-ATTENTION
        # ==========================================
        Q1 = self.fc_q1(X)
        K1 = self.fc_k1(x_reshaped)  # use original x for K
        v1 = self.fc_v1(x_reshaped)  # use original x for V

        q_dim = Q1.shape[-1]
        Q1_batched = Q1.view(b, t * n, -1)
        K1_batched = K1.view(b, t * n, -1)
        v1_batched = v1.view(b, t * n, -1)

        attn_scores_1 = torch.bmm(Q1_batched, K1_batched.transpose(1, 2)) / math.sqrt(q_dim)
        attn_probs_1 = Func.sigmoid(attn_scores_1)
        out1_batched = torch.bmm(attn_probs_1, v1_batched)
        
        out1 = out1_batched.view(-1, q_dim)
        
        # Apply the C-TAG threshold to isolate important agents
        out1_thresholded = self.threshold_relu(out1, self.th, x_original[3])

        # ==========================================
        # STAGE 2: ENVIRONMENTAL CROSS-ATTENTION
        # ==========================================
        if metadata is not None:
            batch_size = x_original[0]
            if metadata.size(0) != batch_size:
                visual_map = metadata.expand(batch_size, -1, -1, -1)
            else:
                visual_map = metadata
                
            compressed_map = self.compressor(visual_map) 
            local_context = self.extract_local_context(compressed_map, abs_coords)
            local_context_flat = local_context.reshape(X.shape[0], -1)
            
            # Query is the important agents
            Q2 = self.fc_q2(out1_thresholded)
            # Key is the visual map
            K2 = self.fc_k2(local_context_flat)
            
            Q2_batched = Q2.view(b, t * n, -1)
            K2_batched = K2.view(b, t * n, -1)
            
            # Value remains the original trajectory motion
            v2_batched = v1_batched

            attn_scores_2 = torch.bmm(Q2_batched, K2_batched.transpose(1, 2)) / math.sqrt(q_dim)
            attn_probs_2 = Func.sigmoid(attn_scores_2)
            final_out_batched = torch.bmm(attn_probs_2, v2_batched)
            
            final_out = final_out_batched.view(-1, q_dim)
        else:
            final_out = out1_thresholded

        # Pass the final refined features to the output projection
        out = self.fc_out(final_out)
        
        out = out.view(b, t, n, -1) 
        out = out.permute(0, 3, 1, 2)
        
        return out
    
class TemporalTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, seq_len, pred_seq_len, d_model=128, nhead=4, num_layers=4):
        super(TemporalTransformer, self).__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len

        # Feature Projection
        self.input_proj = nn.Linear(in_channels, d_model)
        
        # Positional Encoding
        self.pos_encoder = nn.Parameter(torch.zeros(1, seq_len, d_model))
        # Initialize pos encoding
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.pos_encoder.data[0, :, 0::2] = torch.sin(position * div_term)
        self.pos_encoder.data[0, :, 1::2] = torch.cos(position * div_term)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=0.4, batch_first=True) # higher dropout to reduce the overconfidance of transformer
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

        # Output Projection (Flatten T -> Linear -> PredT)
        self.flatten_dim = seq_len * d_model
        # Use a MLP to map from flattened input sequence to flattened output sequence
        self.output_proj = nn.Sequential(
            nn.Linear(self.flatten_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, pred_seq_len * out_channels) # Uses out_channels (5)
        )
        self.out_channels = out_channels # Store for reshape

    def forward(self, x):
        # x input shape: (N, C, T, V) from GCN
        N, C, T, V = x.shape
        
        # Permute to treat each node as a sequence: (N*V, T, C)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(N * V, T, C)
        
        # Project to d_model
        x = self.input_proj(x) # (HV, T, d_model)
        
        # Add Positional Encoding
        x = x + self.pos_encoder
        
        # Transformer Pass
        x = self.transformer_encoder(x) # (HV, T, d_model)
        
        # Flatten time dim
        x = x.reshape(N * V, -1) # (HV, T*d_model)
        
        # Project to Output
        out = self.output_proj(x) # (HV, PredT*C)
        
        # Reshape to (N, C, PredT, V) to match expected output
        out = out.view(N, V, self.pred_seq_len, self.out_channels) # Uses 5
        out = out.permute(0, 3, 2, 1).contiguous() 
        return out


class CTAG(nn.Module):
    def __init__(self, threshold, n_gcnn=1, n_tcnn=1, input_feat=4, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3, hidden_size=64):
        super(CTAG, self).__init__()
        self.vsie = VSIE(input_feat, hidden_size, threshold)
        self.n_gcnn= n_gcnn
        self.n_tcnn = n_tcnn
        
        # REPLACEMENT: Transformer for Temporal Pattern Extraction
        # We reuse n_tcnn to scale the transformer (e.g. layers)
        # Using d_model=128 to ensure capacity for SDD patterns
        
        # 3. Transformer takes 64 channels in, projects to 5 out
        self.temporal_transformer = TemporalTransformer(
            in_channels=hidden_size,    # Input is 64
            out_channels=output_feat,   # [FIX] Added this argument (5)
            seq_len=seq_len,
            pred_seq_len=pred_seq_len,
            d_model=128,
            nhead=4,
            num_layers=max(2, n_tcnn)
        )
            
        self.prelus = nn.ModuleList()
        
        # Legacy: Keeping prelus definition if needed to avoid breaking state_dict loading (though logic changes)
        # But for new model structure, we don't use them. 
        # Since user asked to modify model.py, we can change architecture.
        # self.tpcnns = nn.ModuleList() ... (Removed)
            
        
        
    def forward(self, v, a, abs_coords, metadata=None):
        # print("CTAG Forward Pass - Metadata Received:", metadata is not None)
        assert metadata is not None, "Metadata is required for CTAG model"

        # Handle Batch Processing of Metadata
        maps_list = []
        
        # If metadata is a list/tuple (from batching), iterate
        if isinstance(metadata, (list, tuple)):
            meta_batch = metadata
        else:
            meta_batch = [metadata] # Handle single item (Batch=1)
            
        # Persistent cache to store unique maps loaded across batches
        if not hasattr(self, 'map_cache'):
            self.map_cache = {}
            
        for meta_item in meta_batch:
            # Logic to parse filename from metadata item
            # Revert to simple string conversion if it's just the filename string
            pt_filename = str(meta_item)
            
            # Check if we already loaded this map
            if pt_filename not in self.map_cache:
                map_path = os.path.join('./processed/maps', pt_filename)
                
                if not os.path.exists(map_path):
                     # Fallback logic if needed, or raise cleaner error
                     raise FileNotFoundError(f"Visual Context Map not found: {map_path}")
                
                # Load map [C, H, W]
                # Load from disk ONLY if it's not in our cache
                single_map = torch.load(map_path, map_location=v.device)
                if single_map.dim() == 4:
                    single_map = single_map.squeeze(0)
                
                # Save to cache for subsequent agents/batches
                self.map_cache[pt_filename] = single_map
                
            # Append the tensor directly from memory
            maps_list.append(self.map_cache[pt_filename])
            
        # Stack into [Batch, C, H, W]
        # v is [Batch, 2, Time, Nodes] (input from train.py)
        # We need map_tensor to be [Batch, C, H, W] to match v's Batch dim
        map_tensor = torch.stack(maps_list, dim=0)
        # adding random noise to the map tensor to prevent overfitting and encourage generalization
        if self.training:
            # Create random noise with a standard deviation of 0.1
            noise = torch.randn_like(map_tensor) * 0.1 
            map_tensor = map_tensor + noise
            
            # Use functional dropout2d (No object instantiation)
            map_tensor = Func.dropout2d(map_tensor, p=0.2, training=self.training)
        # Pass to VSIE (Compressor inside VSIE will attach grad)
        v = self.vsie(v, abs_coords, map_tensor) 

        # Transfomer Temporal Extraction
        # v output from GCN is (N, C, T, V)
        # Passed directly to TemporalTransformer
        v = self.temporal_transformer(v)
        
        # Output is (N, C, PredT, V), matching CTAG expectation
        
        # Ensure we return a matching 'a' (adjacency) effectively unchanged or just updated graph state
        return v, a
