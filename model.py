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

class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        # Apply temporal convolution
        x = self.conv(x)
        
        # Graph convolution operation using batch matrix multiplication
        n, c, t, v = x.size()
        x = x.permute(0, 2, 1, 3)  # [n, t, c, v]
        x = x.reshape(n * t, c, v)  # [n*t, c, v]
        
        # Apply adjacency matrix A - we reshape and use batch matmul
        if A.dim() == 4:
            x = torch.matmul(x, A.view(n * t, v, v))
        else:
             assert A.size(0) == self.kernel_size
             x = torch.matmul(x, A.view(self.kernel_size, v, v))  # [n*t, c, v]
        
        # Reshape back to original format
        x = x.view(n, t, c, v)
        x = x.permute(0, 2, 1, 3)  # [n, c, t, v]
        
        return x.contiguous(), A

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
        assert len(kernel_size) == 2
        if kernel_size[0] % 2 == 1:
            padding = ((kernel_size[0] - 1) // 2, 0)
        else:
            padding = (kernel_size[0] // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.GroupNorm(1, out_channels), # Replaced BatchNorm2d with GroupNorm(1, C) -> LayerNorm behavior
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.GroupNorm(1, out_channels), # Replaced BatchNorm2d
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.GroupNorm(1, out_channels), # Replaced BatchNorm2d
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x) # Apply Convolution on the input
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class VSIE(nn.Module):
    # Visual Spatio-Temporal Interaction Encoder
    def __init__(self, in_feat, output_dim, th):
        super(VSIE, self).__init__()
        self.th = th
        self.encoder = nn.LSTM(in_feat, in_feat*2, batch_first=True)
        self.fc = nn.Linear(in_feat*2, in_feat*4)
        self.fc2 = nn.Linear(in_feat, in_feat*4)
        self.fc3 = nn.Linear(in_feat, in_feat*4)
        self.fc_out = nn.Linear(in_feat*4, output_dim)

        # --- C-TAG CAPACITY FIX ---
        # 1. Widen Compressor: 2048 -> 256 (Was 32)
        #    This preserves 8x more visual detail from the ResNet map.
        self.compressor = nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=1)
        
        # 2. Widen Fusion Layer: Input is Motion(in_feat*2) + Visual(256)
        self.visual_fusion = nn.Linear((in_feat*2) + 256, in_feat*2)

    def extract_local_context(self, feature_map, agent_coords, img_w=512.0, img_h=512.0):
        batch_size, channels, h_dim, w_dim = feature_map.shape
        _, _, time_steps, num_nodes = agent_coords.shape
        
        # Permute to [Batch, Time, Nodes, 2]
        coords = agent_coords.permute(0, 2, 3, 1)
        
        # Flatten for grid_sample: [Batch, Time*Nodes, 1, 2]
        flat_coords = coords.reshape(batch_size, -1, 1, 2)
        
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

        if x.dim() == 4 and x.size(1) == 2:
            x = x.permute(0, 2, 3, 1) # [B, T, V, C]
        
        x = self.positional_encoding(x)
        
        b, t, n, c_in = x.size()
        x_reshaped = x.contiguous().view(-1, c_in) 
        
        X_lstm, _ = self.encoder(x_reshaped.unsqueeze(1)) 
        X = X_lstm.squeeze(1) # [B*T*V, C*2]

        # --- C-TAG FUSION LOGIC ---
        if metadata is not None:
            batch_size = x_original[0]
            if metadata.size(0) != batch_size:
                visual_map = metadata.expand(batch_size, -1, -1, -1)
            else:
                visual_map = metadata
                
            compressed_map = self.compressor(visual_map) 
            
            # FIXED: Use abs_coords to sample the map, NOT relative x!
            local_context = self.extract_local_context(compressed_map, abs_coords)
            
            # 3. DYNAMIC RESHAPE (Critical Fix)
            # Use -1 so it automatically adapts to 256 (or any other size)
            local_context_flat = local_context.reshape(X.shape[0], -1)
            
            fused_features = torch.cat([X, local_context_flat], dim=-1)
            X = self.visual_fusion(fused_features)
            
        Q = self.fc(X)     
        K = self.fc2(x_reshaped) 
        v = self.fc3(x_reshaped) 
        
        # Batched Attention
        q_dim = Q.shape[-1]
        Q_batched = Q.view(b, t * n, -1) 
        K_batched = K.view(b, t * n, -1) 
        v_batched = v.view(b, t * n, -1) 
        
        attn_scores = torch.bmm(Q_batched, K_batched.transpose(1, 2))
        attn_probs = Func.sigmoid(attn_scores)
        out_batched = torch.bmm(attn_probs, v_batched)
        
        out = out_batched.view(-1, q_dim)
        
        out = self.threshold_relu(out, self.th, x_original[3]) 
        out = self.fc_out(out)
        
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
    def __init__(self, threshold, n_gcnn=1, n_tcnn=1, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3, hidden_size=64):
        super(CTAG, self).__init__()
        self.vsie = VSIE(input_feat, hidden_size, threshold)
        self.n_gcnn= n_gcnn
        self.n_tcnn = n_tcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(hidden_size, hidden_size, (kernel_size, seq_len)))
        for j in range(1, self.n_gcnn):
            self.st_gcns.append(st_gcn(hidden_size, hidden_size, (kernel_size, seq_len)))

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
            
        for meta_item in meta_batch:
            # Logic to parse filename from metadata item
            # Revert to simple string conversion if it's just the filename string
            pt_filename = str(meta_item)
            
            map_path = os.path.join('./processed/maps', pt_filename)
            
            if not os.path.exists(map_path):
                 # Fallback logic if needed, or raise cleaner error
                 raise FileNotFoundError(f"Visual Context Map not found: {map_path}")
            
            # Load map [C, H, W]
            single_map = torch.load(map_path, map_location=v.device)
            if single_map.dim() == 4:
                single_map = single_map.squeeze(0)
            maps_list.append(single_map)
            
        # Stack into [Batch, C, H, W]
        # v is [Batch, 2, Time, Nodes] (input from train.py)
        # We need map_tensor to be [Batch, C, H, W] to match v's Batch dim
        map_tensor = torch.stack(maps_list, dim=0)

        # Pass to VSIE (Compressor inside VSIE will attach grad)
        v = self.vsie(v, abs_coords, map_tensor) 
        # v = self.vsie(v, None) # run this to check if map has a bug
        for k in range(self.n_gcnn):
            v, a = self.st_gcns[k](v, a)

        # Transfomer Temporal Extraction
        # v output from GCN is (N, C, T, V)
        # Passed directly to TemporalTransformer
        v = self.temporal_transformer(v)
        
        # Output is (N, C, PredT, V), matching CTAG expectation
        
        # Ensure we return a matching 'a' (adjacency) effectively unchanged or just updated graph state
        return v, a
