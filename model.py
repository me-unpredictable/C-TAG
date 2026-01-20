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
        assert A.size(0) == self.kernel_size
        
        # Apply temporal convolution
        x = self.conv(x)
        
        # Graph convolution operation using batch matrix multiplication
        n, c, t, v = x.size()
        x = x.permute(0, 2, 1, 3)  # [n, t, c, v]
        x = x.reshape(n * t, c, v)  # [n*t, c, v]
        
        # Apply adjacency matrix A - we reshape and use batch matmul
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
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
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
                nn.BatchNorm2d(out_channels),
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
    def __init__(self,in_feat,th):
        super(VSIE,self).__init__()
        self.th=th
        self.encoder= nn.LSTM(in_feat,in_feat*2,batch_first=True)
        self.fc=nn.Linear(in_feat*2,in_feat*4)
        self.fc2=nn.Linear(in_feat,in_feat*4)
        self.fc3=nn.Linear(in_feat,in_feat*4)
        self.fc_out=nn.Linear(in_feat*4,in_feat)

        # C-TAG: Visual Context Layers
        # 1. Compressor: Reduce 2048 visual channels to 32
        self.compressor = nn.Conv2d(in_channels=2048, out_channels=32, kernel_size=1)
        # 2. Visual Fusion Layer: Fuses [LSTM_Encoded(in_feat*2) + Visual(32)] back to [LSTM_Encoded(in_feat*2)]
        self.visual_fusion = nn.Linear((in_feat*2) + 32, in_feat*2)

    def extract_local_context(self, feature_map, agent_coords, img_w=512.0, img_h=512.0):
        # Helper to sample features at agent locations.
        batch_size, channels, h_dim, w_dim = feature_map.shape
        _, _, time_steps, num_nodes = agent_coords.shape
        
        # Permute to [Batch, Time, Nodes, 2]
        coords = agent_coords.permute(0, 2, 3, 1)
        
        # Flatten for grid_sample: [Batch, Time*Nodes, 1, 2]
        flat_coords = coords.reshape(batch_size, -1, 1, 2)
        
        # Normalize to [-1, 1] for grid_sample
        norm_coords = torch.zeros_like(flat_coords)
        norm_coords[..., 0] = 2 * (flat_coords[..., 0] / img_w) - 1 # X
        norm_coords[..., 1] = 2 * (flat_coords[..., 1] / img_h) - 1 # Y
        
        # Grid sample
        sampled = Func.grid_sample(feature_map, norm_coords, align_corners=False)
        
        # Reshape to [Batch, Time, Nodes, 32]
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

    def threshold_relu(self,x,threshold,num_nodes):
        # Using torch.where keeps the graph connected unlike manual masking
        return torch.where(x>threshold,x,torch.zeros_like(x))

    def positional_encoding(self, x):
        # NOTE: Be careful with dimensions here. x is [Batch, 2, Seq, Nodes]
        # This implementation assumes specific batch behavior
        x_squeezed = x.squeeze(0) # Squeeze batch if 1
        # If x has 4 dims, this might fail unpacking if batch > 1. 
        # Assuming existing logic works for the data pipeline:
        if x.dim() == 4:
             # Standard C-TAG input [Batch, 2, Seq, Nodes]
             batch_size, in_feat, seq_len, num_nodes = x.shape
             # Need to reshape for encoding logic which expects (Batch, Seq, Feat)
             # But legacy code expects unpacking 3 vals.
             # We will stick to the logic that works for the provided input shape
             pass
        
        batch_size, seq_len, in_feat = x_squeezed.size()
        pos_enc = torch.zeros((seq_len, in_feat), device=x.device)
        for pos in range(seq_len):
            for i in range(0, in_feat, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / in_feat)))
                if i + 1 < in_feat:
                    pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / in_feat)))
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        return x_squeezed + pos_enc

    def forward(self,x,metadata):
        # Capture coords for VSIE extraction (clone input to avoid in-place issues)
        x_input_coords = x.clone() 
        x_original = x.shape 

        # Positional Encoding
        # Note: If your batch size > 1, ensure positional_encoding logic supports it.
        # Assuming batch=1 for now based on legacy code
        x= self.positional_encoding(x)

        x=x.reshape(-1,2)
        X,_=self.encoder(x) 

        # --- C-TAG FUSION LOGIC ---
        if metadata is not None:
            # metadata is the loaded map tensor [1, 2048, 16, 16]
            batch_size = x_original[0]
            
            # Expand map to batch size
            if metadata.size(0) != batch_size:
                visual_map = metadata.expand(batch_size, -1, -1, -1)
            else:
                visual_map = metadata
                
            # 1. Compress the map (Conv2d, keeps grad)
            compressed_map = self.compressor(visual_map)
            
            # 2. Extract Local Context (GridSample, keeps grad)
            local_context = self.extract_local_context(compressed_map, x_input_coords)
            
            # 3. Flatten and Fuse
            # X shape: [TotalSteps, Hidden]
            # local_context shape: [Batch, Seq, Nodes, 32]
            local_context_flat = local_context.view(X.shape[0], 32)
            
            # 4. Fuse (Linear, keeps grad)
            fused_features = torch.cat([X, local_context_flat], dim=-1)
            X = self.visual_fusion(fused_features)
            
        Q=self.fc(X) 
        K=self.fc2(x) 
        v=self.fc3(x) 
        
        out=Func.sigmoid(torch.matmul(Q,K.T))@v
        out=self.threshold_relu(out,self.th,x_original[3]) 
        out=self.fc_out(out)
        out=out.reshape(x_original)
        return out

class CTAG(nn.Module):
    def __init__(self,threshold,n_gcnn =1,n_tcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(CTAG,self).__init__()
        self.vsie = VSIE(input_feat,threshold) 
        self.n_gcnn= n_gcnn
        self.n_tcnn = n_tcnn
                
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1,self.n_gcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))

        if self.n_tcnn>1:
            self.tpcnns = nn.ModuleList()
            self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
            for j in range(1,self.n_tcnn):
                self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
            self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
        else:
            if self.n_tcnn < 1:
                self.n_tcnn = 1
            self.tpcnns = nn.ModuleList()
            self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
            self.tpcnn_ouput = nn.Identity()
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_tcnn):
            self.prelus.append(nn.PReLU())

    def forward(self,v,a,metadata=None):
        assert metadata is not None, "Metadata is required for CTAG model"

        # FIX FOR TUPLE/LIST METADATA
        pt_filename = None
        
        # If metadata is passed as a tuple/list from DataLoader (e.g. [('scene',), ('video',)])
        # if isinstance(metadata, (tuple, list)):
        #     try:
        #         # Extract first element if it's a batch of strings
        #         scene_id = metadata[0][0] if isinstance(metadata[0], (tuple, list)) else metadata[0]
        #         video_id = metadata[1][0] if isinstance(metadata[1], (tuple, list)) else metadata[1]
        #         pt_filename = f"{scene_id}_{video_id}_map.pt"
        #     except Exception:
        #         # Fallback
        pt_filename = str(metadata[0])
        pt_filename = pt_filename.split('.')[0] + '_map.pt'

        if pt_filename is None:
             raise ValueError(f"Could not parse metadata: {metadata}")

        # Load Map
        map_path = os.path.join('./processed/maps', pt_filename)
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Visual Context Map not found at: {map_path}")
            
        # Load Tensor
        # Note: torch.load returns a tensor with requires_grad=False.
        # This is correct. The COMPRESSOR weights (in VSIE) provide the gradients.
        map_tensor = torch.load(map_path, map_location=v.device)

        # Pass to VSIE
        v = self.vsie(v, map_tensor) 

        # ST-GCN Layers
        for k in range(self.n_gcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        
        if self.n_tcnn >= 1:
            v = self.prelus[0](self.tpcnns[0](v))
            for k in range(1, self.n_tcnn - 1):
                v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3]) 

        return v, a
