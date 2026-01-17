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
        # assert kernel size matches adjacency matrix first dimension
        # print('A:',A.shape)
        # print('Kernel Size:',self.kernel_size)
        # if A.size(0) != self.kernel_size:
            # tirm the adjacency matrix to match kernel size
            # A = A[:self.kernel_size] # in the end the sequence length is higher than the kernel size
            # hence it needs to be trimmed
            # trim x to match kernel size
            # only trimming sequence length will not work, vector lenghth must match
            # trimmed connections needs to be reflected in the x by trimming vectors
            # x = x[:,:, :self.kernel_size,:]
        # print('A:',A.shape)
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
        
#         print("outstg",out_channels)
        
        assert len(kernel_size) == 2
        # print('Kernel Size:',kernel_size,kernel_size[0]%2)
        # assert kernel_size[0] % 2 == 1
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

class SIE(nn.Module):
    # Spatio-Temporal Interaction Encoder # meunpredictable
    def __init__(self,in_feat,th):
        super(SIE,self).__init__()
        self.th=th
        self.encoder= nn.LSTM(in_feat,in_feat*2,batch_first=True)
        # self.gru=nn.GRU(in_feat*2,in_feat*2,batch_first=True)
        self.fc=nn.Linear(in_feat*2,in_feat*4)
        self.fc2=nn.Linear(in_feat,in_feat*4)
        self.fc3=nn.Linear(in_feat,in_feat*4)
        self.fc_out=nn.Linear(in_feat*4,in_feat)
    def viz_threshold(self,x):
        # visualize original x and x after threshold in grid
        fig,ax=plt.subplots(1,2,figsize=(6,6))
        x_range,y_range = list(range(x.shape[1])),list(range(x.shape[0]))
        ax[0].imshow(x.cpu().detach().numpy())
        ax[0].set_title('Original')
        ax[0].set_xticks(x_range)
        ax[0].set_yticks(y_range)
        ax[1].set_xticks(x_range)
        ax[1].set_yticks(y_range)
        ax[1].imshow(torch.where(x>self.th,x,torch.zeros_like(x)).cpu().detach().numpy())
        ax[1].set_title('After Threshold')
        # show color bar (vivid colors)
        plt.set_cmap('inferno')
        plt.colorbar(ax[0].imshow(x.cpu().detach().numpy()),ax=ax[0])
        plt.colorbar(ax[1].imshow(torch.where(x>self.th,x,torch.zeros_like(x)).cpu().detach().numpy()),ax=ax[1])
        plt.suptitle('Threshold: {}'.format(self.th))
    
        plt.show()
    def threshold_relu(self,x,threshold,num_nodes):
        # print('Total objects:',num_nodes)
        # print('I got threshold:',threshold)
        # time.sleep(10)
        # print('X:',x.shape)
        # visualize original x and x after threshold in grid
        # self.viz_threshold(x)
        # time.sleep(1)
        # reshape to find objects
        x_obj=x.reshape(-1,num_nodes)
        # find index whith zero values
        torch_idx = torch.where(x_obj < threshold)
        # print('Torch Index:',torch.unique(torch_idx[0]))
        # print('Torch Index:',torch.unique(torch_idx[1]))
        return torch.where(x>threshold,x,torch.zeros_like(x)) # It changes the values of x to 0 if they are less than threshold
    def positional_encoding(self, x):
        # Assuming x is of shape (batch_size, seq_len, in_feat)
        # print('X:',x.shape)
        x=x.squeeze(0)
        batch_size, seq_len, in_feat = x.size()
        pos_enc = torch.zeros((seq_len, in_feat), device=x.device)
        for pos in range(seq_len):
            for i in range(0, in_feat, 2):
                pos_enc[pos, i] = math.sin(pos / (10000 ** ((2 * i) / in_feat)))
                if i + 1 < in_feat:
                    pos_enc[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / in_feat)))
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)
        return x + pos_enc

    def forward(self,x):
        x_original = x.shape # batch, 2dxy, seq_len, num_nodes
        x= self.positional_encoding(x)

        x=x.reshape(-1,2)
        # print('X:',x.shape)
        X,_=self.encoder(x) # Extract the temporal patterns from spatial data sequence over time
        # Q,_=self.gru(X) # Extract the temporal patterns from spatial data sequence over time
        Q=self.fc(X) # Translate spatial data sequence to a higher dimension
        K=self.fc2(x) # Translate spatial data sequence to a higher dimension
        # print('Q:',Q.shape)
        # print('K:',K.shape)
        v=self.fc3(x) # Translate spatial data sequence to a higher dimension
        # print('V:',v.shape)`
        # print(torch.matmul(Q,K.T).shape,v.shape)
        # based on previous spatio temporal pattern and current spatial data which connections are more important
        # then weight the current spatial connections
        # out=Func.softmax(torch.matmul(Q,K.T),dim=1)@v
        
        out=Func.sigmoid(torch.matmul(Q,K.T))@v
        # print('out:',out.shape,'v:',v.shape)
        out=self.threshold_relu(out,self.th,x_original[3]) # Threshold the output # cut the connections which are not important
        out=self.fc_out(out)
        out=out.reshape(x_original)
        return out

# class social_stgcnn(nn.Module):
class TAG(nn.Module):

    def __init__(self,threshold,n_gcnn =1,n_tcnn=1,input_feat=2,output_feat=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(TAG,self).__init__()
        self.sie = SIE(input_feat,threshold) # sei initialization
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
            # skip tcnn layers
            # send the output of stgcn directly to output
            # If n_tcnn < 1 we want to behave like "skip TCNN" while keeping
            # forward() indexing safe. Force n_tcnn to 1 and provide identity modules.
            if self.n_tcnn < 1:
                self.n_tcnn = 1
            self.tpcnns = nn.ModuleList()
            self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
            # self.tpcnns = nn.ModuleList([nn.Identity()])
            self.tpcnn_ouput = nn.Identity()
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_tcnn):
            self.prelus.append(nn.PReLU())

    def viz_threshold(self,v):
        fig,ax=plt.subplots(1,3,figsize=(6,6))
        ax[0].imshow(v[0,0,:,:].cpu().detach().numpy())
        ax[0].set_title('X')
        ax[1].imshow(v[0,1,:,:].cpu().detach().numpy())
        ax[1].set_title('Y')
        ax[2].scatter(v[0,0,:,:].cpu().detach().numpy(),v[0,1,:,:].cpu().detach().numpy())
        ax[2].set_title('Scatter')
        plt.set_cmap('inferno')
        plt.colorbar(ax[0].imshow(v[0,0,:,:].cpu().detach().numpy()),ax=ax[0])
        plt.colorbar(ax[1].imshow(v[0,1,:,:].cpu().detach().numpy()),ax=ax[1])
        plt.title('Reshaped Tensor')
        plt.show()
        
    def forward(self,v,a):
        # print('V:',v.shape) # batch, 2dxy,seq_len, num_nodes
        # print('A:',a.shape) # seq_len, num_nodes, num_nodes
        v = self.sie(v) # SIE layer (Spatio-Temporal Interaction Encoder) meunpredictable
        # print('V:',v.shape)
        # loop through the ST-GCN layers based on parameter n_gcnn
        # Loop through the ST-GCN layers based on parameter n_gcnn
        for k in range(self.n_gcnn):
            v, a = self.st_gcns[k](v, a)

        # Reshape the tensor for the temporal convolutional layers
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])
        # visualize the reshaped tensor
        # self.viz_threshold(v)
        # Apply the first TCNN layer with PReLU activation
        if self.n_tcnn >= 1:
            v = self.prelus[0](self.tpcnns[0](v))

            # Loop through the remaining TCNN layers based on parameter n_tcnn
            for k in range(1, self.n_tcnn - 1):
                v = self.prelus[k](self.tpcnns[k](v)) + v

        # Apply the final TCNN layer
        v = self.tpcnn_ouput(v)

        # Reshape the tensor back to the original shape
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3]) # batch, seq_len, pred_seq_len, num_nodes

        # Return the final output and adjacency matrix
        return v, a
