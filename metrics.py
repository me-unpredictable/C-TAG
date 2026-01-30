import os
import math
import sys
import time
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx


def ade(predAll,targetAll,count_):
    All = len(predAll)
    # sum_all = -0.5*All # ..........
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T): # all positions MSE
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    # sum_all = -0.5*All # ............
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T): # final position MSE
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)
    return sum_all/All


def rmse(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_ += (pred[i, t, 0] - target[i, t, 0])**2 + (pred[i, t, 1] - target[i, t, 1])**2
        sum_all += sum_ / (N * T)
        
    return np.sqrt(sum_all / All)


def seq_to_nodes(seq_):
    max_nodes = seq_.shape[1] #number of pedestrians in the graph
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss_old(V_pred,V_trgt):
    #mux, muy, sx, sy, corr
    #assert V_pred.shape == V_trgt.shape
    normx = V_trgt[:,:,0]- V_pred[:,:,0]
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sigma x (
    sy = torch.exp(V_pred[:,:,3]) #sigma y
    corr = torch.tanh(V_pred[:,:,4]) #corr
    
    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z/(2*negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    
    return result

def bivariate_loss(V_pred, V_trgt, mask=None):
    """
    Bivariate Gaussian NLL Loss (Vectorized).
    V_pred: [Batch, Time, Nodes, 5] (mu_x, mu_y, log_sig_x, log_sig_y, logit_corr)
    V_trgt: [Batch, Time, Nodes, 2] (gt_x, gt_y)
    mask:   [Batch, Time, Nodes] (optional)
    """
    mu_x = V_pred[..., 0]
    mu_y = V_pred[..., 1]
    
    # Exponentiate sigmas (model outputs log_sigma for stability)
    # Clamp log_sigma to prevent Inf/NaN in exp() and division by zero
    log_sx = torch.clamp(V_pred[..., 2], min=-20, max=6)
    log_sy = torch.clamp(V_pred[..., 3], min=-20, max=6)
    
    sx = torch.exp(log_sx) 
    sy = torch.exp(log_sy)
    
    # Tanh correlation (model outputs logit to ensure [-1, 1])
    corr = torch.tanh(V_pred[..., 4])

    x = V_trgt[..., 0]
    y = V_trgt[..., 1]

    # Normalized differences
    normx = x - mu_x
    normy = y - mu_y

    sxsy = sx * sy
    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    
    negRho = 1 - corr**2
    # Clamp for numerical stability
    negRho = torch.clamp(negRho, min=1e-20)
    
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    denom = torch.clamp(denom, min=1e-20)

    result = torch.exp(-z/(2*negRho))
    result = result / denom

    loss = -torch.log(torch.clamp(result, min=1e-20))
    
    # Apply Mask if provided
    if mask is not None:
        # SAFER: Force padded values to 0.0 even if they are NaN/Inf
        # mask is 1 for valid, 0 for pad. ~mask.bool() is True for pad.
        loss = loss.masked_fill(~mask.bool(), 0.0)
        
        num_valid = torch.sum(mask)
        if num_valid > 0:
            return torch.sum(loss) / num_valid
        else:
             # Avoid NaN if batch is empty (unlikely)
            return torch.tensor(0.0, device=loss.device)
            
    # Mean over all dimensions
    return torch.mean(loss)

def masked_mse_loss(V_pred, V_trgt, mask=None):
    """
    Masked MSE Loss for warm-up.
    V_pred: [Batch, Time, Nodes, 5] (mu_x, mu_y, ...) - we only use mu_x, mu_y
    V_trgt: [Batch, Time, Nodes, 2] (gt_x, gt_y)
    mask:   [Batch, Time, Nodes] (optional)
    """
    mu_x = V_pred[..., 0]
    mu_y = V_pred[..., 1]
    
    x = V_trgt[..., 0]
    y = V_trgt[..., 1]
    
    # Squared Error
    loss = (x - mu_x)**2 + (y - mu_y)**2
    
    # Apply Mask if provided
    if mask is not None:
        loss = loss.masked_fill(~mask.bool(), 0.0)
        
        num_valid = torch.sum(mask)
        if num_valid > 0:
            return torch.sum(loss) / num_valid
        else:
            return torch.tensor(0.0, device=loss.device)
            
    return torch.mean(loss)