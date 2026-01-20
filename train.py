from json import load
import os

import math
import sys
import time

from pytest import skip

import torch
import torch.nn as nn
import numpy
from numpy import linalg as LA
import networkx as nx

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=2)
parser.add_argument('--output_size', type=int, default=5)
parser.add_argument('--n_gcnn', type=int, default=2,help='Number of GCN layers')
parser.add_argument('--n_tcnn', type=int, default=6, help='Number of CNN layers')
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--thres', type=float, default=0.3,help='Threshold to make connections between agents 0.1-0.98')

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=12)
parser.add_argument('--dataset', default='eth',
                    help='Apolloscape,eth,hotel,univ,zara1,zara2,SDD')

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=512,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=150,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=75,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
parser.add_argument('--delim', default='\t',help='delimiter used in the dataset,eth=tab,apolloscape/SDD=space')
parser.add_argument('--shuffle', action="store_true", default=False,
                    help='Whether to shuffle the sequences')
parser.add_argument('--reload_data', action="store_true", default=False,
                    help='Whether to reaload the data from all files ')
parser.add_argument('--n_splits', type=int, default=1,
                    help='Number of splits for k-fold cross-validation')
parser.add_argument('--dataset_path', type=str,
                    help='Path to the dataset directory')
parser.add_argument('--skip_val', action="store_true", default=False,
                    help='Whether to skip validation during training ')
                    
args = parser.parse_args()







print('*'*30)
print("Training initiating....")
print(args)
# log all the training loss in a file
log_file= open('./log/'+time.ctime()+'_log'+'.txt','w')
log_file.write(str(args)+'\n') # avoid this line while plotting the graph
log_file.write('Epoch,Train_loss\n')

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = args.dataset_path 
# check if the train dataset is already saved (in .processed)
if os.path.exists('./processed/train/train.pkl'):
    with open('./processed/train/train.pkl', 'rb') as f:
        dset_train = pickle.load(f)
else:
    dset_train = TrajectoryDataset(
                data_set,
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True,delim=args.delim,dataset_name=args.dataset) # delim= space for apolloscape
    print('Training/val/test set processing complete !!!')
    # # save train dataset
    # with open(data_set+'train/dataset.pkl', 'wb') as f:
    #     pickle.dump(dset_train, f)


loader_train = DataLoader(
        dset_train,
        batch_size=1, #This is irrelative to the args batch size parameter
        shuffle =True,
        num_workers=0)


if not args.skip_val:
    # check if the val dataset is already saved
    if os.path.exists('./processed/val/val.pkl'):
        with open('./processed/val/val.pkl', 'rb') as f:
            dset_val = pickle.load(f)
    # else:
    #     dset_val =TrajectoryDataset(
    #             data_set+'val/',
    #             obs_len=obs_seq_len,
    #             pred_len=pred_seq_len,
    #             skip=1,norm_lap_matr=True,delim=args.delim,fill_missing=True)
    #     print('Validation set processing complete!!!')
    #     # save val dataset
    #     with open(data_set+'val/dataset.pkl', 'wb') as f:
    #         pickle.dump(dset_val, f)

    loader_val = DataLoader(
            dset_val,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =False,
            num_workers=1)


#Defining the model 
# tmp_model=SIE(in_feat=2).cuda()
model = TAG(n_gcnn =args.n_gcnn,n_tcnn=args.n_tcnn,
output_feat=args.output_size,seq_len=args.obs_seq_len,
kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len,threshold=args.thres)
model.cuda()


#Training settings 

optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    


checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
    
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    


print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}

def train_fold(model, optimizer, scheduler, loader_train, loader_val, fold, metrics, constant_metrics):
    for epoch in range(args.num_epochs):
        train(epoch, model, optimizer, loader_train, metrics)
        if not args.skip_val:
            vald(epoch, model, loader_val, metrics, constant_metrics)
        
        # Check if the average loss is NaN
        if np.isnan(metrics['train_loss'][-1]) or np.isnan(metrics['val_loss'][-1]):
            print(f"NaN loss detected at epoch {epoch}. Reducing learning rate.")
            scheduler.step()
        elif args.use_lrschd:
            scheduler.step()

        print('*'*30)
        print(f'Fold {fold} - Epoch:', args.tag, ":", epoch)
        for k, v in metrics.items():
            if len(v) > 0:
                print(k, v[-1])

        print(constant_metrics)
        print('*'*30)

        with open(checkpoint_dir + f'metrics_fold_{fold}.pkl', 'wb') as fp:
            pickle.dump(metrics, fp)

        with open(checkpoint_dir + f'constant_metrics_fold_{fold}.pkl', 'wb') as fp:
            pickle.dump(constant_metrics, fp)

def train(epoch, model, optimizer, loader_train, metrics):
    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1

    for cnt, batch in enumerate(loader_train): 
        batch_count += 1

        # 1. Slice off the metadata (last 2 items: metadata tuple and sequence start_end list)
        # The first 10 items are the tensors we need for the model
        batch_tensors = batch[:-1]
    
        # Optional: Capture metadata if you want to use it for logging/debugging
        batch_meta = batch[-1]      # The tuple of 'scene_video.pt' strings
        # print('Batch meta:', batch_meta)
        # batch_start_end = batch[-1] # The sequence start/end indices

        batch = [tensor.cuda() for tensor in batch_tensors]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch [:10]

        optimizer.zero_grad()
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            # print(V_pred.shape, V_tr.shape)
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            loss_batch += loss.item()
            print('TRAIN:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)
            
    metrics['train_loss'].append(loss_batch / batch_count)
    log_file.write(str(epoch) + ',' + str(loss_batch / batch_count) + '\n')

def vald(epoch, model, loader_val, metrics, constant_metrics):
    model.eval()
    loss=torch.ones(1).cuda()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point = int(loader_len / args.batch_size) * args.batch_size + loader_len % args.batch_size - 1
    
    for cnt, batch in enumerate(loader_val): 
        # 1. Slice off the metadata (last 2 items: metadata tuple and sequence start_end list)
        # The first 10 items are the tensors we need for the model
        batch_tensors = batch[:-1]
    
        # Optional: Capture metadata if you want to use it for logging/debugging
        batch_meta = batch[-1]      # The tuple of 'scene_video.pt' strings
        # print('Batch meta:', batch_meta)
        # batch_start_end = batch[-1] # The sequence start/end indices

        batch = [tensor.cuda() for tensor in batch_tensors]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)
        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
        V_pred = V_pred.permute(0, 2, 3, 1)
        V_tr = V_tr.squeeze()
        V_pred = V_pred.squeeze()
        print('Val batch_count:', batch_count)
        
        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss_batch += loss.item()
            print('VALD:', '\t Epoch:', epoch, '\t Loss:', loss_batch / batch_count)

    metrics['val_loss'].append(loss_batch / batch_count)
    if metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] = metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch

def calculate_ade_fde(model, loader_val):
    model.eval()
    ade_ls = []
    fde_ls = []

    with torch.no_grad():
        for batch in loader_val:
            # 1. Slice off the metadata (last 2 items: metadata tuple and sequence start_end list)
            # The first 10 items are the tensors we need for the model
            batch_tensors = batch[:-1]
    
            # Optional: Capture metadata if you want to use it for logging/debugging
            batch_meta = batch[-1]      # The tuple of 'scene_video.pt' strings
            # print('Batch meta:', batch_meta)
            # batch_start_end = batch[-1] # The sequence start/end indices

            batch = [tensor.cuda() for tensor in batch_tensors]
        
            # batch = [tensor.cuda() for tensor in batch]
            obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped, loss_mask, V_obs, A_obs, V_tr, A_tr = batch

            V_obs_tmp = V_obs.permute(0, 3, 1, 2)
            V_pred, _ = model(V_obs_tmp, A_obs.squeeze())
            V_pred = V_pred.permute(0, 2, 3, 1)
            V_tr = V_tr.squeeze()
            V_pred = V_pred.squeeze()

            pred = [V_pred.cpu().numpy()]
            target = [V_tr.cpu().numpy()]
            count = [V_tr.size(0)]

            ade_ls.append(ade(pred, target, count))
            fde_ls.append(fde(pred, target, count))

    return np.mean(ade_ls), np.mean(fde_ls)





# Main cross-validation loop
print('Training initiating....')
print(args)
log_file = open('./log/' + time.ctime() + '_log' + '.txt', 'w')
log_file.write(str(args) + '\n')
log_file.write('Epoch,Train_loss\n')

data_set = args.dataset_path if args.dataset_path else './datasets/' + args.dataset + '/'

# check if train test validation split exists
dir_in_dataset=os.listdir(data_set )


if args.reload_data:
    dset_train = TrajectoryDataset('./processed/train/', obs_len=args.obs_seq_len, pred_len=args.pred_seq_len, skip=1, norm_lap_matr=True, delim=args.delim, shuffle=args.shuffle, n_splits=args.n_splits)
    # Save the updated dataset
    with open('./processed/train/train.pkl', 'wb') as f:
        pickle.dump(dset_train, f)
else:
    # Make sure kfolds is generated for cross-validation
    if args.n_splits > 1:
        if not hasattr(dset_train, 'kfolds') or len(dset_train.kfolds) != args.n_splits:
            print("Generating kfolds for the loaded dataset...")
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=args.n_splits)
            dset_train.kfolds = list(kf.split(np.arange(dset_train.num_seq)))

fold_metrics = []
fold_ade_fde = []
best_val_loss = float('inf')
best_model_state = None

# Normal training with separate validation set if n_splits=1
if args.n_splits == 1:
    print("Using normal training with separate validation set")
    fold = 0
    # Use the existing loaders defined earlier
    model = TAG(n_gcnn=args.n_gcnn, n_tcnn=args.n_tcnn, output_feat=args.output_size, seq_len=args.obs_seq_len, kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len, threshold=args.thres).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    if args.use_lrschd:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    else:
        scheduler = None
    metrics = {'train_loss': [], 'val_loss': []}
    constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}
    if args.skip_val:
        loader_val = loader_train  # Use training data as validation if skipping validation
    train_fold(model, optimizer, scheduler, loader_train, loader_val, fold, metrics, constant_metrics)
    
    ade_calc, fde_calc = calculate_ade_fde(model, loader_val)
    fold_ade_fde.append((ade_calc, fde_calc))
    
    print(f"Training completed - ADE: {ade_calc}, FDE: {fde_calc}")
    
    fold_metrics.append(metrics)
    best_model_state = model.state_dict()
else:
    # K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(dset_train.kfolds):
        print(f"Fold {fold + 1}/{len(dset_train.kfolds)}")
        
        train_subset = torch.utils.data.Subset(dset_train, train_idx)
        val_subset = torch.utils.data.Subset(dset_train, val_idx)
        
        loader_train_fold = DataLoader(train_subset, batch_size=1, shuffle=True, num_workers=0)
        if not args.skip_val:
            loader_val_fold = DataLoader(val_subset, batch_size=1, shuffle=False, num_workers=1)
        
        model = TAG(n_gcnn=args.n_gcnn, n_tcnn=args.n_tcnn, output_feat=args.output_size, seq_len=args.obs_seq_len, kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len, threshold=args.thres).cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
        if args.use_lrschd:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
        
        metrics = {'train_loss': [], 'val_loss': []}
        constant_metrics = {'min_val_epoch': -1, 'min_val_loss': 9999999999999999}
        
        if not args.skip_val:
            train_fold(model, optimizer, scheduler, loader_train_fold, loader_val_fold, fold, metrics, constant_metrics)
        else:
            train_fold(model, optimizer, scheduler, loader_train_fold, loader_train_fold, fold, metrics, constant_metrics)  
        ade_calc, fde_calc = calculate_ade_fde(model, loader_val_fold)
        fold_ade_fde.append((ade_calc, fde_calc))
        
        print(f"Fold {fold + 1} ADE: {ade_calc}, FDE: {fde_calc}")
        
        fold_metrics.append(metrics)
        
        if constant_metrics['min_val_loss'] < best_val_loss:
            best_val_loss = constant_metrics['min_val_loss']
            best_model_state = model.state_dict()

# Save the best model
torch.save(best_model_state, checkpoint_dir + 'best_model.pth')

# Calculate and print overall statistics
all_train_losses = [metrics['train_loss'][-1] for metrics in fold_metrics]
all_val_losses = [metrics['val_loss'][-1] for metrics in fold_metrics]
all_ades = [ade for ade, fde in fold_ade_fde]
all_fdes = [fde for ade, fde in fold_ade_fde]

print("Cross-validation results:")
print(f"Train Losses: {all_train_losses}")
print(f"Validation Losses: {all_val_losses}")
print(f"Mean Train Loss: {np.mean(all_train_losses)}")
print(f"Std Train Loss: {np.std(all_train_losses)}")
print(f"Mean Validation Loss: {np.mean(all_val_losses)}")
print(f"Std Validation Loss: {np.std(all_val_losses)}")
print(f"Validation ADEs: {all_ades}")
print(f"Validation FDEs: {all_fdes}")
print(f"Mean Validation ADE: {np.mean(all_ades)}")
print(f"Std Validation ADE: {np.std(all_ades)}")
print(f"Mean Validation FDE: {np.mean(all_fdes)}")
print(f"Std Validation FDE: {np.std(all_fdes)}")

# write cv results to log_file
log_file.write("Cross-validation results:\n")
log_file.write(f"Train Losses: {all_train_losses}\n")
log_file.write(f"Validation Losses: {all_val_losses}\n")
log_file.write(f"Mean Train Loss: {np.mean(all_train_losses)}\n")
log_file.write(f"Std Train Loss: {np.std(all_train_losses)}\n")
log_file.write(f"Mean Validation Loss: {np.mean(all_val_losses)}\n")
log_file.write(f"Std Validation Loss: {np.std(all_val_losses)}\n")
log_file.write(f"Validation ADEs: {all_ades}\n")
log_file.write(f"Validation FDEs: {all_fdes}\n")
log_file.write(f"Mean Validation ADE: {np.mean(all_ades)}\n")
log_file.write(f"Std Validation ADE: {np.std(all_ades)}\n")
log_file.write(f"Mean Validation FDE: {np.mean(all_fdes)}\n")
log_file.write(f"Std Validation FDE: {np.std(all_fdes)}\n")

log_file.close()



