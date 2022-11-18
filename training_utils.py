
import os
import sys
import math
import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

import wandb

def seed_everything(seed=1234):
    """
    Fixes random seeds, to get reproducible results.
    :param seed: a random seed across all the used packages
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def count_params(net):
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return n_params

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        try:
            m.bias.data.fill_(0.01)
        except:
            print('no bias in the model')


def train_epoch(config, net, optimizer, loss, trainloader, device, scheduler=None):
    net.train()
    running_loss = 0.0
    n_items_processed = 0
    num_batches = len(trainloader)
    for idx, (X, Y) in tqdm(enumerate(trainloader), total=num_batches):
        if config['problem'] == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            output = net(X).squeeze()
        else:
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)
            output = net(X)
            
        output = loss(output, Y)
        output.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        optimizer.zero_grad()
        running_loss += output.item()
        n_items_processed += Y.size(0)

    total_loss = running_loss / num_batches
    print(f'Training loss after epoch: {total_loss}')
    wandb.log({'train loss': total_loss})

def train_epoch_bins(config, net, optimizer, loss, trainloader, device, scheduler=None):
    net.train()
    running_loss = 0.0
    n_items_processed = 0
    num_batches = len(trainloader)
    for idx, (X, Y, length, bin) in tqdm(enumerate(trainloader), total=num_batches):
        if config['problem'] == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            if config['model'] in ['samsa', 'cosformer', 'luna']:
                output = net(X, length).squeeze()
            else:
                output = net(X).squeeze()
        else:
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)
            if config['model'] in ['samsa', 'cosformer', 'luna']:
                output = net(X, length)
            else:
                output = net(X)
        
        output = loss(output, Y)
        output.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
            
        optimizer.zero_grad()
        running_loss += output.item()
        n_items_processed += Y.size(0)

    total_loss = running_loss / num_batches
    print(f'Training loss after epoch: {total_loss}')
    wandb.log({'train loss': total_loss})
    
def eval_model(config, net, valloader, metric, device) -> float:
    net.eval()

    preds = []
    targets = []

    num_batches = len(valloader)
    for idx, (X, Y) in tqdm(enumerate(valloader), total=num_batches):
        if config['problem'] == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            output = net(X).squeeze()
            predicted = output
        else:
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)  
            output = net(X) 
            _, predicted = output.max(1)
        
        targets.extend(Y.detach().cpu().numpy().flatten())
        preds.extend(predicted.detach().cpu().numpy().flatten())

    total_metric = metric(preds, targets)

    wandb.log({'test metric': total_metric})

    return total_metric


def eval_model_bins(config, net, valloader, metric, device) -> float:
    net.eval()

    preds = []
    targets = []
    bins = []

    num_batches = len(valloader)
    for idx, (X, Y, length, bin) in tqdm(enumerate(valloader), total=num_batches):
        if config['problem'] == 'adding':
            X = X.float().to(device)
            Y = Y.float().to(device)
            if config['model'] in ['samsa', 'cosformer', 'luna']:
                output = net(X, length).squeeze()
            else:
                output = net(X).squeeze()
            predicted = output
        else:
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)  
            if config['model'] in ['samsa', 'cosformer', 'luna']:
                output = net(X, length)
            else:
                output = net(X)
            _, predicted = output.max(1)
            
        targets.extend(Y.detach().cpu().numpy().flatten())
        preds.extend(predicted.detach().cpu().numpy().flatten())
        bins.extend(bin)

    try:
        total_metric = metric(preds, targets)
    except ValueError:
        total_metric = 0.5
           
    results = pd.DataFrame(data={'bins': bins, 'predictions': preds, 'labels': targets})

    # Calculate scores for each bin
    def bin_scores(df):
        scores_dict = {}
        for i in sorted(df['bins'].unique()):
            data = df[df['bins'] == i]
            try:
                scores_dict[f'test_bin_2^{i}'] = metric(data['predictions'], data['labels'])
            except:
                # Can't calculate the ROCAUC score because only one class appears in the group
                scores_dict[f'test_bin_2^{i}'] = 0.5
        wandb.log(scores_dict)

    bin_scores(results)
    wandb.log({'test metric': total_metric})

    return total_metric


def train_epoch_texts(config, net, optimizer, loss, trainloader, device, log_every=50, scheduler=None):
    net.train()
    running_loss = 0.0
    n_items_processed = 0
    num_batches = len(trainloader)
    for idx, one_input in tqdm(enumerate(trainloader), total=num_batches):
        if config['problem'] == 'retrieval_4000':
            X0 = one_input['input_ids_0']
            X1 = one_input['input_ids_1']
            Y = one_input['label']

            X0 = X0.to(device)
            X1 = X1.to(device)
            Y = Y.type(torch.LongTensor).to(device)

            output = net(X0, X1)
        else:
            X = one_input['input_ids_0']
            Y = one_input['label']
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)
            output = net(X)

        output = loss(output, Y)
        output.backward()
        optimizer.step()
        optimizer.zero_grad()
        if scheduler:
            scheduler.step()

        running_loss += output.item()
        n_items_processed += Y.size(0)

        if (idx + 1) % log_every == 0:
            print(f'Avg loss after {idx + 1} batches: {running_loss / n_items_processed}')
            # print(f'Current loss on a batch {idx + 1}: {output.item() / len(length)}')

    total_loss = running_loss / num_batches
    print(f'Training loss after epoch: {total_loss}')
    wandb.log({'train loss': total_loss})


def eval_model_texts(config, net, valloader, metric, device) -> float:
    net.eval()

    preds = []
    targets = []
    bins = []

    num_batches = len(valloader)
    for idx, one_input in tqdm(enumerate(valloader), total=num_batches):
        if config['problem'] == 'retrieval_4000':
            X0 = one_input['input_ids_0']
            X1 = one_input['input_ids_1']
            Y = one_input['label']

            X0 = X0.to(device)
            X1 = X1.to(device)
            Y = Y.type(torch.LongTensor).to(device)

            output = net(X0, X1)
        else:
            X = one_input['input_ids_0']
            Y = one_input['label']
            X = X.to(device)
            Y = Y.type(torch.LongTensor).to(device)
            output = net(X)

        _, predicted = output.max(1)
        targets.extend(Y.detach().cpu().numpy().flatten())
        preds.extend(predicted.detach().cpu().numpy().flatten())

    total_metric = metric(preds, targets)
    
    results = pd.DataFrame(data={'predictions': preds, 'labels': targets})

    wandb.log({'test metric': total_metric})

    return total_metric
