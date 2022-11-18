from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model
from dataloader_utils import DatasetCreator, concater_collate, DatasetCreatorFlat
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
from SAMSA import SAMSA
from torch.optim.lr_scheduler import LinearLR

import argparse
import sys
import ast
import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import wandb
import yaml 


parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='cifar10')
parser.add_argument("--model", type=str, default='chordmixer')
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()

config = {
    'problem': 'cifar10',
    'search': False,
    'max_seq_len': 1024,
    'vocab_size': 256,
    'embedding_size': 321,
    'hidden_size': 196,
    'mlp_dropout': 0.,
    'layer_dropout': 0.,
    'n_class': 10,
    'n_epochs': 270,
    'positional_embedding': False,
    'embedding_type': 'linear',
    'learning_rate': 0.0001,
    'batch_size': 96,
    'warmup_epochs': 100,
    'warmup_scale': 1e-8
}


# sys.exit()

torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

# task variables
data_train = torch.load('experiments/data/LRA/cifar10_train.pt')
labels_train = torch.load('experiments/data/LRA/cifar10_labels_train.pt').to(torch.int32)

data_test = torch.load('experiments/data/LRA/cifar10_test.pt')
labels_test = torch.load('experiments/data/LRA/cifar10_labels_test.pt').to(torch.int32)

if config['embedding_type'] == 'linear':
    print('linear mapping')
    mean = 0.5
    std = 0.5
    data_train = data_train.to(torch.float).div(255.).sub(mean).div(std)
    data_test = data_test.to(torch.float).div(255.).sub(mean).div(std)
else:
    data_train = data_train.to(torch.int32)
    data_test = data_test.to(torch.int32)
    
# sys.exit()
#Wandb setting

naming_log = "CIFAR10"
if not config['search']:
    wandb.init(project="SAMSA_LRA", entity=args.wandb, name=naming_log)
    wandb.config = config
else:
    wandb.init(project="SAMSA_LRA", entity=args.wandb, config=config)
    config = wandb.config
    print('CONFIG')
    print(config)

net = SAMSA(
    vocab_size=config['vocab_size'],
    max_seq_len=config['max_seq_len'],
    embedding_type=config['embedding_type'],
    embedding_size=config['embedding_size'],
    hidden_size=config['hidden_size'],
    mlp_dropout=config['mlp_dropout'],
    layer_dropout=config['layer_dropout'],
    n_class=config['n_class'],
    positional_embedding=config['positional_embedding']
)


net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))
print(config)

loss = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    net.parameters(),
    lr=config['learning_rate'],
    betas = (0.9, 0.98), eps = 1e-8, weight_decay=0.01
)

scheduler = LinearLR(optimizer, start_factor=config['warmup_scale'], total_iters=config['warmup_epochs'])
# scheduler = None

# Dataset preparation
    
# Prepare the training loader
trainset = DatasetCreatorFlat(
    df=data_train,
    labels=labels_train
)

trainloader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=True,
    drop_last=False,
    num_workers=4
)


# Prepare the testing loader
testset = DatasetCreatorFlat(
    df=data_test,
    labels=labels_test
)

testloader = DataLoader(
    testset,
    batch_size=config['batch_size'],
    shuffle=False,
    drop_last=False,
    num_workers=4
)


for epoch in range(config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch(config, net, optimizer, loss, trainloader, scheduler=scheduler, device=device)
    acc = eval_model(config, net, testloader, metric=accuracy_score, device=device) 
    print(f'Epoch {epoch+1} completed. Test accuracy: {acc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{test_roc_auc:.3f}.pt')
