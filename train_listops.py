from training_utils import count_params, seed_everything, init_weights, train_epoch, eval_model, train_epoch_bins, eval_model_bins
from dataloader_utils import DatasetCreator, concater_collate, DatasetCreatorFlat, pad_collate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
from SAMSA import SAMSA

import argparse
import sys
import ast
import math
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LinearLR
import pandas as pd
import numpy as np
import wandb
import yaml 

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='listops')
parser.add_argument("--model", type=str, default='samsa')
parser.add_argument("--device_id", type=int, default=1)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()

config = {
    'problem': 'listops',
    'model': 'samsa',
    'search': False,
    'max_seq_len': 1999,
    'vocab_size': 25,
    'embedding_size': 300,
    'hidden_size': 240,
    'mlp_dropout': 0.1,
    'layer_dropout': 0.1,
    'n_class': 10,
    'n_epochs': 270,
    'positional_embedding': False,
    'embedding_type': 'sparse',
    'learning_rate': 0.001,
    'batch_size': 128,
    'warmup_epochs': 8,
    'warmup_scale': 1e-8
}


torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

# task variables
data_train = pd.read_pickle('experiments/data/LRA/listops_train.pkl')
data_test = pd.read_pickle('experiments/data/LRA/listops_test.pkl')


naming_log = "Listops"
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

trainset = DatasetCreator(
    df=data_train,
    batch_size=config['batch_size'],
    var_len=True
)

trainloader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=concater_collate,
    drop_last=False,
    num_workers=4
)


# Prepare the testing loader
testset = DatasetCreator(
    df=data_test,
    batch_size=config['batch_size'],
    var_len=True
)

testloader = DataLoader(
    testset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=concater_collate,
    drop_last=False,
    num_workers=4
)

for epoch in range(config['n_epochs']):

    print(f'Starting epoch {epoch+1}')
    train_epoch_bins(config, net, optimizer, loss, trainloader, scheduler=scheduler, device=device)
    acc = eval_model_bins(config, net, testloader, metric=accuracy_score, device=device) 
    print(f'Epoch {epoch+1} completed. Test accuracy: {acc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{acc:.3f}.pt')

