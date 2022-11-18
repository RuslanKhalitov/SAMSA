from training_utils import count_params, seed_everything, init_weights, train_epoch, \
    train_epoch_bins, eval_model, eval_model_bins
from dataloader_utils import DatasetCreator, DatasetCreatorFlat, concater_collate, pad_collate
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, accuracy_score
from SAMSA import SAMSA
from models.backbones import TransformerModel, LinformerModel, ReformerModel, NystromformerModel, \
    PoolformerModel, CosformerModel, S4_Model, LunaModel, LSTM, CONV

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

tasks = {
    'Carassius vs. Labeo':
        {
            'classes': ['Carassius', 'Labeo'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt',
            'max_seq_len': 100101
        },
    'Danio vs. Cyprinus':
        {
            'classes': ['Danio', 'Cyprinus'],
            'category_name': 'Other vertebrate',
            'name_raw': 'gbvrt',
            'max_seq_len': 261943
        },
    'Mus vs. Rattus':
        {
            'classes': ['Mus', 'Rattus'],
            'category_name': 'Rodent',
            'name_raw': 'gbrod',
            'max_seq_len': 261093
        },
    'Sus vs. Bos':
        {
            'classes': ['Sus', 'Bos'],
            'category_name': 'Other mammalian',
            'name_raw': 'gbmam',
            'max_seq_len': 447010
        },
}
problem_names = ['adding', 'genbank', 'longdoc']

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem_class", type=str, default='genbank')
parser.add_argument("--problem", type=str, default='Sus vs. Bos')
parser.add_argument("--model", type=str, default='linformer')
parser.add_argument("--device_id", type=int, default=1)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()
model = args.model
assert args.problem in tasks.keys(), f'Please use the correct problem name: {tasks.keys()}'
assert args.problem_class == 'genbank', 'Please use the correct problem name: genbank'

# Parsing training config
stream = open("config.yaml", 'r')
cfg_yaml = yaml.safe_load(stream)[args.problem_class][args.problem]
training_config = cfg_yaml['training']
print('training config', training_config)
config = cfg_yaml['models'][args.model]
config['problem'] = 'genbank'
config['model'] = model
print('model config', config)

# sys.exit()

torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
print('set up device:', device)

classes = tasks[args.problem]['classes']
data_train = pd.read_pickle(f'/lhome/ruslank/sparsefactorisation/mixers/{classes[0]}_{classes[1]}_train.pkl')
data_test = pd.read_pickle(f'/lhome/ruslank/sparsefactorisation/mixers/{classes[0]}_{classes[1]}_test.pkl')
max_seq_len = max(max(data_train['len']), max(data_test['len']))

# change binning
bins = [2**i for i in range(5, 22)]
bin_labels = [i for i in range(6, 22)]
for df in [data_train, data_test]:
    df['bin'] = pd.cut(df['len'], bins=bins, labels=bin_labels)
print('Maximum Sequence Length', max_seq_len)

# sys.exit()
#Wandb setting
naming_log = f"Genbank {args.problem} {args.model}"
wandb.init(project="SAMSA", entity=args.wandb, name=naming_log)
wandb.config = config

#Model initialization
if args.model == 'samsa':
    net = SAMSA
    net = net(
        problem=config['problem'],
        vocab_size=config['vocab_size'],
        max_seq_len=max_seq_len,
        embedding_type=config['embedding_type'],
        embedding_size=config['embedding_size'],
        positional_embedding=config['positional_embedding'],
        hidden_size=config['hidden_size'],
        mlp_dropout=config['mlp_dropout'],
        layer_dropout=config['layer_dropout'],
        n_class=training_config['n_class']
    )
elif args.model == 'lstm':
    net = LSTM(
        vocab_size=config['vocab_size'],
        dim=config['embedding_size'],
        depth=config['n_layers'],
        n_class=training_config['n_class'],
        problem=config['problem'],
    )
elif args.model == 'cnn':
    net = CONV(
        problem=config['problem'],
        model='CNN',
        vocab_size=config['vocab_size'],
        dim=config['embedding_size'],
        depth=config['n_layers'],
        kernel_size=3,
        n_class=training_config['n_class']
    )
elif args.model == 'tcn':
    net = CONV(
        problem=config['problem'],
        model='TCN',
        vocab_size=config['vocab_size'],
        dim=config['embedding_size'],
        depth=config['n_layers'],
        kernel_size=3,
        n_class=training_config['n_class']
    )
else:
    if args.model == 'transformer':
        net = TransformerModel
    elif args.model == 'linformer':
        net = LinformerModel
    elif args.model == 'reformer':
        net = ReformerModel
    elif args.model == 'nystromformer':
        net = NystromformerModel
    elif args.model == 'poolformer':
        net = PoolformerModel
    elif args.model == 'cosformer':
        net = CosformerModel
    elif args.model == 'S4':
        net = S4_Model
    elif args.model == 'luna':
        net = LunaModel
    net = net(
        problem=config['problem'],
        vocab_size=config['vocab_size'],
        dim=config['embedding_size'],
        depth=config['n_layers'],
        heads=config['n_heads'],
        n_vec=max_seq_len,
        n_class=training_config['n_class'],
        pooling='avg',
        device=device
    )

net = net.to(device)
net.apply(init_weights)
print('Number of trainable parameters', count_params(net))
print(config)
# sys.exit()

class_weights = compute_class_weight('balanced', classes=[0, 1], y=data_train['label'])
print('class weights:', class_weights)

loss = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float).to(device),
    reduction='mean'
)

optimizer = optim.AdamW(
    net.parameters(),
    lr=config['learning_rate'],
    betas = (0.9, 0.98), eps = 1e-8, weight_decay=0.01
)

# Dataset preparation

# Prepare the training loader
trainset = DatasetCreator(
    df=data_train,
    batch_size=config['batch_size'],
    var_len=True
)

trainloader = DataLoader(
    trainset,
    batch_size=config['batch_size'],
    shuffle=False if config['model'] == 'samsa' else True,
    collate_fn=concater_collate if config['model'] == 'samsa' else pad_collate,
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
    collate_fn=concater_collate if config['model'] == 'samsa' else pad_collate,
    drop_last=False,
    num_workers=4
)

for epoch in range(training_config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch_bins(config, net, optimizer, loss, trainloader, device=device)
    accuracy = eval_model_bins(config, net, testloader, metric=roc_auc_score, device=device) 
    print(f'Epoch {epoch+1} completed. Test accuracy: {accuracy}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{test_roc_auc:.3f}.pt')
