from training_utils import count_params, seed_everything, init_weights, eval_model_texts, train_epoch_texts
from SAMSA import SAMSA
from dataloader_utils import LRADataset

from sklearn.metrics import accuracy_score
import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import math
import numpy as np
import wandb

seed_everything(1)

parser = argparse.ArgumentParser(description="experiments")
parser.add_argument("--problem", type=str, default='retrieval_4000')
parser.add_argument("--model", type=str, default='samsa')
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--wandb", type=str, default='rusx')

args = parser.parse_args()

config = {
    'problem': 'retrieval_4000',
    'model': 'samsa',
    'search': False,
    'max_seq_len': 4000,
    'vocab_size': 256,
    'embedding_size': 240,
    'hidden_size': 240,
    'mlp_dropout': 0.,
    'layer_dropout': 0.,
    'n_class': 2,
    'n_epochs': 270,
    'positional_embedding': False,
    'embedding_type': 'sparse',
    'learning_rate': 0.0005,
    'batch_size': 32,
    'warmup_epochs': 10,
    'warmup_scale': 1e-7
}


# sys.exit()

torch.cuda.set_device(args.device_id)
device = 'cuda:{}'.format(args.device_id) if torch.cuda.is_available() else 'cpu'
print('set up device:', device)

# task variables
trainloader = DataLoader(LRADataset(f'experiments/data/LRA/retrieval_4000.train.pickle', True), batch_size=config['batch_size'], shuffle=True, drop_last=False)
testloader = DataLoader(LRADataset(f'experiments/data/LRA/retrieval_4000.test.pickle', False), batch_size=config['batch_size'], shuffle=False, drop_last=False)


naming_log = "TEXT"
if not config['search']:
    wandb.init(project="SAMSA_LRA", entity=args.wandb, name=naming_log)
    wandb.config = config
else:
    wandb.init(project="SAMSA_LRA", entity=args.wandb, config=config)
    config = wandb.config
    print('CONFIG')
    print(config)


class RetrievalSAMSA(SAMSA):
    def forward(self, data, lengths=None):
        if lengths:
            # variable lengths mode
            n_layers = math.ceil(math.log(lengths[0], 2))
        else:
            # equal lengths mode
            n_layers = self.max_n_layers

        data = self.embedding(data)
        
        if self.positional_embedding and not lengths:
            positions = torch.arange(0, self.max_seq_len).expand(data.size(0), self.max_seq_len)
            positions = positions.to(data.device)
            pos_embed = self.pos_embedding(positions)
            data = data + pos_embed
        
        for layer in range(n_layers):
            data = self.binarymixer_blocks[layer](data, layer, lengths)
        
        # sequence-aware average pooling
        if lengths:
            data = [torch.mean(t, dim=0) for t in torch.split(data, lengths)]
            data = torch.stack(data)
        else:
            data = torch.mean(data, dim=1)
            
        # No pooling
        return data
    
    
net = RetrievalSAMSA(
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


class NetDual(nn.Module):
    def __init__(self, model_part, dim, n_class):
        super(NetDual, self).__init__()
        self.model = model_part
        self.linear = nn.Linear(dim*4, n_class)

    def forward(self, x1, x2):
        y_dim1 = self.model(x1)
        y_dim2 = self.model(x2)
        y_class = torch.cat([y_dim1, y_dim2, y_dim1 * y_dim2, y_dim1 - y_dim2], dim=1)
        y = self.linear(y_class)
        return y

net = NetDual(net, config['embedding_size'], config['n_class'])

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


for epoch in range(config['n_epochs']):
    print(f'Starting epoch {epoch+1}')
    train_epoch_texts(config, net, optimizer, loss, trainloader, device=device, log_every=10000)
    acc = eval_model_texts(config, net, testloader, metric=accuracy_score, device=device)
    print(f'Epoch {epoch+1} completed. Test accuracy: {acc}')
        
    # torch.save(net.state_dict(), f'epoch_{epoch+1}_test_{acc:.3f}.pt')