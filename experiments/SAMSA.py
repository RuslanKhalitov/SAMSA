import sys
import math
import torch
import numpy as np
from torch import nn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Rotate(nn.Module):
    """
    Parameter-free module to perform tracks shift.
    """
    def __init__(self):
        super(Rotate, self).__init__()

    def forward(self, x, shift, lengths=None):
        if not lengths:
            B, seq_len, emb_size = x.shape

            # roll sequences in a batch jointly
            x[:, :, emb_size//2:] = torch.roll(x[:, :, emb_size//2:], shifts=shift, dims=1)
        else:
            seq_len, emb_size = x.shape
            B = len(lengths)
            
            assert seq_len == sum(lengths), f'input lengths {sum(lengths)} and tensor {seq_len} dont match'
            # roll sequences separately
            counter = 0
            for seq_len in lengths:
                x[counter:(counter+seq_len), emb_size//2:] = \
                    torch.roll(x[counter:(counter+seq_len), emb_size//2:], shifts=shift, dims=0)
                counter += seq_len
        return x

class RotateFast(nn.Module):
    """
    Parameter-free module to perform tracks shift.
    torch.roll is faster with long sequences
    """
    def __init__(self):
        super(RotateFast, self).__init__()

    def forward(self, x, shift, lengths=None):
        if not lengths:
            B, seq_len, emb_size = x.shape

            # Roll sequences in a batch jointly
            
            # Split on tracks
            y = torch.split(
                tensor=x,
                split_size_or_sections=emb_size//3,
                dim=-1
            )
            z = torch.cat([
                    y[0],
                    torch.roll(y[1], shifts=shift, dims=1),
                    torch.roll(y[2], shifts=-shift, dims=1)
                ], -1
            )
        else:
            seq_len, emb_size = x.shape
            
            ys = torch.split(
                tensor=x,
                split_size_or_sections=lengths,
                dim=0
            )
            zs = []
            # roll sequences separately
            for y in ys:
                y = torch.split(
                    tensor=y,
                    split_size_or_sections=emb_size//3,
                    dim=-1
                )
                zs.append(
                    torch.cat([
                        y[0],
                        torch.roll(y[1], shifts=shift, dims=0),
                        torch.roll(y[2], shifts=-shift, dims=0)
                        ], -1
                    )
                )
            z = torch.cat(zs, 0)
        return z


class SAMSABlock(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        mlp_dropout,
        layer_dropout
    ):
        super(SAMSABlock, self).__init__()

        self.mixer = Mlp(
            embedding_size,
            hidden_size,
            embedding_size,
            act_layer=nn.GELU,
            drop=mlp_dropout
        )

        self.dropout = nn.Dropout(layer_dropout)

        self.rotator = RotateFast()

    def forward(self, data, layer, lengths=None):
        res_con = data
        data = self.mixer(data)
        data = self.dropout(data)
        data = self.rotator(data, 2**(layer+1), lengths)
        data = data + res_con
        return data


class SAMSA(nn.Module):
    def __init__(self,
        problem,
        vocab_size,
        max_seq_len,
        embedding_type,
        embedding_size,
        positional_embedding,
        hidden_size,
        mlp_dropout,
        layer_dropout,
        n_class
        ):
            super(SAMSA, self).__init__()
            assert embedding_size % 3 == 0, f"Embedding size should be divisible by 3, embedding_size={embedding_size}"
            self.max_seq_len = max_seq_len
            self.max_n_layers = math.ceil(math.log(max_seq_len, 2))
            self.embedding_type = embedding_type
            self.positional_embedding = positional_embedding
            
            self.pos_embedding = nn.Embedding(
                    max_seq_len,
                    embedding_size
                )
            
            # Init embedding layer
            if embedding_type == 'sparse':
                self.embedding = nn.Embedding(
                    vocab_size,
                    embedding_size,
                    padding_idx=0
                )
            elif embedding_type == 'linear':
                in_ch = 2 if problem == 'adding' else 1
                self.embedding = nn.Linear(
                    in_ch,
                    embedding_size
                )
            else:
                print(f'Unknown embedding type — {embedding_type}. Should be one of: sparse, linear.')
                sys.exit()

            self.binarymixer_blocks = nn.ModuleList(
                [
                    SAMSABlock(
                        embedding_size,
                        hidden_size,
                        mlp_dropout,
                        layer_dropout
                    )
                    for _ in range(self.max_n_layers)
                ]
            )

            self.final =  nn.Linear(
                embedding_size,
                n_class
            )

    def forward(self, data, lengths=None):
        if lengths:
            # variable lengths mode
            n_layers = math.ceil(math.log(lengths[0], 2))
        else:
            # equal lengths mode
            n_layers = self.max_n_layers
            
        
        if self.embedding_type == 'sparse':
            data = self.embedding(data)
        elif self.embedding_type == 'linear':
            # data = data.unsqueeze(-1)
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
        data = self.final(data)
        return data