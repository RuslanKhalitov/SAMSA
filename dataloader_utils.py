
import pandas as pd
import numpy as np
import random
import pickle
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence



class DatasetCreator(Dataset):
    def __init__(self, df, batch_size, var_len=False):
        self.var_len = var_len
        if self.var_len:
            # fill in gaps to form full batches
            df = complete_batch(df=df, batch_size=batch_size)
            # shuffle batches
            self.df = shuffle_batches(df=df)[['sequence', 'label', 'len', 'bin']]
        else:
            self.df = df[['sequence', 'label']]
            
    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        if self.var_len:
            X, Y, length, bin = self.df.iloc[index, :]
            Y = torch.tensor(Y)
            X = torch.from_numpy(X)
            return (X, Y, length, bin)
        else:
            X, Y = self.df.iloc[index, :]
            Y = torch.tensor(Y)
            X = torch.from_numpy(X)
            return (X, Y)

    def __len__(self):
        return len(self.df)

class DatasetCreatorFlat(Dataset):
    def __init__(self, df, labels):
        self.df = df
        self.labels = labels
            
    def __getitem__(self, index):
        """
        Returns: tuple (sample, target)
        """
        X = self.df[index]
        Y = self.labels[index]
        return (X, Y)

    def __len__(self):
        return self.df.size(0)

def complete_batch(df, batch_size):
    """
    Function to make number of instances divisible by batch_size
    within each log2-bin
    """
    complete_bins = []
    bins = [bin_df for _, bin_df in df.groupby('bin')]

    for gr_id, bin in enumerate(bins):
        l = len(bin)
        remainder = l % batch_size
        integer = l // batch_size

        if remainder != 0:
            # take the first example and copy (batch_size - remainder) times
            bin = pd.concat([bin, pd.concat([bin.iloc[:1]]*(batch_size - remainder))], ignore_index=True)
            integer += 1
        batch_ids = []
        # create indices 
        for i in range(integer):
            batch_ids.extend([f'{i}_bin{gr_id}'] * batch_size)
        bin['batch_id'] = batch_ids
        complete_bins.append(bin)
    return pd.concat(complete_bins, ignore_index=True)

def shuffle_batches(df):
    """
    Shuffles batches so during training 
    A model sees sequences from different log2-bins
    """
    import random

    batch_bins = [df_new for _, df_new in df.groupby('batch_id')]
    random.shuffle(batch_bins)

    return pd.concat(batch_bins).reset_index(drop=True)


def concater_collate(batch):
    """
    Packs a batch into a long sequence
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx = torch.cat(xx, 0)
    yy = torch.tensor(yy)
    return xx, yy, list(lengths), list(bins)

def pad_collate(batch):
    """
    PAD sequences in a batch
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.tensor(yy)
    return xx_pad, yy, list(lengths), list(bins)

def pad_collate_image(batch):
    """
    PAD sequences in a batch
    """
    (xx, yy, lengths, bins) = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
    yy = torch.tensor(yy)
    return xx_pad, yy

# Refer to https://github.com/mlpen/Nystromformer/blob/main/LRA/code/dataset.py
class LRADataset(Dataset):
    def __init__(self, file_path, endless):

        self.endless = endless
        with open(file_path, 'rb') as f:
            self.examples = pickle.load(f)
            random.shuffle(self.examples)
            self.curr_idx = 0

        print(f'Loaded {file_path}... size={len(self.examples)}', flush=True)

    def __len__(self):
        return len(self.examples)

    def create_inst(self, inst):
        output = {}
        output['input_ids_0'] = torch.tensor(inst['input_ids_0'], dtype=torch.long)
        output['mask_0'] = (output['input_ids_0'] != 0).float()
        if 'input_ids_1' in inst:
            output['input_ids_1'] = torch.tensor(inst['input_ids_1'], dtype=torch.long)
            output['mask_1'] = (output['input_ids_1'] != 0).float()
        output['label'] = torch.tensor(inst['label'], dtype=torch.long)
        return output

    def __getitem__(self, i):
        if not self.endless:
            return self.create_inst(self.examples[i])

        if self.curr_idx >= len(self.examples):
            random.shuffle(self.examples)
            self.curr_idx = 0
        inst = self.examples[self.curr_idx]
        self.curr_idx += 1

        return self.create_inst(inst)