

import sys
from input_pipeline_listops import get_datasets
import numpy as np
import pickle
import pandas as pd

train_ds, eval_ds, test_ds, encoder = get_datasets(
    n_devices = 1, task_name = "basic", data_dir = "./lra_release/lra_release/listops-1000/",
    batch_size = 1, max_length = 2000)

mapping = {"train":train_ds, "val": eval_ds, "test":test_ds}
for component in mapping:
    print(component)
    sequences = []
    labels = []
    for idx, inst in enumerate(iter(mapping[component])):
        sequence = inst["inputs"].numpy()[0]
        label = inst["targets"].numpy()[0]
        sequence = sequence[sequence!=0]
        sequences.append(sequence)
        labels.append(label)
    filename = f'listops.{component}.pkl'
    df = pd.DataFrame({'sequence': sequences, 'label': labels})
    df['len'] = df['sequence'].map(lambda x: len(x))
    df['bin'] = df['len'].map(lambda x: 10 if x < 2**10 else 11)
    df.to_pickle(f'listops_{component}.pkl')