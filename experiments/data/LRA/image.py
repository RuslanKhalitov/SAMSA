import torch

for ds in ['test', 'train', 'valid']:
    label = []
    sequence = []
    lens = []
    split = []
    print(f'starting {ds}')
    with open(f'lra/cifar10/input/{ds}.src', 'r') as sf,\
         open(f'lra/cifar10/label/{ds}.label', 'r') as lf:
        for i in sf:
            seq_str = i.split()
            seq_str = [int(j) for j in seq_str]
            sequence.append(seq_str)
            lens.append(len(seq_str))
            split.append(ds)
            
        for l in lf:
            label.append(int(l))
            
    tt = torch.tensor(sequence)
    tl = torch.tensor(label)
    torch.save(tt, f'cifar10_{ds}.pt')
    torch.save(tl, f'cifar10_labels_{ds}.pt')