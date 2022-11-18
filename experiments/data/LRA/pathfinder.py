import torch

for ds in ['test', 'train', 'valid']:
    label = []
    sequence = []
    lens = []
    split = []
    print(f'starting {ds}')
    with open(f'lra/pathfinder/input/{ds}.src', 'r') as sf,\
         open(f'lra/pathfinder/label/{ds}.label', 'r') as lf:
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
    torch.save(tt, f'pathfinder_{ds}.pt')
    torch.save(tl, f'pathfinder_labels_{ds}.pt')