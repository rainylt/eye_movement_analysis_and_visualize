import numpy as np
import torch
from torch.utils.data import Dataset
import os

def pad_tensor(vec, pad, dim=0):
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
def pad_collate(batch):
    # find longest sequence
    max_len = max(map(lambda x: x[0].shape[0], batch))
    # pad according to max_len
    batch = map(lambda x_y:
                (pad_tensor(x_y[0], pad=max_len, dim=0), x_y[1]), batch)
    # stack all
    xs = torch.stack(map(lambda x: x[0], batch), dim=0)
    ys = torch.LongTensor(map(lambda x: x[1], batch))
    return xs, ys

class EyeDataset(Dataset):
    def __init__(self, npy_path, label_path):
        self.all_data = np.load(npy_path)
        self.label_list = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                self.label_list.append(int(line.strip('\n')))

    def __getitem__(self, index):
        return self.all_data[index], self.label_list[index]

    def __len__(self):
        return len(self.label_list)

