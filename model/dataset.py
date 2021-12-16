import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random

def pad_tensor(vec, pad, dim=0):
    vec = torch.tensor(vec)
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
def pad_collate(batch):
    # find longest sequence
    max_len = max(map(lambda x: x[0].shape[0], batch))
    # pad according to max_len
    batch = list(map(lambda x_y:
                (pad_tensor(x_y[0], pad=max_len, dim=0), x_y[1]), batch))
    # stack all
    xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0).cuda()
    ys = torch.LongTensor(list(map(lambda x: x[1], batch))).cuda()
    return xs, ys

class EyeDataset(Dataset):
    def __init__(self, npy_path, label_path):
        self.all_data = np.load(npy_path)
        with open(label_path, 'r') as f:
            self.label_list = f.readlines()[0]  # '1101101001'

    def __getitem__(self, index):
        while (self.all_data[index].shape[1] != 23):#TODO: find where is the error file
            index = random.randint(0, len(self.label_list) - 1)  # if error, resample
        return self.all_data[index], self.label_list[index]

    def __len__(self):
        return len(self.label_list)

