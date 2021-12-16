import numpy as np
import torch
from torch.utils.data import Dataset
import os

class EyeDataset(Dataset):
    def __init__(self, npy_path, label_path):
        self.all_data = np.load(npy_path)
        self.label_list = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                self.label_list.append(line)

    def __getitem__(self, index):
        return self.all_data[index], self.label_list[index]

    def __len__(self):
        return len(self.label_list)

