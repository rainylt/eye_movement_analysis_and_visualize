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
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
         # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.label_list[idx]
    def __iter__(self):                                                                                                                                                                            return (self.indices[i] for i in torch.multinomial(
        self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

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

