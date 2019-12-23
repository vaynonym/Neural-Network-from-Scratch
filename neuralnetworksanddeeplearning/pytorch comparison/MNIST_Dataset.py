import numpy as np
import torch
from torch.utils.data import Dataset

class MNIST_Dataset(Dataset):
    def __init__(self, data, transform = None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        if self.transform:
            sample = (self.transform(sample[0].reshape([28,28])).reshape([28*28]), sample[1])
        return sample
