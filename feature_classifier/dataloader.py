import torch
from torch.utils.data import Dataset
import numpy as np

class CreateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.Tensor(features)
        self.labels = torch.Tensor(labels)

    # a function to get items by index
    def __getitem__(self, index):
        feature_current = self.features[index]
        label_current = self.labels[index]
        return feature_current, label_current

    # a function to count samples
    def __len__(self):
        n = np.shape(self.features)[0]
        return n