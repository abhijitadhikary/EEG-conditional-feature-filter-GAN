import torch
from torch.utils.data import Dataset
import numpy as np

class CreateDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(self.convert(features, -1, 1), dtype=torch.float32, requires_grad=False)
        self.labels = torch.tensor(labels, dtype=torch.int64, requires_grad=False)
        self.num_samples = len(self.features)
        self.num_classes = 8

    # def cat_con_feature(self, feature, label):
    #     num_channels, height, width = feature.shape
    #
    #     feature_con = torch.ones((num_channels + self.num_classes, height, width), dtype=torch.float32)
    #     feature_con[:num_channels] = feature
    #
    #     for index in range(self.num_classes):
    #         if index == label.item():
    #             label_multiplier = 0.9
    #         else:
    #             label_multiplier = 0.1
    #         current_channel = torch.ones((height, width)) * label_multiplier
    #         feature_con[index+num_channels] = current_channel
    #
    #     return feature_con


    def get_one_hot_label(self, label):
        label_one_hot = np.zeros(self.num_classes)
        label_one_hot[label] = 1
        return label_one_hot

    def get_conditioned_item(self, index):
        feature = self.features[index]
        label = self.labels[index]
        # label_one_hot = self.get_one_hot_label(label)
        # feature_con = self.cat_con_feature(feature, label)

        return feature, label

    # a function to get items by index
    def __getitem__(self, index_A):
        '''
            for any given index_A, randomly choose another index_B and return the corresponding items
        '''
        # TODO add better randomisation, better sampling method
        index_B = np.random.randint(self.num_samples)

        feature_A, label_A = self.get_conditioned_item(index_A)
        feature_B, label_B = self.get_conditioned_item(index_B)

        item = {}
        item.update({
            'A': feature_A,
            'A_label': label_A
        })

        item.update({
            'B': feature_B,
            'B_label': label_B
        })

        return item

    # a function to count samples
    def __len__(self):
        return self.num_samples

    def convert(self, source, min_value=-1, max_value=1):
        smin = source.min()
        smax = source.max()

        a = (max_value - min_value) / (smax - smin)
        b = max_value - a * smax
        target = (a * source + b).astype(source.dtype)

        return target