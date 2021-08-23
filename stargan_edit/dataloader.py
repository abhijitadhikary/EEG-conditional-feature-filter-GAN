from stargan_edit.utils import load_mat
from stargan_edit.utils import convert
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

def get_dataloader(opt, mode_run):
    filename = f'eeg_conditional_images_{mode_run}_{opt.mode_split}.mat'
    data = load_mat(os.path.join(opt.path_dataset, filename))
    images = data['images']
    images = np.transpose(images, (0, 3, 1, 2))
    labels = data['labels'][:, -1]  # take 0-7 labels
    dataset = CreateDataset(opt, images, labels)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.num_workers, shuffle=opt.shuffle_dataloaders)
    return dataloader

# random
class CreateDataset(Dataset):
    def __init__(self, opt, features, labels):
        features = convert(features, opt.min_value_feature, opt.max_value_feature)
        self.features = torch.tensor(features, dtype=torch.float32, requires_grad=False)
        self.labels = torch.tensor(labels, dtype=torch.int64, requires_grad=False)
        self.num_samples = len(self.features)
        self.num_classes = opt.num_classes

    def get_conditioned_item(self, index):
        feature = self.features[index]
        label = self.labels[index]
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

# half-half
# class CreateDataset(Dataset):
#     def __init__(self, opt, features, labels):
#         # make even indices
#         num_samples = len(features)
#         num_samples_half = num_samples // 2
#
#         # randomly shuffle features and labels
#         indices = np.arange(num_samples)
#         np.random.shuffle(indices)
#         features = features[indices]
#         labels = labels[indices]
#
#         features = convert(features, opt.min_value_feature, opt.max_value_feature)
#
#         self.features_A = torch.tensor(features[:num_samples_half], dtype=torch.float32)
#         self.features_B = torch.tensor(features[num_samples_half:], dtype=torch.float32)
#         self.labels_A = torch.tensor(labels[:num_samples_half], dtype=torch.int64)
#         self.labels_B = torch.tensor(labels[num_samples_half:], dtype=torch.int64)
#
#         self.num_classes = opt.num_classes
#         self.num_samples = num_samples_half
#
#     # a function to get items by index
#     def __getitem__(self, index):
#         '''
#             for any given index_A, randomly choose another index_B and return the corresponding items
#         '''
#         # TODO add better randomisation, better sampling method
#
#         item = {}
#         item.update({
#             'A': self.features_A[index],
#             'A_label': self.labels_A[index]
#         })
#
#         item.update({
#             'B': self.features_B[index],
#             'B_label': self.labels_B[index]
#         })
#
#         return item
#
#     # a function to count samples
#     def __len__(self):
#         return self.num_samples