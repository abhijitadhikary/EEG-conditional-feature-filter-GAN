import torch
import os
from torch.utils.data import Dataset
import torchvision
import numpy as np
from scipy.io import loadmat

class CreateDataset(Dataset):
    def __init__(self, args, mode, transform=None):
        dataset_path = args.dataset_path
        filename = f'uci_eeg_images_{mode}_{args.split_variant}.mat'
        self.filepath_full = os.path.join(dataset_path, filename)
        data = loadmat(self.filepath_full)
        self.identity = data['label_id']
        self.stimulus = data['label_stimulus']
        self.alcoholism = data['label_alcoholism']
        self.images = data['data']
        self.num_samples = len(self.images)
        self.transform = transform
        self.toTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        identity = torch.tensor(self.identity[index], dtype=torch.float32)
        stimulus = torch.tensor(self.stimulus[index], dtype=torch.float32)
        alcoholism = torch.tensor(self.alcoholism[index], dtype=torch.float32)

        image = self.images[index]
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
        image = convert(image, 0, 1)

        if not self.transform is None:
            image = self.transform(image)

        # prepare mask for conditional generation
        image_c_real, image_c_fake, condition_array_real, condition_array_fake = get_conditioned_image(image)
        targets_real_cls = torch.cat((identity, stimulus, alcoholism)).reshape(-1, 1)
        targets_fake_cls = targets_real_cls * condition_array_fake # float
        targets_real_adv = get_adv_label('real')
        targets_fake_adv = get_adv_label('fake')
        return image, image_c_real, image_c_fake, condition_array_real, condition_array_fake, identity, stimulus, alcoholism, \
               targets_real_cls, targets_real_adv, targets_fake_cls, targets_fake_adv

def get_conditioned_image(image):
    '''
    create random conditions {0,1} for each feature and concatenate them to the image
    '''
    num_channels, height, width = image.shape
    num_features = 3

    condition_array_fake = torch.tensor([(np.random.rand(1) > 0.5).astype(np.float32) for index in range(num_features)])
    filter_identity, filter_stimulus, filter_alcoholism = ([condition_array_fake[index] * torch.ones((1, height, width), dtype=torch.float32) for index in range(3)])
    image_c_fake = torch.cat((image, filter_identity, filter_stimulus, filter_alcoholism), dim=0)

    condition_array_real = torch.ones_like(condition_array_fake)
    filter_identity, filter_stimulus, filter_alcoholism = ([condition_array_fake[index] * torch.ones((1, height, width), dtype=torch.float32) for index in range(3)])
    image_c_real = torch.cat((image, filter_identity, filter_stimulus, filter_alcoholism), dim=0)

    return image_c_real, image_c_fake, condition_array_real, condition_array_fake

def convert(source, min_value=0, max_value=1):
    smin = source.min()
    smax = source.max()
    a = (max_value - min_value) / (smax - smin)
    b = max_value - a * smax
    target = (a * source + b)
    return target

def get_adv_label(mode='real'):
    '''
        needs to be updated to random multipliers
    '''
    label = torch.ones(1, dtype=torch.float32)
    if mode == 'real':
        multiplier = 0.9
    elif mode == 'fake':
        multiplier = 0.1
    else:
        raise NotImplementedError
    label *= multiplier
    return label