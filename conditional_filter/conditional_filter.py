import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import create_dirs
from conditional_filter.create_datasest import CreateDataset
from conditional_filter.discriminator import Discriminator_cls, Discriminator_adv
from conditional_filter.generator import UNet

class ConditionalFilter():
    def __init__(self):
        self.args = argparse.Namespace()
        self.args.seed = 1324
        self.set_seed()
        self.set_hyperparameters()
        self.args.dataset_path = os.path.join('.', 'datasets', 'eeg')
        # self.args.accuracy_best = np.NINF
        # self.args.loss_best = np.Inf
        self.args.num_keep_best = 5
        self.resume_epoch = 8
        self.resume_condition = False
        self.args.checkpoint_mode = 'accuracy'  # accuracy, loss
        self.args.checkpoint_path = os.path.join('.', 'conditional_filter', 'checkpoints')
        self.create_dataloaders()
        self.create_model()
        self.create_dirs()
        self.set_device()
        self.set_model_options()
        # self.load_model()

    def set_hyperparameters(self):
        self.args.model_name = 'baseline'  # baseline
        self.args.split_variant = 'within'
        self.args.batch_size = 784
        self.args.learning_rate_adv = 0.0002
        self.args.learning_rate_cls = 0.0002
        self.args.learning_rate_G = 0.0002
        self.args.momentum = 0.9
        self.args.weight_decay = 5e-4
        self.args.optimizer_variant = 'SGD'  # SGD, Adam
        self.args.start_epoch = 0
        self.args.num_epochs = 200
        self.args.shuffle_train = False
        self.args.shuffle_test = False
        self.args.shuffle_val = False
        self.args.batch_size_train = 192
        self.args.batch_size_test = 192
        self.args.batch_size_val = 192
        self.args.loss_D_cls_factor = 10
        self.args.loss_D_adv_factor = 1
        self.args.loss_D_total_factor = 1
        self.args.loss_G_gan_factor = 10
        self.args.loss_G_l1_factor = 30

    def create_dirs(self):
        dir_list = [
            ['.', 'conditional_filter'],
            ['.', 'conditional_filter', 'checkpoints'],
            ['.', 'conditional_filter', 'checkpoints', self.args.model_name, self.args.split_variant],
            ['.', 'conditional_filter', 'runs', self.args.model_name, self.args.split_variant]
        ]
        create_dirs(dir_list)

    def set_device(self):
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.args.model_D_cls = torch.nn.DataParallel(self.args.model_D_cls)
        self.args.model_D_adv = torch.nn.DataParallel(self.args.model_D_adv)
        self.args.model_G = torch.nn.DataParallel(self.args.model_G)

    def set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    def create_dataloaders(self):
        dataset_train = CreateDataset(self.args, 'train')
        dataset_test = CreateDataset(self.args, 'test')
        dataset_val = CreateDataset(self.args, 'val')

        # create dataloders
        self.dataloader_train = DataLoader(dataset_train, batch_size=self.args.batch_size_train,
                                           shuffle=self.args.shuffle_train)
        self.dataloader_test = DataLoader(dataset_test, batch_size=self.args.batch_size_test,
                                          shuffle=self.args.shuffle_test)
        self.dataloader_val = DataLoader(dataset_val, batch_size=self.args.batch_size_val,
                                         shuffle=self.args.shuffle_val)

    def create_model(self):
        if self.args.model_name == 'baseline':
            self.args.model_D_cls = Discriminator_cls()
            self.args.model_D_adv = Discriminator_adv()
            self.args.model_G = UNet()
        else:
            raise NotImplementedError(f'model_name [{self.args.model_name}] not implemented.')

    def set_model_options(self):
        self.args.criterion_D_cls = nn.BCELoss()
        self.args.criterion_D_adv = nn.BCELoss()
        self.args.criterion_G = nn.L1Loss()

        self.args.criterion_D_alc = nn.L1Loss()
        self.args.criterion_D_stm = nn.L1Loss()
        self.args.criterion_D_id = nn.L1Loss()

        if self.args.optimizer_variant == 'SGD':
            self.args.optimizer_D_cls = optim.SGD(self.args.model_D_cls.parameters(), lr=self.args.learning_rate_cls,
                                                  momentum=self.args.momentum,
                                                  weight_decay=self.args.weight_decay)
            self.args.optimizer_D_adv = optim.SGD(self.args.model_D_adv.parameters(), lr=self.args.learning_rate_adv,
                                                  momentum=self.args.momentum,
                                                  weight_decay=self.args.weight_decay)
            self.args.optimizer_G = optim.SGD(self.args.model_G.parameters(), lr=self.args.learning_rate_G,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay)
        elif self.args.optimizer_variant == 'Adam':
            self.args.optimizer_D_cls = optim.Adam(self.args.model_D_cls.parameters(), lr=self.args.learning_rate_cls,
                                                   betas=(0.5, 0.999))
            self.args.optimizer_D_adv = optim.Adam(self.args.model_D_adv.parameters(), lr=self.args.learning_rate_adv,
                                                   betas=(0.5, 0.999))
            self.args.optimizer_G = optim.Adam(self.args.model_G.parameters(), lr=self.args.learning_rate_G,
                                               betas=(0.5, 0.999))
        else:
            raise NotImplementedError(f'Invalid optimizer_variant selected: {self.args.optimizer_variant}')

        self.args.writer = SummaryWriter(os.path.join('conditional_filter', 'runs', self.args.model_name, self.args.split_variant))