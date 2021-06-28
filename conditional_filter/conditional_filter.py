import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.utils import create_dirs
from conditional_filter.forward_pass import forward_pass
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
        self.create_dataloaders()
        self.create_model()
        self.set_device()
        self.set_model_options()
        self.args.checkpoint_path = os.path.join('.', 'conditional_filter', 'checkpoints', self.args.model_name, self.args.split_variant)
        self.create_dirs()
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
            ['.', 'conditional_filter', 'runs', self.args.model_name, self.args.split_variant],
            ['.', 'output', self.args.model_name, self.args.split_variant]
        ]
        create_dirs(dir_list)

    def set_device(self):
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_D_cls = torch.nn.DataParallel(self.model_D_cls)
        self.model_D_adv = torch.nn.DataParallel(self.model_D_adv)
        self.model_G = torch.nn.DataParallel(self.model_G)

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
            self.model_D_cls = Discriminator_cls()
            self.model_D_adv = Discriminator_adv()
            self.model_G = UNet()
        else:
            raise NotImplementedError(f'model_name [{self.args.model_name}] not implemented.')

    def set_model_options(self):
        self.criterion_D_cls = nn.BCELoss()
        self.criterion_D_adv = nn.BCELoss()
        self.criterion_G = nn.L1Loss()

        self.criterion_D_alc = nn.L1Loss()
        self.criterion_D_stm = nn.L1Loss()
        self.criterion_D_id = nn.L1Loss()

        if self.args.optimizer_variant == 'SGD':
            self.optimizer_D_cls = optim.SGD(self.model_D_cls.parameters(), lr=self.args.learning_rate_cls,
                                                  momentum=self.args.momentum,
                                                  weight_decay=self.args.weight_decay)
            self.optimizer_D_adv = optim.SGD(self.model_D_adv.parameters(), lr=self.args.learning_rate_adv,
                                                  momentum=self.args.momentum,
                                                  weight_decay=self.args.weight_decay)
            self.optimizer_G = optim.SGD(self.model_G.parameters(), lr=self.args.learning_rate_G,
                                              momentum=self.args.momentum,
                                              weight_decay=self.args.weight_decay)
        elif self.args.optimizer_variant == 'Adam':
            self.optimizer_D_cls = optim.Adam(self.model_D_cls.parameters(), lr=self.args.learning_rate_cls,
                                                   betas=(0.5, 0.999))
            self.optimizer_D_adv = optim.Adam(self.model_D_adv.parameters(), lr=self.args.learning_rate_adv,
                                                   betas=(0.5, 0.999))
            self.optimizer_G = optim.Adam(self.model_G.parameters(), lr=self.args.learning_rate_G,
                                               betas=(0.5, 0.999))
        else:
            raise NotImplementedError(f'Invalid optimizer_variant selected: {self.args.optimizer_variant}')

        self.args.writer = SummaryWriter(os.path.join('conditional_filter', 'runs', self.args.model_name, self.args.split_variant))

    def remove_previous_checkpoints(self):
        all_files = sorted(os.listdir(self.args.checkpoint_path))
        if len(all_files) >= self.args.num_keep_best:
            current_full_path = os.path.join(self.args.checkpoint_path, all_files[0])
            os.remove(current_full_path)

    def save_model(self, accuracy, loss):
        if self.args.mode == 'val':
            save_condition = False
            if self.args.checkpoint_mode == 'accuracy':
                if accuracy > self.args.accuracy_best:
                    save_condition = True
                    self.args.accuracy_best = accuracy
            if self.args.checkpoint_mode == 'loss':
                if loss < self.args.loss_best:
                    save_condition = True
                    self.args.loss_best = loss

            if save_condition:
                self.remove_previous_checkpoints()
                save_path = os.path.join(self.args.checkpoint_path, f'{self.args.index_epoch+1}.pth')
                save_dict = {'args': self.args,
                             'criterion': self.criterion,
                             'model_state_dict': self.model.state_dict(),
                             'optim_state_dict': self.optimizer.state_dict()
                             }
                torch.save(save_dict, save_path)
                print(f'New best model saved at epoch: {self.args.index_epoch+1}')

    def save_model(self, loss_D_cls, loss_D_adv, loss_G):
        '''
            need to update this to accommodate D loss
        '''

        if loss_G < self.args.loss_G_best and self.args.save_condition:
            self.args.loss_G_best = loss_G

            self.remove_previous_checkpoints()

            save_path = os.path.join(self.args.checkpoint_path, f'{self.args.index_epoch+1}.pth')
            save_dict = {'args': self.args,
                         'G_state_dict': self.model_G.state_dict(),
                         'G_optim_dict': self.optimizer_G.state_dict(),
                         'D_cls_state_dict': self.model_D_cls.state_dict(),
                         'D_cls_optim_dict': self.optimizer_D_cls.state_dict(),
                         'D_adv_state_dict': self.model_D_adv.state_dict(),
                         'D_adv_optim_dict': self.optimizer_D_adv.state_dict()
                         }
            torch.save(save_dict, save_path)
            print(f'*********************** New best model saved at {self.args.index_epoch + 1} ***********************')

    def load_model(self):
        load_path = os.path.join(self.args.checkpoint_path, f'{self.args.resume_epoch}.pth')

        if load_path is not None:
            if not os.path.exists(load_path):
                raise FileNotFoundError(f'File {load_path} doesn\'t exist')

            checkpoint = torch.load(load_path)

            self.args = checkpoint['args']
            self.model_D_cls.load_state_dict(checkpoint['D_cls_state_dict'])
            self.optimizer_D_cls.load_state_dict(checkpoint['D_cls_optim_dict'])
            self.model_D_adv.load_state_dict(checkpoint['D_adv_state_dict'])
            self.optimizer_D_adv.load_state_dict(checkpoint['D_adv_optim_dict'])
            self.model_G.load_state_dict(checkpoint['G_state_dict'])
            self.optimizer_G.load_state_dict(checkpoint['G_optim_dict'])

            print(f'Model successfully loaded from epoch {self.args.resume_epoch}')
            self.args.start_epoch = self.args.index_epoch + 1

    def run(self):
        for index_epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.args.index_epoch = index_epoch

            # train
            loss_D_cls_epoch_train, loss_D_adv_epoch_train, loss_D_total_epoch_train, loss_G_epoch_train, \
            D_cls_conf_real_epoch_train, D_adv_conf_real_epoch_train, D_cls_conf_fake_epoch_train, D_adv_conf_fake_epoch_train, \
                = forward_pass(self.args, self.dataloader_train, mode='train')
            self.args.loss_D_cls_train_running.append(loss_D_cls_epoch_train)
            self.args.loss_D_adv_train_running.append(loss_D_adv_epoch_train)
            self.args.loss_D_total_train_running.append(loss_D_total_epoch_train)
            self.args.loss_G_train_running.append(loss_G_epoch_train)

            # validate
            loss_D_cls_epoch_val, loss_D_adv_epoch_val, loss_D_total_epoch_val, loss_G_epoch_val, \
            D_cls_conf_real_epoch_val, D_adv_conf_real_epoch_val, D_cls_conf_fake_epoch_val, D_adv_conf_fake_epoch_val \
                = forward_pass(self.args, self.dataloader_val, mode='val')
            self.args.loss_D_cls_val_running.append(loss_D_cls_epoch_val)
            self.args.loss_D_adv_val_running.append(loss_D_adv_epoch_val)
            self.args.loss_D_total_val_running.append(loss_D_total_epoch_val)
            self.args.loss_G_val_running.append(loss_G_epoch_val)

            self.args.writer.add_scalars('Loss', {
                'train_D_cls': loss_D_cls_epoch_train,
                'train_D_adv': loss_D_adv_epoch_train,
                'val_D_cls': loss_D_cls_epoch_val,
                'val_D_adv': loss_D_adv_epoch_val,
                'train_G': loss_G_epoch_train,
                'val_G': loss_G_epoch_val
            }, index_epoch + 1)

            self.args.writer.add_scalars('Confidence_train', {
                'D_cls_real': D_cls_conf_real_epoch_train,
                'D_cls_fake': D_cls_conf_fake_epoch_train,
                'D_adv_real': D_adv_conf_real_epoch_train,
                'D_adv_fake': D_adv_conf_fake_epoch_train,
            }, index_epoch + 1)

            self.args.writer.add_scalars('Confidence_val', {
                'D_cls_real': D_cls_conf_real_epoch_val,
                'D_cls_fake': D_cls_conf_fake_epoch_val,
                'D_adv_real': D_adv_conf_real_epoch_val,
                'D_adv_fake': D_adv_conf_fake_epoch_val,
            }, index_epoch + 1)

            # need to update this function to accommodate loss_D_adv
            self.save_model(loss_D_cls_epoch_val, loss_D_adv_epoch_val, loss_G_epoch_val)