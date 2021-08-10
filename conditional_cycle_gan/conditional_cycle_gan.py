import torch
import argparse
import numpy as np
import os
import scipy.io as sio
import time
# from utils.visualizer import Visualizer
from conditional_cycle_gan.create_dataset import CreateDataset
from conditional_cycle_gan.train_options import TrainOptions
from conditional_cycle_gan.conditional_cyclegan_model import ConditionalCycleGANModel
from torch.utils.tensorboard import SummaryWriter
from conditional_cycle_gan.create_dataloader import CreateDataLoader
# from conditional_cycle_gan.

class ConditionalCycleGAN:
    def __init__(self):
        self.args = argparse.Namespace()
        self.args.seed = 1324
        self.args.batch_size = 20
        self.args.split_mode = 'within'
        self.args.dataset_path = os.path.join('datasets', 'eeg')
        self.create_dataloaders()
        # self.create_model()

    def load_mat(self, filename):
        return sio.loadmat(os.path.join(self.args.dataset_path, filename))

    def create_dataloaders(self):
        self.num_classes = 8
        modes = ['train', 'val', 'test']
        datasets = []
        for mode in modes:
            filename = f'eeg_conditional_images_{mode}_{self.args.split_mode}.mat'
            data = self.load_mat(filename)

            images = data['images']
            images = np.transpose(images, (0, 3, 1, 2))

            # labels = data['labels'][:, :3] # take one_by_three labels
            labels = data['labels'][:, -1]  # take 0-7 labels

            dataset = CreateDataset(images, labels)
            datasets.append(dataset)

        # features, labels, features_val, labels_val, features_test, labels_test = self.process_conditional_features(data)

        # create dataloders
        self.dataloader_train = torch.utils.data.DataLoader(datasets[0], batch_size=self.args.batch_size, shuffle=True)

        self.dataloader_val = torch.utils.data.DataLoader(datasets[1], batch_size=self.args.batch_size, shuffle=True)

        self.dataloader_test = torch.utils.data.DataLoader(datasets[2], batch_size=self.args.batch_size, shuffle=True)

    def create_model(self, opt):
        model = None
        model = ConditionalCycleGANModel()
        model.initialize(opt)
        print(f'model [{model.name()}] was created')
        return model

    def create_dirs(self, dir_list):
        for current_dir in dir_list:
            current_path = current_dir[0]
            if len(current_dir) > 1:
                for sub_dir_index in range(1, len(current_dir)):
                    current_path = os.path.join(current_path, current_dir[sub_dir_index])
            if not os.path.exists(current_path):
                os.makedirs(current_path)

    def create_tensorboard(self, opt):
        writer_path = os.path.join('conditional_cycle_gan', 'runs', f'{opt.name}')
        self.create_dirs([writer_path])
        self.writer = SummaryWriter(writer_path)

    def print_current_losses(self, epoch, index_batch, index_step, losses):
        message = f'(Epoch: {epoch}, Batch: {index_batch}, Step: {index_step}'
        for k, v in losses.items():
            message += f'  {k}: {v:.3f}'
        print(message)

    def update_tensorboard(self, losses, index_epoch):
        self.writer.add_scalars('Loss', losses, index_epoch + 1)

    def train(self):
        opt = TrainOptions().parse()
        dataloader = self.dataloader_train
        model = self.create_model(opt)
        model.setup(opt)
        self.create_tensorboard(opt)

        index_step = 0
        for index_epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            for index_batch, data in enumerate(dataloader):
                model.set_input(data)
                model.optimize_parameters()

                losses = model.get_current_losses()
                self.print_current_losses(index_epoch, index_batch, index_step, losses)
                self.update_tensorboard(losses, index_batch)
                index_step += 1

            if index_epoch % opt.save_epoch_freq == 0:
                print(f'Saving at epoch: {index_epoch}')
                # model.save_networks('latest')
                # model.save_networks(index_epoch)

            print('End of index_epoch %d / %d \t Time Taken: %d sec' % (index_epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()
