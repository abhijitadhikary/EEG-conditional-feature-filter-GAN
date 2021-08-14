import torch
import argparse
import numpy as np
import os
import scipy.io as sio
import time
from torchvision.utils import make_grid, save_image
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

        # create dataloders
        self.dataloader_train = torch.utils.data.DataLoader(datasets[0], batch_size=self.args.batch_size, shuffle=True)

        self.dataloader_val = torch.utils.data.DataLoader(datasets[1], batch_size=self.args.batch_size, shuffle=True)

        self.dataloader_test = torch.utils.data.DataLoader(datasets[2], batch_size=self.args.batch_size, shuffle=True)

    def create_model(self, opt):
        self.model = None
        self.model = ConditionalCycleGANModel()
        self.model.initialize(opt)
        print(f'model [{self.model.name()}] was created')

    def create_dirs(self):
        dir_list = [
            ['.', 'conditional_cycle_gan', 'runs'],
            ['.', 'conditional_cycle_gan', 'output'],
        ]
        for current_dir in dir_list:
            current_path = current_dir[0]
            if len(current_dir) > 1:
                for sub_dir_index in range(1, len(current_dir)):
                    current_path = os.path.join(current_path, current_dir[sub_dir_index])
            if not os.path.exists(current_path):
                os.makedirs(current_path)

    def create_tensorboard(self, opt):
        writer_path = os.path.join(f'{opt.tensorboard_path}', f'{opt.name}')
        # self.create_dirs([writer_path])
        self.writer = SummaryWriter(writer_path)

    def print_current_losses(self, epoch, epoch_end, index_batch, num_batches, index_step, losses):
        message = f'(Epoch: [{epoch} / {epoch_end-1}],\tBatch: [{index_batch} / {num_batches}],\tStep: {index_step}\n'
        for k, v in losses.items():
            message += f'\t{k}: {v:.3f}'
        print(message)

    def update_tensorboard(self, losses, index):
        self.writer.add_scalars('Loss', losses, index + 1)

    def convert(self, source, min_value=-1, max_value=1):
        smin = source.min()
        smax = source.max()

        a = (max_value - min_value) / (smax - smin)
        b = max_value - a * smax
        target = (a * source + b)

        return target

    def train(self):
        self.create_dirs()
        opt = TrainOptions().parse()
        dataloader = self.dataloader_train
        self.create_model(opt)
        self.model.setup(opt)
        self.create_tensorboard(opt)

        index_step = 0
        epoch_start = opt.epoch_count
        epoch_end = opt.niter + opt.niter_decay + 1
        num_batches = len(dataloader)
        for index_epoch in range(epoch_start, epoch_end):

            if index_epoch % 2:
                print(f'Saving at epoch: {index_epoch}')
                self.model.save_networks(index_epoch)
            # if index_epoch % 1 == 0:
            #     print(f'Loading at epoch: {index_epoch}')
            #     self.model.load_networks(index_epoch)
            epoch_start_time = time.time()
            num_samples_epoch = 0
            correct_epoch_A = 0
            correct_epoch_B = 0
            for index_batch, data in enumerate(dataloader):
                num_samples_batch = len(data['A'])
                num_samples_epoch += num_samples_batch

                self.model.set_input(data)
                self.model.optimize_parameters()

                # load model
                # model.load_networks(index_epoch)
                # print('Model Loaded')

                losses = self.model.get_current_losses()
                self.print_current_losses(index_epoch, epoch_end, index_batch, num_batches, index_step, losses)
                self.update_tensorboard(losses, index_step)
                index_step += 1

                # acc_batch_A = self.model.correct_batch_A / num_samples_batch
                # acc_batch_B = self.model.correct_batch_B / num_samples_batch
                correct_epoch_A += self.model.correct_batch_A
                correct_epoch_B += self.model.correct_batch_B

                # ------------------------------------------------------------------------------------------------------
                if index_epoch % 2 == 0 and index_batch == 0:
                    num_display = 8
                    mode = 'train'
                    img_grid_real_A = make_grid(self.convert(self.model.real_A[:num_display], 0, 1))
                    img_grid_fake_A = make_grid(self.convert(self.model.fake_A[:num_display], 0, 1))
                    img_grid_rec_A = make_grid(self.convert(self.model.rec_A[:num_display], 0, 1))
                    img_grid_idt_A = make_grid(self.convert(self.model.idt_A[:num_display], 0, 1))


                    img_grid_real_B = make_grid(self.convert(self.model.real_B[:num_display], 0, 1))
                    img_grid_fake_B = make_grid(self.convert(self.model.fake_B[:num_display], 0, 1))
                    img_grid_rec_B = make_grid(self.convert(self.model.rec_B[:num_display], 0, 1))
                    img_grid_idt_B = make_grid(self.convert(self.model.idt_B[:num_display], 0, 1))

                    # combine the grids
                    img_grid_combined = torch.cat((
                        img_grid_real_A, img_grid_fake_B, img_grid_rec_A, img_grid_idt_A,
                        img_grid_real_B, img_grid_fake_A, img_grid_rec_B, img_grid_idt_B,
                    ), dim=1)
                    output_path_full = os.path.join(f'{opt.output_path}', f'{index_epoch}_{index_batch}_{mode}.jpg')
                    save_image(img_grid_combined, output_path_full)
                # ------------------------------------------------------------------------------------------------------


                print(f'   Correct - Batch\t\tA: [{self.model.correct_batch_A} / {num_samples_batch}]\t'
                      f' {(self.model.correct_batch_A/num_samples_batch)*100:.3f} %\t\t'
                      f'B: [{self.model.correct_batch_B} / {num_samples_batch}]\t'
                      f' {(self.model.correct_batch_B/num_samples_batch)*100:.3f} %')

            print(f'\nCorrect - Epoch\t\tA: [{correct_epoch_A} / {num_samples_epoch}]\t'
                  f'{(correct_epoch_A/num_samples_epoch)*100:.3f} %\t\t'
                  f'B: [{correct_epoch_B} / {num_samples_epoch}]\t'
                  f'{(correct_epoch_B/num_samples_epoch)*100:.3f} %')



            print(f'End of index_epoch {index_epoch} / {opt.niter} \t Time Taken: {time.time() - epoch_start_time} sec')

            self.model.update_learning_rate()
