import torch
import argparse
import numpy as np
import os
import scipy.io as sio
import time
from utils.visualizer import Visualizer
from conditional_cycle_gan.create_dataset import CreateDataset
from conditional_cycle_gan.train_options import TrainOptions
from conditional_cycle_gan.conditional_cyclegan_model import ConditionalCycleGANModel
from conditional_cycle_gan.create_dataloader import CreateDataLoader
# from conditional_cycle_gan.

class ConditionalCycleGAN:
    def __init__(self):
        self.args = argparse.Namespace()
        self.args.seed = 1324
        self.args.batch_size = 1
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
        print("model [%s] was created" % (model.name()))
        return model


    def parse(self):
        self.isTrain = True

    def train(self):
        opt = TrainOptions().parse()
        # data_loader = CreateDataLoader(opt)
        # dataset = data_loader.load_data()
        # dataset_size = len(data_loader)
        # print('#training images = %d' % dataset_size)
        dataset = self.dataloader_train
        dataset_size = len(dataset)

        model = self.create_model(opt)
        model.setup(opt)
        visualizer = Visualizer(opt)
        total_steps = 0

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                iter_start_time = time.time()

                if total_steps % opt.print_freq == 0:
                    t_data = iter_start_time - iter_data_time

                visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize

                model.set_input(data)

                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                    # model.save_networks('latest')

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
                # model.save_networks('latest')
                # model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

            model.update_learning_rate()
