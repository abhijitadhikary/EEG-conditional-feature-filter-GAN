import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
import os
import argparse
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from utils.utils import create_dirs
from feature_classifier_conditional.create_dataset import CreateDataset
from feature_classifier_conditional.forward_pass import forward_pass


class FeatureClassifierConditional():
    def __init__(self):
        self.args = argparse.Namespace()
        self.args.seed = 1324
        self.set_seed()
        self.set_hyperparameters()
        self.args.dataset_path = os.path.join('.', 'datasets', 'eeg')
        self.args.accuracy_best = np.NINF
        self.args.loss_best = np.Inf
        self.args.num_keep_best = 5
        self.resume_epoch = 8
        self.resume_condition = False
        self.args.save_condition = True
        self.args.checkpoint_mode = 'loss'  # accuracy, loss
        self.args.split_mode = 'across'
        # self.get_num_classes()
        self.create_dataloaders()
        self.create_model()
        self.set_device()
        self.set_model_options()
        self.load_model()
        self.args.checkpoint_path = os.path.join('.', 'feature_classifier_conditional', 'checkpoints', self.args.feature, self.args.model_name)
        self.create_dirs()

    def set_hyperparameters(self):
        self.args.model_name = 'ResNet34'  # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        self.args.feature = 'alcoholism'  # alcoholism, stimulus, id
        self.args.batch_size = 384
        self.args.learning_rate = 0.0002
        self.args.momentum = 0.9
        self.args.weight_decay = 5e-4
        self.args.optimizer_variant = 'Adam' # SGD, Adam
        self.args.start_epoch = 0
        self.args.num_epochs = 200

    def create_model(self):
        if self.args.model_name == 'ResNet18':
            self.model = ResNet18(self.num_classes)
        elif self.args.model_name == 'ResNet34':
            self.model = ResNet34(self.num_classes)
        elif self.args.model_name == 'ResNet50':
            self.model = ResNet50(self.num_classes)
        elif self.args.model_name == 'ResNet101':
            self.model = ResNet101(self.num_classes)
        elif self.args.model_name == 'ResNet152':
            self.model = ResNet152(self.num_classes)
        else:
            raise NotImplementedError(f'model_name [{self.args.model_name}] not implemented.')

    def process_conditional_features(self, data):
        images = data['images']
        images = np.transpose(images, (0, 3, 1, 2))
        # labels = data['labels'][:, :3] # take one_by_three labels
        labels = data['labels'][:, -1] # take 0-7 labels
        self.num_classes = 8

        #####################################################################################
        check_one_vs_one = False

        if check_one_vs_one:
            self.num_classes = 2
            variant = 1
            # ---------- TEST Alcoholism
            if variant == 0:
                # Alcoholism:   TEST
                # Stimulus:     YES
                # ID:           YES
                label_a = 0
                label_b = 1
            elif variant == 1:
                # ----------------------------- does not ( < 50% )
                # Alcoholism:   TEST
                # Stimulus:     YES
                # ID:           NO
                label_a = 6
                label_b = 7
            elif variant == 2:
                # ----------------------------- does not work ( < 50% )
                # Alcoholism:   TEST
                # Stimulus:     NO
                # ID:           NO
                label_a = 2
                label_b = 3
            elif variant == 3:
                # ----------------------------- does not work ( < 50% )
                # Alcoholism:   TEST
                # Stimulus:     NO
                # ID:           YES
                label_a = 4
                label_b = 5




            # ---------- TEST Stimulus
            elif variant == 4:
                # Alcoholism:   YES
                # Stimulus:     TEST
                # ID:           YES
                label_a = 0
                label_b = 4
            elif variant == 5:
                # ----------------------------- moderate (85%)
                # Alcoholism:   YES
                # Stimulus:     TEST
                # ID:           NO
                label_a = 6
                label_b = 2
            elif variant == 6:
                # ----------------------------- moderate (85%)
                # Alcoholism:   NO
                # Stimulus:     TEST
                # ID:           NO
                label_a = 7
                label_b = 3
            elif variant == 7:
                # Alcoholism:   NO
                # Stimulus:     TEST
                # ID:           YES
                label_a = 1
                label_b = 5




            # ---------- TEST ID
            elif variant == 8:
                # Alcoholism:   YES
                # Stimulus:     YES
                # ID:           TEST
                label_a = 0
                label_b = 6
            elif variant == 9:
                # Alcoholism:   YES
                # Stimulus:     NO
                # ID:           TEST
                label_a = 2
                label_b = 4
            elif variant == 10:
                # Alcoholism:   NO
                # Stimulus:     NO
                # ID:           TEST
                label_a = 3
                label_b = 5
            elif variant == 10:
                # Alcoholism:   NO
                # Stimulus:     YES
                # ID:           TEST
                label_a = 1
                label_b = 7

            index_a = np.where(labels == label_a)
            index_b = np.where(labels == label_b)

            indices = np.hstack((index_a, index_b)).reshape(-1)
            images = images[indices]
            labels = labels[indices]

            label_a, label_b = np.unique(labels)
            labels = np.where(labels == label_a, 0, 1)

            #####################################################################################

        num_samples = len(images)

        train_portion = 0.7
        val_portion = 0.33

        mask_train = np.random.rand(num_samples) <= train_portion
        images_train = images[mask_train]
        labels_train = labels[mask_train]

        images_temp = images[~mask_train]
        labels_temp = labels[~mask_train]
        num_temp = len(images_temp)

        mask_val = np.random.rand(num_temp) <= val_portion
        images_val = images_temp[mask_val]
        labels_val = labels_temp[mask_val]

        images_test = images_temp[~mask_val]
        labels_test = labels_temp[~mask_val]

        return images_train, labels_train, images_val, labels_val, images_test, labels_test


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

    def load_mat(self, filename):
        return sio.loadmat(os.path.join(self.args.dataset_path, filename))

    def create_dirs(self):
        dir_list = [
            ['.', 'feature_classifier_conditional'],
            ['.', 'feature_classifier_conditional', 'checkpoints'],
            ['.', 'feature_classifier_conditional', 'checkpoints', self.args.feature, self.args.model_name]
        ]
        create_dirs(dir_list)

    def set_device(self):
        self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.nn.DataParallel(self.model)

    def set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = True

    # def get_num_classes(self):
    #     self.num_classes = 2
        # if self.args.feature == 'alcoholism':
        #     self.num_classes = 2
        # elif self.args.feature == 'stimulus':
        #     self.num_classes = 5
        # elif self.args.feature == 'id':
        #     self.num_classes = 122
        # else:
        #     raise ValueError(f'feature [{self.args.feature}] not recognized.')

    def set_model_options(self):
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_bce = nn.BCELoss()
        self.criterion_mse = nn.MSELoss()

        if self.args.optimizer_variant == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer_variant == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                        betas=(0.5, 0.999))
        else:
            raise NotImplementedError(f'Invalid optimizer_variant selected: {self.args.optimizer_variant}')

    # def optimize_parameters(self, labels_real, labels_pred):
    #     loss = self.criterion_mse(labels_real, labels_pred)
    #
    #     # loss = self.criterion(labels_pred, labels_real.long())
    #     if self.args.mode == 'train':
    #         loss.backward()
    #         self.optimizer.step()
    #     return loss.item()

    def optimize_parameters(self, labels_real, labels_pred):
        loss = self.criterion(labels_pred, labels_real)
        if self.args.mode == 'train':
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def remove_previous_checkpoints(self):
        all_files = sorted(os.listdir(self.args.checkpoint_path))
        if len(all_files) >= self.args.num_keep_best:
            current_full_path = os.path.join(self.args.checkpoint_path, all_files[0])
            os.remove(current_full_path)

    def save_model(self, accuracy, loss):
        if self.args.mode == 'val' and self.args.save_condition:
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
                print(f'*********************** New best model saved at {self.args.index_epoch + 1} ***********************')

    def load_model(self):
        if self.resume_condition:
            load_path = os.path.join(self.args.checkpoint_path, f'{self.resume_epoch}.pth')
            if load_path is not None:
                if not os.path.exists(load_path):
                    raise FileNotFoundError(f'File {load_path} doesn\'t exist')

                checkpoint = torch.load(load_path)

                # args.start_epoch = checkpoint['epoch']
                self.args = checkpoint['args']
                self.criterion = checkpoint['criterion']
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optim_state_dict'])
                print(f'Model successfully loaded from epoch {self.resume_epoch}')
                self.args.start_epoch = self.args.index_epoch + 1

    def run(self):
        for index_epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.args.index_epoch = index_epoch
            for mode in ['train', 'val']:
                self.args.mode = mode
                forward_pass(self)



