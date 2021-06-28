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
from feature_classifier.create_dataset import CreateDataset
from feature_classifier.forward_pass import forward_pass


class FeatureClassifier():
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
        self.args.checkpoint_mode = 'accuracy'  # accuracy, loss
        self.get_num_classes()
        self.create_dataloaders()
        self.create_model()
        self.set_device()
        self.set_model_options()
        self.load_model()
        self.args.checkpoint_path = os.path.join('.', 'feature_classifier', 'checkpoints', self.args.feature, self.args.model_name)
        self.create_dirs()

    def set_hyperparameters(self):
        self.args.model_name = 'ResNet18'  # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
        self.args.feature = 'alcoholism'  # alcoholism, stimulus, id
        self.args.batch_size = 784
        self.args.learning_rate = 0.1
        self.args.momentum = 0.9
        self.args.weight_decay = 5e-4
        self.args.optimizer_variant = 'SGD' # SGD, Adam
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

    def create_dataloaders(self):
        # load the augmented dataset if we are using id as feature
        self.args.filename_train = 'uci_eeg_images_train_within.mat' if self.args.feature == 'id' else 'eeg_images_train_augmented_within.mat'
        self.args.filename_test = 'uci_eeg_images_test_within.mat'
        data_train = self.load_mat(self.args.filename_train)
        data_test = self.load_mat(self.args.filename_test)
        features_train = np.transpose(data_train['data'], (0, 3, 2, 1)).astype(np.float32)
        features_test = np.transpose(data_test['data'], (0, 3, 2, 1)).astype(np.float32)
        label = f'label_{self.args.feature}'
        labels_train = data_train[label].astype(np.int32)
        labels_train = labels_train.reshape(np.shape(labels_train)[0])
        labels_test = data_test[label].astype(np.int32)
        labels_test = labels_test.reshape(np.shape(labels_test)[0])

        # create dataloders
        dataset_train = CreateDataset(features_train, labels_train)
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.args.batch_size, shuffle=True)

        dataset_test = CreateDataset(features_test, labels_test)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.args.batch_size, shuffle=True)

    def load_mat(self, filename):
        return sio.loadmat(os.path.join(self.args.dataset_path, filename))

    def create_dirs(self):
        dir_list = [
            ['.', 'feature_classifier'],
            ['.', 'feature_classifier', 'checkpoints'],
            ['.', 'feature_classifier', 'checkpoints', self.args.feature, self.args.model_name]
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

    def get_num_classes(self):
        if self.args.feature == 'alcoholism':
            self.num_classes = 2
        elif self.args.feature == 'stimulus':
            self.num_classes = 5
        elif self.args.feature == 'id':
            self.num_classes = 122
        else:
            raise ValueError(f'feature [{self.args.feature}] not recognized.')

    def set_model_options(self):
        self.criterion = nn.CrossEntropyLoss()
        if self.args.optimizer_variant == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        elif self.args.optimizer_variant == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                        betas=(0.5, 0.999))
        else:
            raise NotImplementedError(f'Invalid optimizer_variant selected: {self.args.optimizer_variant}')

    def optimize_parameters(self, labels_real, labels_pred):
        loss = self.criterion(labels_pred, labels_real.long())
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



