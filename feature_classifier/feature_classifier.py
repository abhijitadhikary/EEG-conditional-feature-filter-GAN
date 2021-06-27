import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import argparse
from models.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
import numpy as np
import scipy.io as sio
import torch.utils.data as Data
from sklearn.metrics import recall_score
from tqdm import tqdm
import os
from feature_classifier.utils import create_model, create_dirs, remove_previous_checkpoints
from feature_classifier.create_dataset import CreateDataset

class FeatureClassifier():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            self.initialized = True
            self.seed = 1324
            self.set_seed()
            self.set_hyperparameters()
            self.dataset_path = os.path.join('.', 'datasets', 'eeg')
            self.model_name = 'ResNet18' # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
            self.feature = 'alcoholism' # alcoholism, stimulus, id
            self.batch_size = 784
            self.learning_rate = 0.1
            self.accuracy_best = np.NINF
            self.loss_best = np.Inf
            self.num_keep_best = 5
            self.checkpoint_mode = 'accuracy' # accuracy, loss
            self.checkpoint_path = os.path.join('.', 'feature_classifier', 'checkpoints', self.feature, self.model_name)
            self.resume = False
            self.get_num_classes()
            self.create_dataloaders()
            create_model(self)
            create_dirs(self)
            self.set_device()
            self.set_model_options()

    def create_dataloaders(self):
        # load the augmented dataset if we are using id as feature
        self.filename_train = 'uci_eeg_images_train_within.mat' if self.feature == 'id' else 'eeg_images_train_augmented_within.mat'
        self.filename_test = 'uci_eeg_images_test_within.mat'
        data_train = self.load_mat(self.filename_train)
        data_test = self.load_mat(self.filename_test)
        features_train = np.transpose(data_train['data'], (0, 3, 2, 1)).astype(np.float32)
        features_test = np.transpose(data_test['data'], (0, 3, 2, 1)).astype(np.float32)
        label = f'label_{self.feature}'
        labels_train = data_train[label].astype(np.int32)
        labels_train = labels_train.reshape(np.shape(labels_train)[0])
        labels_test = data_test[label].astype(np.int32)
        labels_test = labels_test.reshape(np.shape(labels_test)[0])

        # create dataloders
        dataset_train = CreateDataset(features_train, labels_train)
        self.dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)

        dataset_test = CreateDataset(features_test, labels_test)
        self.dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=self.batch_size, shuffle=True)

    def load_mat(self, filename):
        return sio.loadmat(os.path.join(self.dataset_path, filename))

    def set_device(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.nn.DataParallel(self.model)

    def set_seed(self):
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True

    def get_num_classes(self):
        if self.feature == 'alcoholism':
            self.num_classes = 2
        elif self.feature == 'stimulus':
            self.num_classes = 5
        elif self.feature == 'id':
            self.num_classes = 122
        else:
            raise ValueError(f'feature [{self.feature}] not recognized.')

    def set_hyperparameters(self):
        self.batch_size = 784
        self.learning_rate = 0.1
        self.momentum = 0.9
        self.weight_decay = 5e-4
        self.beta_1 = 0.5
        self.beta_2 = 0.999
        self.optimizer_variant = 'SGD' # SGD, Adam
        self.start_epoch = 0
        self.num_epochs = 200

    def set_model_options(self):
        self.criterion = nn.CrossEntropyLoss()
        if self.optimizer_variant == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum,
                                       weight_decay=self.weight_decay)
        elif self.optimizer_variant == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                        betas=(self.beta_1, self.beta_2))
        else:
            raise NotImplementedError(f'Invalid optimizer_variant selected: {self.optimizer_variant}')

    def optimize_parameters(self, labels_real, labels_pred):
        loss = self.criterion(labels_pred, labels_real.long())
        if self.mode == 'train':
            loss.backward()
            self.optimizer.step()
        return loss.item()

    def save_model(self, accuracy, loss):
        if self.mode == 'val':
            save_condition = False
            if self.checkpoint_mode == 'accuracy':
                if accuracy > self.accuracy_best:
                    save_condition = True
                    self.accuracy_best = accuracy
            if self.checkpoint_mode == 'loss':
                if loss < self.loss_best:
                    save_condition = True
                    self.loss_best = loss

            if save_condition:
                remove_previous_checkpoints(self)
                save_path = os.path.join(self.checkpoint_path, f'{self.index_epoch+1}.pth')
                save_dict = {'epoch': self.index_epoch + 1,
                             'learning_rate': self.learning_rate,
                             'state_dict': self.model.state_dict(),
                             'optim_dict': self.optimizer.state_dict()
                             }
                torch.save(save_dict, save_path)
                print(f'New best model saved at epoch: {self.index_epoch+1}')

    def forward_pass(self):
        # train
        if self.mode == 'train':
            self.model.train()
            dataloader = self.dataloader_train
        else:
            self.model.eval()
            dataloader = self.dataloader_test

        loss_epoch = 0
        correct_epoch = 0
        num_samples = 0
        accuracy_list, sensitivity_list, specificity_list = [], [], []

        for index_batch, (features, labels_real) in enumerate(dataloader):
            features, labels_real = features.to(self.device), labels_real.to(self.device)
            self.optimizer.zero_grad()
            labels_pred = self.model(features)
            loss_batch = self.optimize_parameters(labels_real, labels_pred)

            loss_epoch += loss_batch
            labels_pred = torch.argmax(labels_pred, dim=1)
            correct_batch = torch.sum(labels_pred == labels_real).item()
            length_batch = len(labels_real)

            accuracy_batch = (correct_batch / length_batch)
            accuracy_list.append(accuracy_batch)

            if self.feature == 'alcoholism' and self.mode == 'val':
                sensitivity_batch = recall_score(labels_real.cpu(), labels_pred.cpu(), pos_label=1) * 100.
                specificity_batch = recall_score(labels_real.cpu(), labels_pred.cpu(), pos_label=0) * 100.
                sensitivity_list.append(sensitivity_batch)
                specificity_list.append(specificity_batch)

        num_samples_epoch = len(accuracy_list)
        accuracy_epoch = 100 * (np.sum(accuracy_list) / num_samples_epoch)
        loss_epoch /= num_samples_epoch

        print(f'Epoch:\t[{self.index_epoch+1}/{self.num_epochs}]\t{self.mode.upper()} Loss:\t{loss_epoch:.3f}\tAccuracy:\t{accuracy_epoch:.3f} %\tCorrect:\t[{correct_epoch}/{num_samples}]', end='')
        if self.feature == 'alcoholism' and self.mode == 'val':
            sensitivity_epoch = (np.sum(sensitivity_list) / num_samples_epoch)
            specificity_epoch = (np.sum(specificity_list) / num_samples_epoch)
            print(f'\tSensitivity:\t{sensitivity_epoch:.3f} %\tSpecificity:\t{specificity_epoch:.3f} %')
        else:
            print()
        self.save_model(accuracy_epoch, loss_epoch)

    def run(self):
        for index_epoch in range(self.start_epoch, self.num_epochs):
            self.index_epoch = index_epoch
            for mode in ['train', 'val']:
                self.mode = mode
                self.forward_pass()



