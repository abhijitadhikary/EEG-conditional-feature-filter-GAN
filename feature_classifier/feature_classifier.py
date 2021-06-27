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
import os
from feature_classifier.utils import get_num_classes, define_model

class FeatureClassifier():
    def __init__(self):
        self.initialized = False

    def initialize(self):
        if not self.initialized:
            self.initialized = True
            self.dataset_path = os.path.join('.', 'datasets', 'eeg')
            self.model_name = 'ResNet18'
            self.feature = 'id'
            self.batch_size = 784
            self.learning_rate = 0.1
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.resume = False
            get_num_classes(self)
            define_model(self)

    def create_datasets(self):
        # load the augmented dataset if we are using id as feature
        self.filename_train = 'uci_eeg_images_train_within.mat' if self.feature == 'id' else 'eeg_images_train_augmented_within.mat'
        self.filename_test = 'uci_eeg_images_test_within.mat'
        self.dataset_train = self.load_mat(self.filename_train)
        self.dataset_test = self.load_mat(self.filename_test)

    def load_mat(self, filename):
        return sio.loadmat(os.path.join(self.dataset_path, filename))



