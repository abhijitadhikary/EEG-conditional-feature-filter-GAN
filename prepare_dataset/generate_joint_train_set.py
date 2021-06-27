# -*- coding: utf-8 -*-
"""
Created on Wed May 20 01:17:56 2020

@author: konaw
"""

import scipy.io as sio
import numpy as np
import os


'''
Generate joint training set by combining the dummy EEG images with the training set 
'''
class GenerateJointTrainSet():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.generate_joint_train_set()

    def generate_joint_train_set(self):
        split_variants = ['within', 'cross']
        for split_variant in split_variants:
            print(f'Generating joint training set: {split_variant}')

            filename_real = f'uci_eeg_images_train_{split_variant}.mat'
            fullpath_real = os.path.join(self.dataset_path, filename_real)
            data_real = sio.loadmat(fullpath_real)

            filename_dummy = f'eeg_dummy_images_w_label_step3_{split_variant}.mat'
            fullpath_dummy = os.path.join(self.dataset_path, filename_dummy)
            data_dummy = sio.loadmat(fullpath_dummy)

            data = np.append(data_real['data'],
                             data_dummy['data'],
                             axis=0)
            label_alc = np.append(data_real['label_alcoholism'],
                                  data_dummy['label_alcoholism'],
                                  axis=0)
            label_stimulus = np.append(data_real['label_stimulus'],
                                       data_dummy['label_stimulus'],
                                       axis=0)
            filename = f'eeg_images_train_augmented_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            sio.savemat(filepath_full,
                        {'data':data,
                        'label_alcoholism':label_alc,
                        'label_stimulus':label_stimulus})