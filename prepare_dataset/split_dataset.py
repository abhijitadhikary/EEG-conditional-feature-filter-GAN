import scipy.io as sio
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

class SplitDataset():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.split_dataset()
        ## TODO check if source dataset exists

    def save_mat(self, data, label, mode, split_variant):
            filename = f'uci_eeg_{mode}_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            sio.savemat(filepath_full,
                        {'data': data,
                         'label_alcoholism': np.reshape(label[:, 0], (-1, 1)),
                         'label_stimulus': np.reshape(label[:, 1], (-1, 1)),
                         'label_id': np.reshape(label[:, 2], (-1, 1))})
            print(f'Saved: {filename}')

    def split_dataset(self):
        '''
            Prepares dataset for both within and cross
        '''

        dataset_link = 'https://anu365-my.sharepoint.com/:u:/g/personal/u7035746_anu_edu_au/EQ5YGXodkp1JomFAkfzOTjIBvzYzAEV5tnrOuAhZDBZ4eg?e=cr8sQb'
        # TODO update to automattically download dataset from given link

        filename = 'ucieeg.mat'
        filepath_full = os.path.join(self.dataset_path, filename)
        mat = sio.loadmat(filepath_full)
        dataset = mat['X'].astype('float32')
        label_alcoholism = mat['y_alcoholic'].astype('int')
        label_alcoholism = label_alcoholism.reshape(np.shape(dataset)[0])
        label_stimulus = mat['y_stimulus'].astype('int') - 1  # labels start from 0
        label_stimulus = label_stimulus.reshape(np.shape(dataset)[0])  # labels start from 0
        label_id = mat['subjectid'].astype('int') - 1
        label_id = label_id.reshape(np.shape(dataset)[0])

        train_data, train_label = [], []
        train_cyc_data, train_cyc_label = [], []
        val_data, val_label = [], []
        test_data, test_label = [], []

        print(f'Splitting dataset: "{filename}"')
        num_subjects = 122

        split_variants = ['within', 'cross']

        for split_variant in split_variants:
            print(f'Splitting variant: {split_variant}')
            if split_variant == 'within':
                # loop through each subject to split dataset within subject
                for index_subject in tqdm(range(num_subjects), leave=False, desc='Looping through each subject to split dataset within subject'):
                    index_selected = np.where(label_id == index_subject)
                    data_i = np.copy(dataset[index_selected])

                    label_alcoholism_i = label_alcoholism[index_selected]
                    label_stimulus_i = label_stimulus[index_selected]
                    label_id_i = label_id[index_selected]

                    # try 8-1-1 train-test-validation splitting
                    train_portion = 0.8
                    test_portion = 1 - train_portion
                    val_portion = 0.5  # out of test portion
                    label_stack_i = np.stack((label_alcoholism_i, label_stimulus_i, label_id_i), axis=1)
                    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(data_i, label_stack_i,
                                                                                test_size=test_portion,
                                                                                random_state=1)
                    X_test_i, X_val_i, y_test_i, y_val_i = train_test_split(X_test_i, y_test_i,
                                                                            test_size=val_portion,
                                                                            random_state=1)

                    # if alcoholic, sample 70% dataset for the cyclegan-based model training => balanced dataset
                    percentage_alcoholic = 0.7
                    if label_alcoholism_i[0] == 0 or \
                            ((label_alcoholism_i[0] == 1) and (np.random.rand() <= percentage_alcoholic)):
                        train_cyc_data.append(X_train_i)

                    train_data.append(X_train_i)
                    val_data.append(X_val_i)
                    test_data.append(X_test_i)
                    train_label.append(y_train_i)
                    test_label.append(y_test_i)
                    val_label.append(y_val_i)

                train_data = np.concatenate(train_data)
                train_label = np.concatenate(train_label)
                val_data = np.concatenate(val_data)
                val_label = np.concatenate(val_label)
                test_data = np.concatenate(test_data)
                test_label = np.concatenate(test_label)

            elif split_variant == 'cross':
                num_datapoints = dataset.shape[0]
                mask = np.zeros(num_subjects)

                # 7-2-1 for train-test-validation cross-subject data splitting
                train_percentage = 0.7
                test_percentage = 0.2
                val_percentage = 0.1
                for i in range(num_subjects):
                    r = np.random.rand()
                    if r < train_percentage:
                        mask[i] = 0
                    elif (r >= train_percentage) and r < (train_percentage+test_percentage):
                        mask[i] = 1
                    else:
                        mask[i] = 2

                # split according to subject id
                # 70% of subjects will be in training set
                train_data = [dataset[i] for i in range(num_datapoints) if mask[label_id[i]] == 0]
                train_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoints) if mask[label_id[i]] == 0]
                train_label_stimulus = [label_stimulus[i] for i in range(num_datapoints) if mask[label_id[i]] == 0]
                train_label_id = [label_id[i] for i in range(num_datapoints) if mask[label_id[i]] == 0]

                # 20% subjects in testing set
                test_data = [dataset[i] for i in range(num_datapoints) if mask[label_id[i]] == 1]
                test_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoints) if mask[label_id[i]] == 1]
                test_label_stimulus = [label_stimulus[i] for i in range(num_datapoints) if mask[label_id[i]] == 1]
                test_label_id = [label_id[i] for i in range(num_datapoints) if mask[label_id[i]] == 1]

                # 10% subjects for validation set
                val_data = [dataset[i] for i in range(num_datapoints) if mask[label_id[i]] == 2]
                val_label_alcoholism = [label_alcoholism[i] for i in range(num_datapoints) if mask[label_id[i]] == 2]
                val_label_stimulus = [label_stimulus[i] for i in range(num_datapoints) if mask[label_id[i]] == 2]
                val_label_id = [label_id[i] for i in range(num_datapoints) if mask[label_id[i]] == 2]

                # combine the data and labels into lists for saving as .mat files
                train_label = np.expand_dims([train_label_alcoholism, train_label_stimulus, train_label_id], 0)
                test_label = np.expand_dims([test_label_alcoholism, test_label_stimulus, test_label_id], 0)
                val_label = np.expand_dims([val_label_alcoholism, val_label_stimulus, val_label_id], 0)
            else:
                raise NotImplementedError('Invalid Split variant')

            # save the .mat files
            self.save_mat(train_data, train_label, 'train', split_variant)
            self.save_mat(test_data, test_label, 'test', split_variant)
            self.save_mat(val_data, val_label, 'val', split_variant)