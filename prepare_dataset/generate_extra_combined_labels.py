import scipy.io as sio
import numpy as np
import os

class GenerateExtraCombinedLabels():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.generate_extra_combined_labels()

    def generate_extra_combined_labels(self):
        split_variants = ['within', 'cross']
        for split_variant in split_variants:
            print(f'Generating Extra Combined Labels: {split_variant}')
            self.add_combined_label(f'eeg_dummy_images_w_label_step3_{split_variant}')
            self.add_combined_label(f'uci_eeg_images_train_{split_variant}')
        print(f'Generating Extra Combined Labels Completed')
    def add_combined_label(self, filename):
        filepath_full = os.path.join(self.dataset_path, filename)
        data = sio.loadmat(filepath_full)
        label_combined = []
        for i in range(data['label_stimulus'].shape[0]):  # 0,1  0,1,2,3,4
            label = 5 * data['label_alcoholism'][i, 0] + data['label_stimulus'][i, 0]
            label_combined.append(label)
        label_combined = np.reshape(label_combined, (-1, 1))
        if 'label_id' in data.keys():
            sio.savemat(filepath_full + '_extra.mat', {'data': data['data'],
                                                  'label_alcoholism': data['label_alcoholism'],
                                                  'label_stimulus': data['label_stimulus'],
                                                  'label_id': data['label_id'],
                                                  'label_combined': label_combined})
        else:
            sio.savemat(filepath_full + '_extra.mat', {'data': data['data'],
                                                  'label_alcoholism': data['label_alcoholism'],
                                                  'label_stimulus': data['label_stimulus'],
                                                  'label_combined': label_combined})


