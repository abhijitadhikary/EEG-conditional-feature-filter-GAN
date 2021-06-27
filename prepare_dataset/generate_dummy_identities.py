import scipy.io as sio
import os
import numpy as np
from tqdm import tqdm

class GenerateDummyIdentities():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.convert_time_frequency()

    def convert_time_frequency(self):
        split_variants = ['within', 'cross']
        for split_variant in split_variants:
            print(f'Time-Frequency conversion: {split_variant}')

            filename = f'uci_eeg_train_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            data = sio.loadmat(filepath_full)
            X = np.transpose(data['data'], (0, 2, 1))
            num_exp, num_ch, rate = X.shape
            spectrums = []

            # time-frequency conversion on all the EEG signals
            for i in tqdm(range(num_exp), leave=False):
                spectrum = []
                for ch in range(num_ch):
                    time_domain = X[i][ch]
                    f, magnitude = self.get_fft(time_domain)
                    spectrum.append(magnitude)
                spectrums.append(spectrum)
            filename = f'eeg_spectrum_train_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            sio.savemat(filepath_full, {'data': spectrums,
                                                          'label_alcoholism': data['label_alcoholism'],
                                                          'label_stimulus': data['label_stimulus'],
                                                          'label_id': data['label_id']})
        print(f'Time-Frequency conversion completed')

    def get_fft(self, snippet):
        '''
        time-frequency convertion using FFT
        '''
        Fs = 256.0  # sampling rate
        y = snippet
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / Fs
        frq = k / T  # two sides frequency range
        frq = frq[range(int(n / 2))]  # one side frequency range

        Y = np.fft.fft(y) / n  # fft computing and normalization
        Y = Y[range(int(n / 2))]
        return frq, abs(Y)





