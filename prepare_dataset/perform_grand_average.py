from __future__ import print_function
import scipy.io as sio
import numpy as np
import math as m
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
from prepare_dataset.utils import augment_EEG, cart2sph, pol2cart
import os
from tqdm import tqdm

seed = 1234
np.random.seed(seed)

'''
generate dummy identities using grand average
'''

'''
helper function for EEG2Img, 3D location projected to 2D plane
code from Yao
'''

class PerformGrandAverage():
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.preform_grand_average()

    def preform_grand_average(self):
        split_variants = ['within', 'cross']
        for split_variant in split_variants:
            print(f'Performing Grand Average: {split_variant}')
            # load EEG spectrums in training set
            filename = f'eeg_spectrum_train_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            data = sio.loadmat(filepath_full)

            label_disease_range = 2
            label_stimulus_range = 5
            # Group the EEG spectrums
            # generate 10 groups: 10 combinations of alcoholism and stimulus attributes
            groups = [{} for i in range(label_disease_range * label_stimulus_range)]  # map: subject id -> data

            # allocate each trial of EEG signals to the corresponding group
            for i in tqdm(range(len(data['data'])), leave=False, desc='Allocating each trial of EEG signals to the corresponding group'):
                index_label = label_stimulus_range * data['label_alcoholism'][i, 0] + data['label_stimulus'][i, 0]
                index_id = data['label_id'][i, 0]

                # add EEG data to the corresponding group and subject
                if index_id in groups[index_label]:
                    groups[index_label][index_id].append(data['data'][i])
                else:
                    groups[index_label][index_id] = [data['data'][i]]

            # Generate labelled EEG spectrums with dummy identities
            dummy = []
            dummy_label_disease = []
            dummy_label_stimulus = []

            # loop through 10 groups
            for i in tqdm(range(len(groups)), leave=False, desc='Looping through 10 groups'):
                # sort each group w.r.t the number of EEG signals for each subject
                candidate_list = sorted(list(groups[i].values()), key=len)
                step = 3  # a sliding window of 3
                # average trials of EEG siganls of the subjects within the window
                for j in range(len(candidate_list) - step + 1):
                    for k in range(len(candidate_list[j])):
                        # average across subjects
                        new = np.mean([item[k] for item in candidate_list[j:j + step]], axis=0)
                        dummy.append(new)
                        # label the dunmmy identity
                        dummy_label_disease.append(i // label_stimulus_range)
                        dummy_label_stimulus.append(i % label_stimulus_range)

            dummy = np.array(dummy)

            # Generate EEG images with dummy identities
            # Load electrode locations
            filename = f'Neuroscan_locs_orig.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            locs = sio.loadmat(filepath_full)
            locs_3d = locs['A']
            locs_2d = []

            # Convert to 2D
            for e in locs_3d:
                locs_2d.append(self.azim_proj(e))
            X = self.make_frames(dummy, 1)
            X_1 = X.reshape(np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])
            images = self.gen_images(np.array(locs_2d), X_1, 32, normalize=False)
            images = np.transpose(images, (0, 3, 2, 1))
            # save the dummy EEG images
            filename = f'eeg_dummy_images_w_label_step3_{split_variant}.mat'
            filepath_full = os.path.join(self.dataset_path, filename)
            sio.savemat(filepath_full,
                        {'data': images, 'label_alcoholism': np.reshape(dummy_label_disease, (-1, 1)),
                         'label_stimulus': np.reshape(dummy_label_stimulus, (-1, 1))})

        print(f'Grand Average completed')


    def azim_proj(self, pos):
        """
        Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
        Imagine a plane being placed against (tangent to) a globe. If
        a light source inside the globe projects the graticule onto
        the plane the result would be a planar, or azimuthal, map
        projection.

        :param pos: position in 3D Cartesian coordinates
        :return: projected coordinates using Azimuthal Equidistant Projection
        """
        [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])
        return pol2cart(az, m.pi / 2 - elev)


    '''
    helper function for EEG2Img, generate EEG images
    code from Yao
    '''


    def gen_images(self, locs, features, n_gridpoints, normalize=True,
                   augment=False, pca=False, std_mult=0.1, n_components=2, edgeless=False):
        """
        Generates EEG images given electrode locations in 2D space and multiple feature values for each electrode

        :param locs: An array with shape [n_electrodes, 2] containing X, Y
                            coordinates for each electrode.
        :param features: Feature matrix as [n_samples, n_features]
                                    Features are as columns.
                                    Features corresponding to each frequency band are concatenated.
                                    (alpha1, alpha2, ..., beta1, beta2,...)
        :param n_gridpoints: Number of pixels in the output images
        :param normalize:   Flag for whether to normalize each band over all samples
        :param augment:     Flag for generating augmented images
        :param pca:         Flag for PCA based data augmentation
        :param std_mult     Multiplier for std of added noise
        :param n_components: Number of components in PCA to retain for augmentation
        :param edgeless:    If True generates edgeless images by adding artificial channels
                            at four corners of the image with value = 0 (default=False).
        :return:            Tensor of size [samples, colors, W, H] containing generated
                            images.
        """
        feat_array_temp = []
        nElectrodes = locs.shape[0]  # Number of electrodes
        # Test whether the feature vector length is divisible by number of electrodes
        assert features.shape[1] % nElectrodes == 0
        n_colors = int(features.shape[1] / nElectrodes)
        for c in range(n_colors):
            feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c + 1)])
        if augment:
            if pca:
                for c in range(n_colors):
                    feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
            else:
                for c in range(n_colors):
                    feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
        nSamples = features.shape[0]
        # Interpolate the values
        grid_x, grid_y = np.mgrid[
                         min(locs[:, 0]):max(locs[:, 0]):n_gridpoints * 1j,
                         min(locs[:, 1]):max(locs[:, 1]):n_gridpoints * 1j
                         ]
        temp_interp = []
        for c in range(n_colors):
            temp_interp.append(np.zeros([nSamples, n_gridpoints, n_gridpoints]))
        # Generate edgeless images
        if edgeless:
            min_x, min_y = np.min(locs, axis=0)
            max_x, max_y = np.max(locs, axis=0)
            locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)
            for c in range(n_colors):
                feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((nSamples, 4)), axis=1)
        # Interpolating
        for i in tqdm(range(nSamples), leave=False, desc='Interpolating'):
            for c in range(n_colors):
                temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                                   method='cubic', fill_value=np.nan)
        # Normalizing
        for c in range(n_colors):
            if normalize:
                temp_interp[c][~np.isnan(temp_interp[c])] = \
                    scale(temp_interp[c][~np.isnan(temp_interp[c])])
            temp_interp[c] = np.nan_to_num(temp_interp[c])  # Replace nan with zero and inf with large finite numbers.
        return np.swapaxes(np.asarray(temp_interp), 0, 1)  # swap axes to have [samples, colors, W, H]


    '''
    helper function for EEG2Img, obtain features of 3 frequency band by averaging spectrum attitude
    code from Yao
    '''
    def theta_alpha_beta_averages(self, f, Y):
        theta_range = (4, 8)
        alpha_range = (8, 13)
        beta_range = (13, 30)
        theta = Y[(f > theta_range[0]) & (f <= theta_range[1])].mean()
        alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
        beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()
        return theta, alpha, beta


    def make_frames(self, df, frame_duration):
        '''
        in: dataframe or array with all channels, frame duration in seconds
        out: array of theta, alpha, beta averages for each probe for each time step
            shape: (n-frames,m-probes,k-brainwave bands)
        '''
        Fs = 256.0

        frames = []
        for i in tqdm(range(0, np.shape(df)[0]), leave=False, desc='Making frames'):
            frame = []
            for channel in range(0, np.shape(df)[1]):
                snippet = df[i][channel]
                theta, alpha, beta = self.theta_alpha_beta_averages(np.array(range(len(snippet))), snippet)
                frame.append([theta, alpha, beta])
            frames.append(frame)
        return np.array(frames)


