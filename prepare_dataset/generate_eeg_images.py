from __future__ import print_function
import numpy as np
import math as m
import scipy.io
from scipy.interpolate import griddata
from sklearn.preprocessing import scale
import scipy.io as sio
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from prepare_dataset.utils import augment_EEG, cart2sph, pol2cart

seed = 1234
np.random.seed(seed)

class GenerateEEGImages():
    def __init__(self, dataset_path):
        # Load electrode locations
        self.dataset_path = dataset_path
        filename_locs = 'Neuroscan_locs_orig.mat'
        filepath_locs_full = os.path.join(self.dataset_path, filename_locs)
        locs = scipy.io.loadmat(filepath_locs_full)
        locs_3d = locs['A']
        locs_2d = []

        # Convert to 2D
        for e in locs_3d:
            locs_2d.append(azim_proj(e))
        split_variants = ['within', 'cross']
        for split_variant in split_variants:
            print(f'Generating EEG images: {split_variant}')
            for mode in ('train', 'test', 'val'):
                filename = 'uci_eeg_' + mode + f'_{split_variant}.mat'
                filepath_full = os.path.join(self.dataset_path, filename)
                mat = sio.loadmat(filepath_full)
                data = mat['data']
                label_alcoholic = mat['label_alcoholism']
                label_stimulus = mat['label_stimulus']
                label_id = mat['label_id']

                tras_X = np.transpose(data, (0, 2, 1))
                X = make_frames(tras_X, 1)
                X_1 = X.reshape(np.shape(X)[0], np.shape(X)[1] * np.shape(X)[2])

                images = gen_images(np.array(locs_2d), X_1, 32, normalize=False)
                images = np.transpose(images, (0, 3, 2, 1))
                filename = 'uci_eeg_images_' + mode + f'_{split_variant}.mat'
                filepath_full = os.path.join(self.dataset_path, filename)
                sio.savemat(filepath_full,
                            {'data': images,
                             'label_alcoholism': label_alcoholic,
                             'label_stimulus': label_stimulus,
                             'label_id': label_id})
                img_in = images[0, :, :, :]
                img_in -= np.min(img_in)
                img_in /= np.max(img_in)

                plt.clf()
                plt.subplot(1, 1, 1)
                plt.imshow(img_in)

def azim_proj(pos):
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


def gen_images(locs, features, n_gridpoints, normalize=True,
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
    nElectrodes = locs.shape[0]     # Number of electrodes
    # Test whether the feature vector length is divisible by number of electrodes
    assert features.shape[1] % nElectrodes == 0
    n_colors = int(features.shape[1] / nElectrodes)
    for c in range(n_colors):
        feat_array_temp.append(features[:, c * nElectrodes: nElectrodes * (c+1)])
    if augment:
        if pca:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=True, n_components=n_components)
        else:
            for c in range(n_colors):
                feat_array_temp[c] = augment_EEG(feat_array_temp[c], std_mult, pca=False, n_components=n_components)
    num_samples = features.shape[0]
    # Interpolate the values
    grid_x, grid_y = np.mgrid[
                     min(locs[:, 0]):max(locs[:, 0]):n_gridpoints*1j,
                     min(locs[:, 1]):max(locs[:, 1]):n_gridpoints*1j
                     ]
    temp_interp = []
    for c in range(n_colors):
        temp_interp.append(np.zeros([num_samples, n_gridpoints, n_gridpoints]))
    # Generate edgeless images
    if edgeless:
        min_x, min_y = np.min(locs, axis=0)
        max_x, max_y = np.max(locs, axis=0)
        locs = np.append(locs, np.array([[min_x, min_y], [min_x, max_y],[max_x, min_y],[max_x, max_y]]),axis=0)
        for c in range(n_colors):
            feat_array_temp[c] = np.append(feat_array_temp[c], np.zeros((num_samples, 4)), axis=1)
    # Interpolating
    for i in range(num_samples):
        for c in range(n_colors):
            temp_interp[c][i, :, :] = griddata(locs, feat_array_temp[c][i, :], (grid_x, grid_y),
                                               method='cubic', fill_value=np.nan)
        print('Interpolating {0}/{1}\r'.format(i+1, num_samples), end='\r')
    
    # Normalizing
    for c in range(n_colors):
        if normalize:
            temp_interp[c][~np.isnan(temp_interp[c])] = \
                scale(temp_interp[c][~np.isnan(temp_interp[c])])
        temp_interp[c] = np.nan_to_num(temp_interp[c])    #Replace nan with zero and inf with large finite numbers.
    return np.swapaxes(np.asarray(temp_interp), 0, 1)     # swap axes to have [samples, colors, W, H]

def get_fft(snippet):
    Fs = 256.0  # sampling rate
    y = snippet
    n = len(y) # length of the signal
    k = np.arange(n)
    T = n/Fs
    frq = k/T # two sides frequency range
    frq = frq[range(int(n/2))] # one side frequency range

    Y = np.fft.fft(y)/n # fft computing and normalization
    Y = Y[range(int(n/2))]
    return frq, abs(Y)

def theta_alpha_beta_averages(f,Y):
    theta_range = (4, 8)
    alpha_range = (8, 13)
    beta_range = (13, 30)
    theta = Y[(f > theta_range[0]) & (f <= theta_range[1])].mean()
    alpha = Y[(f > alpha_range[0]) & (f <= alpha_range[1])].mean()
    beta = Y[(f > beta_range[0]) & (f <= beta_range[1])].mean()
    return theta, alpha, beta


def make_frames(df,frame_duration):
    '''
    in: dataframe or array with all channels, frame duration in seconds
    out: array of theta, alpha, beta averages for each probe for each time step
        shape: (n-frames,m-probes,k-brainwave bands)
    '''
    Fs = 256.0
    frame_length = Fs*frame_duration

    frames = []

    for i in tqdm(range(0, np.shape(df)[0]), leave=False):
        frame = []

        for channel in range(0, np.shape(df)[1]):
            snippet = df[i][channel]
            f, Y =  get_fft(snippet)
            theta, alpha, beta = theta_alpha_beta_averages(f, Y)
            frame.append([theta, alpha, beta])
            
        frames.append(frame)
    return np.array(frames)
