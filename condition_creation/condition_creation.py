import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd
from tqdm import tqdm
import os
from condition_creation.utils import convert_time_to_frequency, get_2d_electrode_locations, make_frames, gen_images

def load_data(filename='ucieeg.mat'):
    # load the .mat file
    data_mat = loadmat(filename)
    # create separate arrays for each feature/target
    data_time = data_mat['X']
    label_alcoholism = data_mat['y_alcoholic'].reshape(-1)
    label_stimulus = data_mat['y_stimulus'].reshape(-1)
    label_id = data_mat['subjectid'].reshape(-1)

    # create dataframe of labels
    labels = {
        'alcoholism': label_alcoholism,
        'stimulus': label_stimulus,
        'id': label_id
    }
    df = pd.DataFrame(labels)

    # remove all trials containing stimulus 3 and 4 (data)
    indices_stimulus_4 = df.loc[(df['stimulus'] == 4)].index
    indices_stimulus_5 = df.loc[(df['stimulus'] == 5)].index
    indices_stimulus_4_n_5 = np.array(indices_stimulus_4.append(indices_stimulus_5))
    data_time = np.delete(data_time, indices_stimulus_4_n_5, 0)

    # remove all trials containing stimulus 3 and 4 (labels)
    num_samples = len(data_time)
    indices_new = pd.Series(np.arange(0, num_samples))
    df = df.drop(indices_stimulus_4_n_5, axis=0).set_index(indices_new)
    return data_time, df


def get_average_data(data_input, window_size):
    '''
        Returns the average of the data_input over the window_size
    '''
    data_avg_list = []
    for index in range(window_size // 2, len(data_input) - window_size + (window_size // 2)):
        start = index - window_size // 2
        end = index + window_size // 2 + 1

        indices = [index_a for index_a in range(start, end)]
        data_selected = data_input[indices]
        data_avg = np.mean(data_selected, axis=0)
        data_avg_list.append(data_avg)
    data_avg_list = np.array(data_avg_list)
    return data_avg_list


def hide_identity_keep_stimulus(df, data_freq, alcoholism_condition, window_size=3, to_sample=2000, random_state=0):
    # V stands for variable alcoholism
    to_sample = to_sample // 3
    to_sample += window_size

    # extract the indices for the three different stimulus conditons with the specifiec alcoholism condition
    indices_V_1_T = df.query(f'alcoholism == {alcoholism_condition} and stimulus == 1').index
    indices_V_2_T = df.query(f'alcoholism == {alcoholism_condition} and stimulus == 2').index
    indices_V_3_T = df.query(f'alcoholism == {alcoholism_condition} and stimulus == 3').index

    # extract the corresponding entries in the dataframe
    df_V_1_T = df.iloc[indices_V_1_T].sample(n=to_sample, replace=True, random_state=random_state)
    df_V_2_T = df.iloc[indices_V_2_T].sample(n=to_sample, replace=True, random_state=random_state)
    df_V_3_T = df.iloc[indices_V_3_T].sample(n=to_sample, replace=True, random_state=random_state)

    indices_selected__V_1_T = df_V_1_T.index
    indices_selected__V_2_T = df_V_2_T.index
    indices_selected__V_3_T = df_V_3_T.index

    data_selected__V_1_T = data_freq[indices_selected__V_1_T]
    data_selected__V_2_T = data_freq[indices_selected__V_2_T]
    data_selected__V_3_T = data_freq[indices_selected__V_3_T]

    # TODO adapt a better sampling strategy to make sure that people with same identities
    # do not end up in a given window size

    # average over the identities, while keeping stimulus and alcoholism informaiton
    data_V_1_F = get_average_data(data_selected__V_1_T, window_size)
    data_V_2_F = get_average_data(data_selected__V_2_T, window_size)
    data_V_3_F = get_average_data(data_selected__V_3_T, window_size)

    # stack the three different stimulius conditions into one
    data_V_T_F = np.vstack((data_V_1_F, data_V_2_F, data_V_3_F))

    return data_V_T_F


def hide_identity_hide_stimulus(df, data_freq, alcoholism_condition, window_size=3, to_sample=2000, random_state=0):
    # V stands for variable alcoholism
    to_sample += window_size

    # extract the indices corresponding to alcoholism
    indices_V_T_T = df.query(f'alcoholism == {alcoholism_condition}').index

    # extract the corresponding entries in the dataframe
    df_V_T_T = df.iloc[indices_V_T_T].sample(n=to_sample, replace=True, random_state=random_state)
    indices_selected = df_V_T_T.index
    data_selected = data_freq[indices_selected]
    # TODO adapt a better sampling strategy to make sure that people with same identities
    # do not end up in a given window size

    # average over the identities and stimulus while keeping alcoholism informaiton
    data_selected = get_average_data(data_selected, window_size)

    return data_selected


def keep_identity_hide_stimulus(df, data_freq, alcoholism_condition, window_size=3, to_sample=2000, random_state=0):
    # V stands for variable alcoholism
    to_sample_in = to_sample
    if alcoholism_condition == 1:
        to_sample = to_sample // 77
    else:
        to_sample = to_sample // 20
    to_sample += window_size

    data_V_F_VI_array = []

    identity_range = np.arange(1, 123)
    for index_id in identity_range:
        # extract the indices corresponding to alcoholism
        indices_V_T_VI = df.query(f'alcoholism == {alcoholism_condition} and id == {index_id}').index
        if len(indices_V_T_VI) > 0:
            # extract the corresponding entries in the dataframe
            df_V_T_VI = df.iloc[indices_V_T_VI].sample(n=to_sample, replace=True, random_state=random_state)
            indices_selected = df_V_T_VI.index
            data_selected = data_freq[indices_selected]

            # average over the stimulus, while keeping identity and alcoholism informaiton
            data_V_F_VI = get_average_data(data_selected, window_size)
            if len(data_V_F_VI_array) == 0:
                data_V_F_VI_array = data_V_F_VI
            else:
                data_V_F_VI_array = np.vstack((data_V_F_VI_array, data_V_F_VI))

    num_elements = len(data_V_F_VI_array)

    if num_elements > to_sample_in:
        # if num_elements is larger, randomly sample a subset
        data_V_F_VI_array = data_V_F_VI_array[np.random.choice(np.arange(num_elements), num_elements)][:to_sample_in]
    if num_elements < to_sample_in:
        # if num_elements is smaller, randomly resample some points and append to the existing
        num_diff = to_sample_in - num_elements

        data_temp = data_V_F_VI_array[np.random.choice(np.arange(num_elements), num_diff)]
        data_V_F_VI_array = np.vstack((data_V_F_VI_array, data_temp))

    return data_V_F_VI_array


def keep_identity_keep_stimulus(df, data_freq, alcoholism_condition, window_size=3, to_sample=2000, random_state=0):
    # extract the indices corresponding to alcoholism
    indices_V_T_T = df.query(f'alcoholism == {alcoholism_condition}').index
    df_V_T_T = df.iloc[indices_V_T_T].sample(n=to_sample, replace=True, random_state=random_state)

    indices_V_T_T = df_V_T_T.index
    data_V_T_T = data_freq[indices_V_T_T]

    return data_V_T_T

def create_synthetic_dummy_data(df, data_freq, samples_per_category, window_size):
    # 1. (T T T) Alcoholism + stimulus + identity
    data_T_T_T = keep_identity_keep_stimulus(df, data_freq, alcoholism_condition=1, window_size=window_size, to_sample=samples_per_category)

    # 2. (F T T) NO-Alcoholism + stimulus + identity
    data_F_T_T = keep_identity_keep_stimulus(df, data_freq, alcoholism_condition=0, window_size=window_size, to_sample=samples_per_category)

    # 3. (T F F) Alcoholism + NO-stimulus + NO-identity
    data_T_F_F = hide_identity_hide_stimulus(df, data_freq, alcoholism_condition=1, window_size=window_size, to_sample=samples_per_category)

    # 4. (F F F) NO-Alcoholism + NO-stimulus + NO-identity
    data_F_F_F = hide_identity_hide_stimulus(df, data_freq, alcoholism_condition=0, window_size=window_size, to_sample=samples_per_category)

    # 5. (T F T) Alcoholism + NO-stimulus + identity
    data_T_F_T = keep_identity_hide_stimulus(df, data_freq, alcoholism_condition=1, window_size=window_size, to_sample=samples_per_category)

    # 6. (F F T) Alcoholism + NO-stimulus + identity
    data_F_F_T = keep_identity_hide_stimulus(df, data_freq, alcoholism_condition=0, window_size=window_size, to_sample=samples_per_category)

    # 7. (T T F) Alcoholism + stimulus + NO-identity
    data_T_T_F = hide_identity_keep_stimulus(df, data_freq, alcoholism_condition=1, window_size=window_size, to_sample=samples_per_category)

    # 8. (F T F) NO-Alcoholism + stimulus + NO-identity
    data_F_T_F = hide_identity_keep_stimulus(df, data_freq, alcoholism_condition=0, window_size=window_size, to_sample=samples_per_category)

    data_synthetic_dummy = [data_T_T_T, data_F_T_T, data_T_F_F, data_F_F_F, data_T_F_T, data_F_F_T, data_T_T_F, data_F_T_F]
    return data_synthetic_dummy

def generate_synthetic_dummy_labels(samples_per_category):
    # generate the labels [Alcoholism, Stimulus, Identity, Category(0-7)]
    labels_T_T_T = np.repeat(np.array([[1, 1, 1, 0]]), samples_per_category, axis=0)
    labels_F_T_T = np.repeat(np.array([[0, 1, 1, 1]]), samples_per_category, axis=0)
    labels_T_F_F = np.repeat(np.array([[1, 0, 0, 2]]), samples_per_category, axis=0)
    labels_F_F_F = np.repeat(np.array([[0, 0, 0, 3]]), samples_per_category, axis=0)
    labels_T_F_T = np.repeat(np.array([[1, 0, 1, 4]]), samples_per_category, axis=0)
    labels_F_F_T = np.repeat(np.array([[0, 0, 1, 5]]), samples_per_category, axis=0)
    labels_T_T_F = np.repeat(np.array([[1, 1, 0, 6]]), samples_per_category, axis=0)
    labels_F_T_F = np.repeat(np.array([[0, 1, 0, 7]]), samples_per_category, axis=0)

    # stack the labels into a single array
    labels = np.vstack((labels_T_T_T, labels_F_T_T, labels_T_F_F, labels_F_F_F, labels_T_F_T, labels_F_F_T, labels_T_T_F, labels_F_T_F))

    return labels

def split_data_within(images, labels):
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

    data_split_within = {
        'train': (images_train, labels_train),
        'val': (images_val, labels_val),
        'test': (images_test, labels_test)
    }

    return data_split_within

def save_data_mat(data_split, split_mode, path_dataset):
    print(f'Saving images and labels ({split_mode})')

    for current_split in data_split.keys():
        images = data_split[current_split][0]
        labels = data_split[current_split][1]

        save_filename = f'eeg_conditional_images_{current_split}_{split_mode}.mat'
        save_path_full = os.path.join(path_dataset, save_filename)
        savemat(save_path_full, {
                    'images': images,
                    'labels': labels,
                    f'{split_mode}': split_mode
                })

        print(f'Save successful! ({split_mode})')

def split_data_across(df, data_freq):
    num_samples = len(df)
    train_portion = 0.7
    val_portion = 0.33

    mask_train = np.random.rand(num_samples) <= train_portion

    df_train = df.iloc[mask_train].copy().reset_index()
    data_freq_train = data_freq[mask_train]

    df_temp = df.iloc[~mask_train].copy().reset_index()
    data_freq_temp = data_freq[~mask_train]
    num_temp = len(df_temp)

    mask_val = np.random.rand(num_temp) <= val_portion
    df_val = df_temp.iloc[mask_val].copy().reset_index()
    data_freq_val = data_freq_temp[mask_val]

    df_test = df_temp.iloc[~mask_val].copy().reset_index()
    data_freq_test = data_freq_temp[~mask_val]

    data_split_across = {
        'train': (df_train, data_freq_train),
        'val': (df_val, data_freq_val),
        'test': (df_test, data_freq_test)
    }

    # data_split_across = [(df_train, data_freq_train), (df_val, data_freq_val), (df_test, data_freq_test)]

    return data_split_across

def generate_synthetic_data(window_size=7, samples_per_category=3000, frame_size=32, num_channels=3, load_freq_data=False, path_dataset=None):

    # set seed
    np.random.seed(0)

    # load the complete ucieeg dataset
    if path_dataset is None:
        path_dataset = os.path.join('..', 'datasets', 'eeg')

    if load_freq_data:
        data_mat = loadmat(os.path.join(path_dataset, 'eeg_freq_data.mat'))
        data_freq = data_mat['data_freq']

        df = pd.read_csv(os.path.join(path_dataset, 'eeg_freq_label.csv'))
    else:
        data_time, df = load_data(os.path.join(path_dataset, 'ucieeg.mat'))

        # convert data from time to frequency domain
        print('Converting from time to frequency domain')
        data_freq = convert_time_to_frequency(data_time)

        savemat(os.path.join(path_dataset, 'eeg_freq_data.mat'), {'data_freq': data_freq})
        df.to_csv(os.path.join(path_dataset, 'eeg_freq_label.csv'), index=False)

    # convert 3d locations of electrodes to 2d
    locs_2d = get_2d_electrode_locations(path_dataset)

    # within : within each subject, split train/test/val
    # across : select 70% people for training, 20% for testing and 10% for validation

    # ------------------------------------------------ within ----------------------------------------------------------
    # split_mode = 'within'
    # print(f'Creating synthetic conditioned data ({split_mode})')
    # synthetic_dummy_data = create_synthetic_dummy_data(df, data_freq, samples_per_category, window_size)
    #
    # print(f'Making frames ({split_mode})')
    # # create frames (extracts only the theta, aplha and beta channels (3))
    # frames = [make_frames(data_current, 1) for data_current in synthetic_dummy_data]
    #
    # # convert signals to RGB images
    # print(f'Generating images from signals ({split_mode})')
    # images = [gen_images(locs_2d, image_current, frame_size, normalize=False) for image_current in frames]
    #
    # # stack images into a single array
    # images = np.array(images).reshape(-1, frame_size, frame_size, num_channels)
    #
    # # generate the labels [Alcoholism, Stimulus, Identity, Category(0-7)]
    # labels = generate_synthetic_dummy_labels(samples_per_category)
    #
    # data_split_within = split_data_within(images, labels)
    # save_data_mat(data_split_within, split_mode, path_dataset)
    # ------------------------------------------------ within ----------------------------------------------------------
    # ******************************************************************************************************************
    # ------------------------------------------------ across ----------------------------------------------------------
    split_mode = 'across'
    data_split_across = split_data_across(df, data_freq)

    for current_split in data_split_across.keys():
        df = data_split_across[current_split][0]
        data_freq = data_split_across[current_split][1]

        print(f'Creating synthetic conditioned data ({split_mode}: {current_split})')
        samples_per_category_list = {'train': 2100, 'val': 300, 'test': 600}
        synthetic_dummy_data = create_synthetic_dummy_data(df, data_freq, samples_per_category_list[current_split], window_size)

        print(f'Making frames ({split_mode}: {current_split})')
        # create frames (extracts only the theta, aplha and beta channels (3))
        frames = [make_frames(data_current, 1) for data_current in synthetic_dummy_data]

        # convert signals to RGB images
        print(f'Generating images from signals ({split_mode}: {current_split})')
        images = [gen_images(locs_2d, image_current, frame_size, normalize=False) for image_current in frames]

        # stack images into a single array
        images = np.array(images).reshape(-1, frame_size, frame_size, num_channels)

        # generate the labels [Alcoholism, Stimulus, Identity, Category(0-7)]
        labels = generate_synthetic_dummy_labels(samples_per_category)
        data_split_across_current = {
            f'{current_split}': (images, labels)
        }
        save_data_mat(data_split_across_current, split_mode, path_dataset)
        print('done')



if __name__=='__main__':
    generate_synthetic_data()
