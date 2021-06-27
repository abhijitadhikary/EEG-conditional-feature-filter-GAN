import os
from prepare_dataset.split_dataset import SplitDataset
from prepare_dataset.generate_eeg_images import GenerateEEGImages
from prepare_dataset.generate_dummy_identities import GenerateDummyIdentities
from prepare_dataset.perform_grand_average import PerformGrandAverage
from prepare_dataset.generate_joint_train_set import GenerateJointTrainSet
from prepare_dataset.generate_extra_combined_labels import GenerateExtraCombinedLabels
class PrepareDataset():
    def __init__(self):
        dataset_path = os.path.join('.', 'datasets', 'eeg')
        SplitDataset(dataset_path)
        GenerateEEGImages(dataset_path)
        GenerateDummyIdentities(dataset_path)
        PerformGrandAverage(dataset_path)
        GenerateJointTrainSet(dataset_path)
        GenerateExtraCombinedLabels(dataset_path)

        print('Dataset preparation completed')
