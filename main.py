import os
from prepare_dataset.prepare_dataset import PrepareDataset
from feature_classifier.feature_classifier import FeatureClassifier
from conditional_feature_classifier.conditional_feature_classifier import ConditionalFeatureClassifier
from conditional_filter.conditional_filter import ConditionalFilter
from condition_creation import generate_synthetic_data
from conditional_cycle_gan import ConditionalCycleGAN
from ccg import CCG
from stargan import StarGAN
from stargan_edit import StarGANEdit

if __name__ == '__main__':
    # PrepareDataset()
    # feature_classifier = FeatureClassifier()
    # feature_classifier.run()

    # generate synthetic data
    # generate_synthetic_data(path_dataset=os.path.join('datasets', 'eeg'), load_freq_data=True)

    # train conditional feature classifier
    feature_classifier_conditional = ConditionalFeatureClassifier()
    feature_classifier_conditional.run()

    # train cyclegan
    # conditional_cycle_gan = ConditionalCycleGAN()
    # conditional_cycle_gan.train()

    # # ccg
    # ccg = CCG()
    # ccg.train()

    # stargan
    # stargan = StarGAN()
    # stargan.train()

    # stargan edit
    stargan_edit = StarGANEdit()
    stargan_edit.train()



    # conditional_filter = ConditionalFilter()
    # conditional_filter.run()

    print()
