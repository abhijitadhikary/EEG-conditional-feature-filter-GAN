import os
from conditional_feature_filter.conditional_feature_classifier import ConditionalFeatureFilterStarGAN

if __name__ == '__main__':
    # PrepareDataset()
    # feature_classifier = FeatureClassifier()
    # feature_classifier.run()

    # generate synthetic data
    # generate_synthetic_data(path_dataset=os.path.join('datasets', 'eeg'), load_freq_data=True)

    # train conditional feature classifier
    feature_classifier_conditional = ConditionalFeatureFilterStarGAN()
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
    # stargan_edit = StarGANEdit()
    # stargan_edit.train()



    # conditional_filter = ConditionalFilter()
    # conditional_filter.run()

    print()
