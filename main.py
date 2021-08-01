import os
from prepare_dataset.prepare_dataset import PrepareDataset
from feature_classifier.feature_classifier import FeatureClassifier
from feature_classifier_conditional.feature_classifier_conditional import FeatureClassifierConditional
from conditional_filter.conditional_filter import ConditionalFilter
from condition_creation import generate_synthetic_data

if __name__ == '__main__':
    # PrepareDataset()
    # feature_classifier = FeatureClassifier()
    # feature_classifier.run()

    # generate synthetic data
    generate_synthetic_data(path_dataset=os.path.join('datasets', 'eeg'), load_freq_data=True)

    # train conditional feature classifier
    feature_classifier_conditional = FeatureClassifierConditional()
    feature_classifier_conditional.run()

    # train cyclegan
    # conditional_filter = ConditionalFilter()
    # conditional_filter.run()

    print()
