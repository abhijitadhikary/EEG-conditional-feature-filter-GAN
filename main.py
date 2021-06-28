from prepare_dataset.prepare_dataset import PrepareDataset
from feature_classifier.feature_classifier import FeatureClassifier
from conditional_filter.conditional_filter import ConditionalFilter

if __name__ == '__main__':
    # PrepareDataset()
    # feature_classifier = FeatureClassifier()
    # feature_classifier.run()

    conditional_filter = ConditionalFilter()
    conditional_filter.run()

    print()
