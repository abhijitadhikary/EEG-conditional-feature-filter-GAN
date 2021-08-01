from prepare_dataset.prepare_dataset import PrepareDataset
from feature_classifier.feature_classifier import FeatureClassifier
from feature_classifier_conditional.feature_classifier_conditional import FeatureClassifierConditional
from conditional_filter.conditional_filter import ConditionalFilter

if __name__ == '__main__':
    # PrepareDataset()
    # feature_classifier = FeatureClassifier()
    # feature_classifier.run()

    # train conditional feature classifier
    feature_classifier_conditional = FeatureClassifierConditional()
    feature_classifier_conditional.run()

    # train cyclegan
    # conditional_filter = ConditionalFilter()
    # conditional_filter.run()

    print()
