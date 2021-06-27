from prepare_dataset.prepare_dataset import PrepareDataset
from feature_classifier.feature_classifier import FeatureClassifier
import argparse

if __name__ == '__main__':
    # PrepareDataset()

    feature_classifier = FeatureClassifier()
    feature_classifier.run()
    print()
