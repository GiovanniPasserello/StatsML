from math import sqrt

import numpy as np
import pickle
import random

from statsml.decisionclassifier import DecisionTreeClassifier
from statsml.decisionclassifier.dataset import Dataset
from statsml.decisionclassifier.utils import majority_label


class RandomForestClassifier(object):
    """
    A random forest decision tree classifier

    Attributes:
        is_trained {bool} -- Keeps track of whether the classifier has been trained
    Methods:
        train(X, y) -- Constructs a random forest from data X and label y
        predict(X) -- Predicts the class label of samples X
    """

    def __init__(self, n):
        self.n = n  # The number of decision trees
        self.is_trained = False
        self.trees = [DecisionTreeClassifier() for _ in range(n)]
        self.dataset = None

    def train(self, attributes, labels):
        """ Constructs a decision tree classifier from data

        Arguments:
            attributes {np.ndarray} -- An N by K numpy array (N is # of instances, K is # of attributes)
            labels {np.ndarray} -- An N-dimensional numpy array
        Returns:
            {DecisionTreeClassifier} -- A copy of the DecisionTreeClassifier instance
        """

        # Assert that each set of attributes has a label
        assert attributes.shape[0] == len(labels), "Training failed: Missing training data."
        assert len(attributes) != 0, "Empty Training Data"

        self.dataset = Dataset(attributes, labels)

        total_samples = len(self.dataset)
        total_features = len(attributes[0])

        # Heuristically take 2 * sqrt of the total number of features for each tree.
        # This strikes a good balance between tree correlation and strength.
        # From the 'Random Forests' paper, sqrt can be too low as the trees have low correlation but also far too low
        # strength if there are some 'garbage' features -> use 2 * sqrt.
        # 2 * sqrt also trains about 4 times quicker than using all features.
        num_features = 2 * int(sqrt(total_features))

        # Heuristically take 2/3 of the dataset for each tree, this way if n is large (> 50), each dataset sample
        # is probabilistically guaranteed to be selected at least once (proof omitted)
        num_samples = (2 * total_samples) // 3

        # Edge case of very small dataset -> just use all samples
        if num_samples == 0:
            num_samples = total_samples

        for i in range(self.n):
            # Random selection of features and dataset samples as proposed in the 'Random Forests' paper
            # N.B. Could look for importance of feature weighting in pre-processing, but this would hinder
            # the generalisation of the problem, so use strictly randomly selected features instead.
            feature_indices = random.sample(range(total_features), num_features)
            dataset_indices = random.sample(range(total_samples), num_samples)

            atts = self.dataset.attributes[dataset_indices]
            labs = self.dataset.labels[dataset_indices]

            print("Training Tree", i + 1)
            # N.B. In random forest, we don't prune as we want to maintain specificity across trees
            self.trees[i].train(atts, labs, prune=False, valid_features=feature_indices)

        self.is_trained = True

        return self

    def predict(self, attrs):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.

        Arguments:
            attrs {np.ndarray} -- An N by K numpy array of attributes samples (N is # of samples, K is # of attributes)
        Returns:
            {np.ndarray} -- An N-dimensional numpy array containing the predicted class label for each instance in attrs
        """

        assert self.is_trained, "Random Forest classifier has not yet been trained."

        predictions = np.zeros(len(attrs), dtype=str)
        for i, att in enumerate(attrs):
            # Get predictions from each tree and calculate the majority
            preds = [tree.predict([att]) for tree in self.trees]
            predictions[i], _ = majority_label(preds)

        return predictions

    def prune(self):
        for tree in self.trees:
            tree.prune()

    # Saves a decision tree classifier as a .tree file
    def save_decision_tree(self, filename):
        # 'wb' = write-only file in binary mode
        with open(filename, 'wb') as tree_file:
            pickle.dump(self.trees, tree_file)

    # Constructs a decision tree classifier a previously saved .tree file
    def load_decision_tree(self, tree_filename):
        # 'rb' = read-only file in binary mode
        with open(tree_filename, 'rb') as tree_file:
            self.trees = pickle.load(tree_file)

        # Set a flag so that we know that the classifier has been trained
        self.is_trained = True
