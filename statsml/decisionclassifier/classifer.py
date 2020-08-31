import numpy as np
import pickle
import time

from statsml.decisionclassifier.dataset import Dataset
from statsml.decisionclassifier.tree import DecisionTree
from statsml.decisionclassifier.tree_node import DecisionTreeNode
from statsml.decisionclassifier.utils import DEFAULT_MIN_SPLIT_SIZE, DEFAULT_MIN_NODE_SIZE, PRECISION, merge_histograms
from statsml.metrics.confusion_evaluator import ConfusionEvaluator


class DecisionTreeClassifier(object):
    """
    A decision tree classifier

    Attributes:
        is_trained {bool} -- Keeps track of whether the classifier has been trained
    Methods:
        train(X, y) -- Constructs a decision tree from data X and label y
        predict(X) -- Predicts the class label of samples X
    """

    def __init__(self):
        self.is_trained = False
        self.tree = None
        self.dataset = None
        self.start = None
        self.progress_target = 0.1
        self.progress = 0
        self.validation = None

    def train(self, attributes, labels, prune=True, validation=None, valid_features=None,
              min_split_size=DEFAULT_MIN_SPLIT_SIZE, min_node_size=DEFAULT_MIN_NODE_SIZE):
        """ Constructs a decision tree classifier from data

        Arguments:
            attributes {np.ndarray} -- An N by K numpy array (N samples of K features)
            labels {np.ndarray} -- An N-dimensional numpy array (target outputs, i.e. true labels)
            prune {Boolean} -- Whether or not the trained tree should be automatically pruned (default True)
            validation {Dataset} -- Optional validation dataset used for pruning evaluation
            valid_features {np.ndarray} --
                Contains the indices of all attributes we can split on.
                Used for random forests which are allocated a subset of features.
            min_split_size {int} -- Detailed in utils.py (but tunable upon request for hyper-parameter tuning)
            min_node_size {int} -- Detailed in utils.py (but tunable upon request for hyper-parameter tuning)
        Returns:
            {DecisionTreeClassifier} -- A copy of the DecisionTreeClassifier instance
        """

        assert attributes.shape[0] == len(labels), "Training failed: Missing training data."

        self.dataset = Dataset(attributes, labels, min_split_size=min_split_size, min_node_size=min_node_size)
        self.progress_target = 0.1  # used to keep track of progress in 10% increments for console output
        self.progress = 0

        # If not random forest, split over all attributes
        if valid_features is None:
            valid_features = range(len(attributes[0]))

        print("Training...")
        self.start = time.time()
        self.tree = DecisionTree(self.build_tree_node(self.dataset, valid_features))
        end = round(time.time() - self.start, PRECISION)
        print("Finished training in", str(end) + "s")

        self.is_trained = True

        if prune:
            self.validation = validation
            self.prune()

        return self

    # Create a tree node given a dataset
    def build_tree_node(self, dataset, valid_features):
        # Base case: All labels are the same so this node will be a leaf
        if len(set(dataset.labels)) <= 1:
            self.update_train_progress(dataset)
            return DecisionTreeNode.make_leaf(dataset.lab_hist())

        # Find the attribute index and split values resulting in the 'optimal' split of the dataset
        att_ind, split_vals = dataset.find_optimal_split_att_and_vals(valid_features)
        if att_ind is None:
            self.update_train_progress(dataset)
            # There is no possible split so this node will be a leaf with the given label distribution
            #   e.g. this could be because all child nodes after a split are < min_node_size
            return DecisionTreeNode.make_leaf(dataset.lab_hist())

        # Build an internal split node based on the attribute index and split values and then split the dataset
        node = DecisionTreeNode.make_internal(att_ind, split_vals)
        child_datasets = dataset.split(att_ind, split_vals)

        # Recursively build the tree for each child dataset split
        for child_ds in child_datasets:
            node.add_child(self.build_tree_node(child_ds, valid_features))

        # Merge children if their subtrees are the same (i.e. have the same split points and labels at the leaves)
        node.merge_children()

        return node

    # Used to display progress updates in console
    #   len(self.dataset) is the entire training data size
    #   self.progress tracks how much we have covered so far
    #   self.progress_target tracks which 10% increment we are next to hit
    def update_train_progress(self, dataset):
        self.progress += len(dataset)
        while self.progress / len(self.dataset) > self.progress_target:
            print(
                "Trained {0:.0f}% of rows after {1:.4f}s".format(
                    self.progress_target * 100, time.time() - self.start
                ))
            self.progress_target += 0.1

    def predict(self, attrs):
        """ Predict a set of samples using the trained DecisionTreeClassifier.

        Arguments:
            attrs {np.ndarray} --
                An N by K numpy array of attribute / feature samples (N is # of samples, K is # of attributes)
        Returns:
            {np.ndarray}
                An N-dimensional numpy array containing the predicted class label for each instance in attrs
        """

        assert self.is_trained, "Decision Tree classifier has not yet been trained."

        return np.array([self.tree.predict(att) for att in attrs])

    # Prune the tree starting from the root, evaluating against the validation set
    def prune(self):
        assert self.is_trained, "Cannot prune an untrained tree."
        assert self.dataset is not None, "No dataset provided to prune on."

        print("Pruning...")
        start = time.time()
        self.tree = DecisionTree(self.prune_node(self.tree.root))
        end = round(time.time() - start, PRECISION)
        print("Finished pruning in", str(end) + "s")

    # Greedy accuracy-based pruning algorithm
    def prune_node(self, node):
        # Base case: No further pruning possible
        if node.is_leaf:
            return node

        # Prune all children
        for i in range(len(node.children)):
            node.children[i] = self.prune_node(node.children[i])

        # If all children are leaves, we can prune this node based on increased validation accuracy
        if all(map(lambda c: c.is_leaf, node.children)):
            # Get current accuracy on validation set pre-prune
            preds = self.predict(self.validation.attributes)
            conf = ConfusionEvaluator.confusion_matrix(preds, self.validation.labels)
            acc_pre_prune = ConfusionEvaluator.accuracy(conf)

            # Simulate pruning by temporarily setting this node to be a leaf
            # Create histogram of total child labels, this node then represents a leaf with the max frequency label
            node.is_leaf = True
            node.label_cnts = {}
            for child in node.children:
                node.label_cnts = merge_histograms(node.label_cnts, child.label_cnts)

            # Re-evaluate accuracy with the current node 'pruned'
            preds = self.predict(self.validation.attributes)
            conf = ConfusionEvaluator.confusion_matrix(preds, self.validation.labels)
            acc_post_prune = ConfusionEvaluator.accuracy(conf)

            # Greedily prune if the accuracy is higher to further generalise the tree to unseen data
            if acc_post_prune > acc_pre_prune:
                print("Accuracy on validation.txt, pre-prune:", acc_pre_prune, "post-prune:", acc_post_prune)
                # Keep this node as a leaf and forget about any children
                node.children = []
                node.split_values = None
                node.attribute_index = None
            else:
                # Reset and keep this node as internal
                node.label_cnts = {}
                node.is_leaf = False

        return node

    # Saves a decision tree classifier as a .tree file
    def save_decision_tree(self, filename):
        # 'wb' = write-only file in binary mode
        with open(filename, 'wb') as tree_file:
            pickle.dump(self.tree, tree_file)

    # Constructs a decision tree classifier a previously saved .tree file
    def load_decision_tree(self, tree_filename):
        # 'rb' = read-only file in binary mode
        with open(tree_filename, 'rb') as tree_file:
            self.tree = pickle.load(tree_file)

        # Set a flag so that we know that the classifier has been trained
        self.is_trained = True

    def print_decision_tree(self):
        assert self.is_trained, "Cannot print an untrained tree."
        self.tree.print_tree()
