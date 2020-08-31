import numpy as np

from statsml.decisionclassifier.purity import PurityFuncs
from statsml.decisionclassifier.utils import copy_append, DEFAULT_MIN_SPLIT_SIZE, DEFAULT_MIN_NODE_SIZE, build_histogram


class Dataset:

    def __init__(self, attributes, labels, min_split_size=DEFAULT_MIN_SPLIT_SIZE, min_node_size=DEFAULT_MIN_NODE_SIZE):
        assert attributes.shape[0] == len(labels), "Label and attribute arrays must be same size"
        self.attributes = attributes
        self.labels = labels
        self.min_split_size = min_split_size
        self.min_node_size = min_node_size

    def __len__(self):
        return len(self.labels)

    def __str__(self):
        return str(build_histogram(self.labels))

    def num_attributes(self):
        return len(self.attributes[0]) if len(self.attributes) else 0

    # Efficiently fill the dataset's attributes and labels from a comma separated file
    # Expected format for a single sample is: 'att1,att2,att3,...,attn,label'
    @staticmethod
    def load_from_file(filename, min_split_size=DEFAULT_MIN_SPLIT_SIZE, min_node_size=DEFAULT_MIN_NODE_SIZE):
        with open(filename) as file:
            attributes = []
            labels = []
            for line in file.readlines():
                split_line = line.rstrip().split(',')  # strip to remove spaces and '\n' from label
                attributes.append(list(map(int, split_line[:-1])))
                labels.append(split_line[-1])

        return Dataset(np.asarray(attributes), np.asarray(labels), min_split_size, min_node_size)

    def filter_by_attribute(self, attribute_index, pred):
        """ Returns a new dataset containing only those tuples which have an attribute at the specified index that
            satisfy the predicate, pred

        Arguments:
            attribute_index {int} -- The index of the attribute we want to filter on
            pred {int -> Boolean} -- A function that returns whether a given attribute value matches a predicate
        Returns:
            {Dataset} -- The filtered dataset
        """

        # pair[0] to fetch the sample's array of attributes from the zipped pair
        filtered = filter(lambda pair: pred(pair[0][attribute_index]), zip(self.attributes, self.labels))
        attributes, labels = zip(*filtered)  # Unzip the filtered tuples
        return Dataset(np.asarray(attributes), np.asarray(labels), self.min_split_size, self.min_node_size)

    def split(self, attribute_index, split_values):
        """ Splits current dataset into a list of subsets using the given attribute and split points.

        Arguments:
            attribute_index {int} -- The index of the attribute we want to use for the split
            split_values {[int]} --
                A list of end points to use as the end points of the split buckets.
                E.g. [1, 2] creates three buckets  [ att <= 1, 1 < att <= 2, 2 < att ]
        Returns:
            {[Dataset]} -- The list of subsets after the splitting operation was completed
        """

        assert len(split_values) > 0, "List of split values cannot be empty"

        # Get the dataset from the first split value
        datasets = [self.filter_by_attribute(attribute_index, lambda att: att <= split_values[0])]
        # Go through each split pair and get the next resulting dataset
        for i in range(1, len(split_values)):
            datasets.append(
                self.filter_by_attribute(attribute_index, lambda att: split_values[i - 1] < att <= split_values[i])
            )
        # Get the dataset from the last split value
        datasets.append(self.filter_by_attribute(attribute_index, lambda att: att > split_values[-1]))

        return datasets

    def statistical_gain(self, children, purity_func):
        """ Calculates the statistical gain of a given split dataset using the purity function specified

        Arguments:
            children {[Dataset]} -- A list of datasets that has been split using an attribute / split value combination
            purity_func {np.ndarray -> int} --
                A function mapping a numpy array of labels to an integer representing the purity of the list of labels
        Returns:
            {int} -- The statistical gain of the given split
        """

        # Information gain is: purity(parent) - sum(weight(child[i]) * purity(child[i]))
        gain = purity_func(self.labels)  # Start with the parent purity
        for child in children:
            purity = purity_func(child.labels)  # Calculate child purity
            w = len(child) / len(self)  # Weight child's purity by the proportion of parent that is occupied by child
            gain -= w * purity  # Accumulate the information gain
        return gain

    # Brute force every subsequence of split indices for the unique splitting attributes under size max_splits
    #   e.g. all_split_points(3, 2) = [[0], [1], [2], [0, 1], [0, 2], [1, 2]]
    @staticmethod
    def all_split_points(num_unique, max_splits):
        arr = [[i] for i in range(num_unique)]
        last = arr.copy()
        for _ in range(max_splits - 1):
            last = [copy_append(a, x) for a in last for x in range(a[-1] + 1, num_unique)]
            arr.extend(last)
        return sorted(arr, key=len)

    # Calculate the splitting indices for this attribute index with the highest information gain
    def find_optimal_split_for_att(self, att_ind):
        assert len(set(self.attributes[:, att_ind])) > 1, "Attribute cannot be split further"

        max_gain = float("-inf")
        opt_split = None

        # Get all unique values in the dataset for the att_ind column of attributes
        unique_vals = sorted(list(set(self.attributes[:, att_ind])))
        # The maximum number of splits allowed for a node with n unique values
        # This aims to create a more generalised tree, not one that is overfitted and splits for each sample directly
        max_splits = round(len(unique_vals) / self.min_split_size)
        # Every possible subsequence of the split indices to test
        split_points = self.all_split_points(len(unique_vals) - 1, max_splits)

        for points in split_points:
            # Get a list of the average values between each neighbouring unique attribute pair as a split value
            split_vals = list(map(lambda p: (unique_vals[p] + unique_vals[p + 1]) / 2, points))
            # Calculate the dataset of the children generated by this split
            child_datasets = self.split(att_ind, split_vals)
            # Not a valid split if the child datasets are smaller than the minimum node size
            if not all(map(lambda ds: len(ds) >= self.min_node_size, child_datasets)):
                continue
            # Calculate the information gain of this simulated split using Gini impurity
            gain = self.statistical_gain(child_datasets, PurityFuncs.gini)
            if gain > max_gain:
                opt_split = split_vals  # Store this splitting rule if it has the max gain
                max_gain = gain

        return max_gain, opt_split

    # Find the optimal attribute to split on and its optimal splitting indices
    def find_optimal_split_att_and_vals(self, valid_features):
        max_gain = float("-inf")
        opt_att = None
        opt_split = None

        for att_ind in valid_features:
            # There must be more than one unique attribute value
            if len(set(self.attributes[:, att_ind])) <= 1:
                continue

            # Calculate the optimal splitting indices for this attribute
            att_gain, att_split = self.find_optimal_split_for_att(att_ind)

            # Skip the attribute if it could not be split
            if att_gain == float("inf") or att_split is None:
                continue

            if att_gain > max_gain:
                opt_att = att_ind
                opt_split = att_split
                max_gain = att_gain

        return opt_att, opt_split

    def lab_hist(self):
        return build_histogram(self.labels)
