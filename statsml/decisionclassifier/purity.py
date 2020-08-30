import math
import numpy as np


class PurityFuncs:

    @staticmethod
    def entropy(labels):
        """ Calculate label entropy

        Arguments:
            labels {numpy.array} -- A single dimensional array that we wish to calculate the entropy of
        Returns:
            {int} -- The entropy of the labels
        """

        total = len(labels)
        unique_labels, counts = np.unique(labels, return_counts=True)
        label_counts = dict(zip(unique_labels, counts))

        entropy = 0
        for label in unique_labels:
            prob = label_counts[label] / total
            entropy -= prob * math.log2(prob)
        return entropy

    @staticmethod
    def gini(labels):
        """ Calculates gini impurity of a given node's labels

        Arguments:
            labels {numpy.array} -- A single dimensional array that we wish to calculate the entropy of
        Returns:
            {int} -- The gini impurity of the node
                For J classes, an impurity of (J-1)/J indicates an even distribution
                e.g. gini(['a','a','b','b','c','c') = 2 / 3 = 0.66667
        """

        total = len(labels)
        _, counts = np.unique(labels, return_counts=True)
        return 1 - np.sum(list(map(lambda p: (p / total) ** 2, counts)))
