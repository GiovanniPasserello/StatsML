import numpy as np
import random

# DEFAULT_MIN_SPLIT_SIZE is used to govern the number of splits allowed at any node. We take the number of unique
# attribute values at each node and divide it by this to calculate the number of unique attribute values allowed
from statsml.decisionclassifier.tree_node import DecisionTreeNode

DEFAULT_MIN_SPLIT_SIZE = 5

# DEFAULT_MIN_NODE_SIZE is used to define the minimum number of class labels in a node.
# This means that we don't split a node further if there are <= x labels in it
DEFAULT_MIN_NODE_SIZE = 1

# Numerical precision for output
PRECISION = 4


# Build a histogram from a list of labels (dictionary of label to frequency)
def build_histogram(arr):
    """ Build a histogram (dictionary of label to frequency) from a list of labels

    Arguments:
        arr {np.ndarray} -- A numpy array of class labels
    Returns:
        {dict<class_label:int>} -- A dictionary of unique class labels to their frequency in arr
    """

    unique, counts = np.unique(arr, return_counts=True)
    return dict(zip(unique, counts))


# Efficiently aggregate the entries/counts of two histograms into a single one
def merge_histograms(h1, h2):
    if h1 is None:
        return h2
    if h2 is None:
        return h1

    h3 = {**h1, **h2}
    for key, value in h3.items():
        if key in h1 and key in h2:
            h3[key] = value + h1[key]
    return h3


# Merge the contents of two DecisionTreeNodes into a single node
def merge_nodes(n1, n2):
    assert n1 == n2

    # If both are leaves, combine the two directly
    if n1.is_leaf and n2.is_leaf:
        h1 = n1.label_cnts
        h2 = n2.label_cnts
        return DecisionTreeNode.make_leaf(merge_histograms(h1, h2), parent=n1.parent)

    # If not both leaves, combine the children pairwise and recursively
    merged = DecisionTreeNode.make_internal(n1.attribute_index, n1.split_values, n1.parent)
    for c1, c2 in zip(n1.children, n2.children):
        merged.add_child(merge_nodes(c1, c2))
    return merged


# Return the most frequent label and it's count (if multiple, choose one at random)
def majority_label(labels):
    labs, cnts = np.unique(labels, return_counts=True)
    label_counts = list(zip(labs, cnts))
    _, max_freq = max(label_counts, key=lambda p: p[1])
    max_list = [lab for (lab, cnt) in label_counts if cnt == max_freq]
    rand_max = random.randrange((len(max_list)))
    return max_list[rand_max], max_freq


# Expand a histogram into a 1D list of labels
def expand_lab_hist(h):
    labs = []
    for k, v in h.items():
        labs.extend([k] * v)
    return labs


# Return a copy of arr with el appended to it
def copy_append(arr, el):
    arr2 = arr.copy()
    arr2.append(el)
    return arr2
