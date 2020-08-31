import random

from statsml.decisionclassifier import utils
from statsml.decisionclassifier.purity import PurityFuncs


class DecisionTreeNode:

    def __init__(self, is_leaf, parent=None, attribute_index=None, split_values=None, label_cnts=None):
        self.attribute_index = attribute_index  # Index of attribute ot split on
        self.split_values = split_values  # Split points e.g. (3 <= x <= 5) => [3, 5]
        self.is_leaf = is_leaf
        self.label_cnts = label_cnts  # Dictionary of label to frequency
        self.majority_label = None
        self.parent = parent
        self.children = []

    # Check if two leaves are equal, or two subtrees are equivalent
    def __eq__(self, other):
        if type(self) != type(other) or self.is_leaf != other.is_leaf:
            return False

        if self.is_leaf and other.is_leaf:
            return self.get_majority_label() == other.get_majority_label()
        else:
            return self.attribute_index == other.attribute_index and \
                   self.split_values == other.split_values and \
                   all(map(lambda p: p[0] == p[1], zip(self.children, other.children)))

    @staticmethod
    def make_leaf(label_cnts, parent=None):
        return DecisionTreeNode(is_leaf=True, label_cnts=label_cnts, parent=parent)

    @staticmethod
    def make_internal(attribute_index, split_values, parent=None):
        return DecisionTreeNode(
            is_leaf=False, attribute_index=attribute_index, split_values=split_values, parent=parent
        )

    def add_child(self, child):
        assert not self.is_leaf
        child.parent = self
        self.children.append(child)

    def merge_children(self):
        i = 0
        while i < len(self.children) - 1:
            # If two neighbouring children are leaves and have the same majority label, they can be combined into a
            # single leaf with that majority label. Since we use <= we delete the lower child.
            # e.g. for child leaves with split values 4.5 and 7.5 both mapping to class A we can delete the child with
            # the split value 4.5 since that range is covered by <= 7.5
            if self.children[i] == self.children[i + 1]:
                self.children[i] = utils.merge_nodes(self.children[i], self.children[i + 1])
                del self.children[i + 1]
                del self.split_values[i]
            else:
                i += 1

        # If there is only a single child left, fill the current node with its values (may be leaf or internal)
        if len(self.children) == 1:
            child = self.children[0]
            self.__init__(
                is_leaf=child.is_leaf,
                parent=self.parent,
                label_cnts=child.label_cnts,
                attribute_index=child.attribute_index,
                split_values=child.split_values
            )

    # Return the most commonly occurring label in this node's dictionary label set (if  multiple, randomly choose one)
    def get_majority_label(self):
        if self.majority_label is None:
            max_freq = max(self.label_cnts.values())  # Get the highest label frequency
            max_list = [key for key, val in self.label_cnts.items() if val == max_freq]  # All labels with this freq
            rand_max = random.randrange((len(max_list)))  # Choose a random one
            self.majority_label = max_list[rand_max]
        return self.majority_label

    # Predict the output label for the given attribute values
    def predict(self, x):
        if self.is_leaf:
            return self.get_majority_label()

        # Find the first split value that is greater than or equal to the attribute value, recurse into this child.
        # e.g. if x[0] == 10   split_values=[3.5, 7.5, 12.5, 15.5]  it would recurse into the 3rd child.
        #                       children=[ c1,  c2,  c3,   c4,  c5]
        #                                            ^^
        for i, val in enumerate(self.split_values):
            if val >= x[self.attribute_index]:
                return self.children[i].predict(x)

        # Greater than all the split values => recurse into the last child
        return self.children[-1].predict(x)

    '''
    BEGIN Node visualisation
    '''

    # Format the node for output in visualisation
    def __str__(self):
        if len(self.children) != 0:
            return "Split: att[{0}]".format(self.attribute_index)
        else:
            labs = utils.expand_lab_hist(self.label_cnts)  # Expand the histogram to a single list
            entropy = round(PurityFuncs.entropy(labs), utils.PRECISION)
            gini = round(PurityFuncs.gini(labs), utils.PRECISION)
            return "Label: {0}\nEntropy={1}\nGini={2}\nClasses:{3}".format(
                self.get_majority_label(), entropy, gini, self.sanitise_lab_hist()
            )

    # Format the label histogram for output in visualisation
    def sanitise_lab_hist(self):
        s = "{\n"
        for k, v in self.label_cnts.items():
            s += "'" + k + "': " + str(v) + ",\n"
        return s + "}"

    # Recursively print the tree from this node
    def print_node(self, level):
        print(str(self).replace('\n', ' '))
        for child in self.children:
            print('\t' * level, end="")  # Separate
            child.print_node(level + 1)  # Children at a lower level need to be spaced out
