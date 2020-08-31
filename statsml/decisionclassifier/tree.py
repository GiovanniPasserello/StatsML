class DecisionTree:

    def __init__(self, root):
        self.root = root

    def predict(self, x):
        return self.root.predict(x)

    def print_tree(self):
        self.root.print_node(1)
