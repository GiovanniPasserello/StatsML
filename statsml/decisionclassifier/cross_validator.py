import numpy as np

from statsml.decisionclassifier import DecisionTreeClassifier
from statsml.decisionclassifier.utils import PRECISION
from statsml.metrics.confusion_evaluator import ConfusionEvaluator


class CrossValidator:

    def __init__(self, dataset, k):
        assert k > 1, "Cannot perform one fold, (empty training set)"
        assert k <= len(dataset), "Cannot perform more folds than the size of your dataset"
        self.k = k

        # Shuffle the dataset to randomly distribute it (same shuffle for attrs & labs)
        perm = np.random.permutation(len(dataset))
        self.attrs = dataset.attributes[perm]
        self.labs = dataset.labels[perm]

        # Size of test set for each fold
        self.chunk_size = len(self.attrs) // self.k
        # Number of test sets that will have one extra tuple (due to uneven division by k)
        self.num_larger_chunks = len(self.attrs) - k * self.chunk_size

    def cross_validate(self):
        print("\nCross validating over", self.k, "folds")

        # To keep track of the position of the test set for each fold within the dataset
        covered_range = 0

        # Accuracy for each fold
        accuracies = np.zeros(self.k)
        evaluator = ConfusionEvaluator()

        # Populate the accuracies array for each fold by training & testing on a specific split
        for i in range(self.k):
            # The size of this fold's test set (possibly larger for first few as described above)
            span = self.chunk_size + 1 if i < self.num_larger_chunks else self.chunk_size

            # Create a mask so we can take an index range for the test set, and the inverse range for the training set
            mask = np.ones(len(self.attrs), np.bool)
            mask[covered_range:covered_range + span] = 0

            # Take the test range
            test_atts = self.attrs[covered_range:covered_range + span]
            test_labs = self.labs[covered_range:covered_range + span]
            # The training set is the dataset without tuples corresponding to 0 in mask (test data)
            training_atts = self.attrs[mask]
            training_labs = self.labs[mask]
            covered_range += span

            # Train the model on the remaining training set
            classifier = DecisionTreeClassifier()
            print("\nBeginning training for fold", i + 1)
            classifier.train(training_atts, training_labs)
            classifier.save_decision_tree('trees/folds/fold' + str(i + 1) + ".tree")

            # Get predictions on the test set from the trained model
            print("Beginning prediction for fold", i + 1)
            preds = classifier.predict(test_atts)

            print("Predicted:")
            print(preds)
            print("Actual:")
            print(test_labs)

            # Pass the class_labels to ensure the entire label space is spanned (in case of small dataset)
            conf = evaluator.confusion_matrix(preds, test_labs, class_labels=np.unique(self.labs))
            accuracies[i] = evaluator.accuracy(conf)
            print("Fold", i + 1, "Accuracy:", round(accuracies[i], PRECISION))

        return accuracies
