import numpy as np

from statsml.decisionclassifier.classifer import DecisionTreeClassifier
from statsml.decisionclassifier.cross_validator import CrossValidator
from statsml.decisionclassifier.dataset import Dataset
from statsml.decisionclassifier.utils import PRECISION, majority_label
from statsml.metrics.confusion_evaluator import ConfusionEvaluator

"""
Example usage of DecisionTreeClassifier on fake dataset
"""


def standard_example_main():
    print("Creating the decision tree...")
    classifier = DecisionTreeClassifier()

    print("Loading the training dataset...")
    x = np.array([
        [5, 7, 1],
        [4, 6, 2],
        [4, 6, 3],
        [1, 3, 1],
        [2, 1, 2],
        [5, 2, 6]
    ])
    y = np.array(["A", "A", "A", "C", "C", "C"])

    print("\nTraining the decision tree...")
    classifier = classifier.train(x, y, prune=False)

    print("\nLoading the test set...")
    x_test = np.array([
        [1, 6, 3],
        [0, 5, 5],
        [1, 5, 0],
        [2, 4, 2]
    ])
    y_test = np.array(["A", "A", "C", "C"])
    classes = ["A", "C"]

    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))

    print("\nEvaluating test predictions...")
    evaluator = ConfusionEvaluator()

    print("Confusion matrix:")
    confusion = evaluator.confusion_matrix(predictions, y_test)
    print(confusion)

    accuracy = evaluator.accuracy(confusion)
    print("\nAccuracy: {}".format(accuracy))

    (p, macro_p) = evaluator.precision(confusion)
    (r, macro_r) = evaluator.recall(confusion)
    (f, macro_f) = evaluator.f1_score(confusion)

    print("\nClass: Precision, Recall, F1")
    for (i, (p1, r1, f1)) in enumerate(zip(p, r, f)):
        print("{}: {:.2f}, {:.2f}, {:.2f}".format(classes[i], p1, r1, f1));

    print("\nMacro-averaged Precision: {:.2f}".format(macro_p))
    print("Macro-averaged Recall: {:.2f}".format(macro_r))
    print("Macro-averaged F1: {:.2f}".format(macro_f))


# Perform a function (e.g. max/min) on an array and return it's value and which fold it belongs to
def find_arr_metric(func, arr):
    metric = func(arr)
    return round(metric, PRECISION), np.where(arr == metric)[0][0] + 1  # '+ 1' as folds are 1-k


# Perform k-fold cross-validation on a given file of example data
# Each sample in the dataset should be of format: 'att1,att2,att3,...,attn,label'
def cross_validation_example_main(file, k):
    dataset = Dataset.load_from_file(file)
    validator = CrossValidator(dataset, k)

    # Perform cross validation, and calculate the time taken to run
    import time
    start = time.time()
    accuracies = validator.cross_validate()
    end = time.time()

    # Calculate notable metrics
    mn, mn_fold = find_arr_metric(np.min, accuracies)
    mx, mx_fold = find_arr_metric(np.max, accuracies)
    mean = round(accuracies.mean(), PRECISION)
    std = round(accuracies.std(), PRECISION)
    time = round((end - start) / k, PRECISION)  # Average fold time

    print("\nCross Validation Over", k, "Folds Complete:")
    print("Avg Time per fold:", str(time) + "s")
    print("Min Accuracy (fold", str(mn_fold) + "):", mn)
    print("Max Accuracy (fold", str(mx_fold) + "):", mx)
    print("Mean Accuracy:", mean, "+-", std, "\n")

    # Return the max-accuracy fold to get the max fold for use later (Question 3.4)
    return mx_fold


if __name__ == "__main__":
    standard_example_main()
