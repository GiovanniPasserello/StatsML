import numpy as np


class ConfusionEvaluator(object):
    """
    Collection of evaluation metrics for confusion matrices
    """

    @staticmethod
    def confusion_matrix(prediction, annotation, class_labels=None):
        """ Compute a confusion matrix from predictions and actual values (annotation)

        Arguments:
            prediction {np.array} --
                an N dimensional numpy array containing the predicted class labels
            annotation {np.array} --
                an N dimensional numpy array containing the ground truth class labels
            class_labels {np.array} --
                a C dimensional numpy array containing the ordered set of class labels
        Returns:
            {np.array} --
                a C by C matrix, where C is the number of classes.
                Classes should be ordered by class_labels.
                Rows are ground truth per class (annotation), columns are predictions (prediction).
        """

        # If None, default to all unique annotation class labels
        if class_labels is None:
            class_labels = np.unique(annotation)

        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)

        for i in range(len(prediction)):
            truth, pred = annotation[i], prediction[i]

            ind1 = np.where(class_labels == truth)[0][0]
            ind2 = np.where(class_labels == pred)[0][0]
            confusion[ind1][ind2] += 1

        return confusion

    @staticmethod
    def accuracy(confusion):
        """ Compute the accuracy given a confusion matrix

        Arguments:
            confusion {np.array} --
                The confusion matrix (C by C, where C is the number of classes).
                Rows are ground truth per class, columns are predictions.
        Returns:
            {float} -- The accuracy (between 0.0 to 1.0 inclusive)
        """

        total = np.sum(confusion)
        return np.trace(confusion) / total if total else 0

    @staticmethod
    def precision(confusion):
        """ Computes the precision score per class given a confusion matrix. Also returns macro-averaged precision.

        Arguments:
            confusion {np.array} --
                The confusion matrix (C by C, where C is the number of classes).
                Rows are ground truth per class, columns are predictions.
        Returns:
            {np.array} -- A C-dimensional numpy array, with the precision score for each class of the confusion matrix.
            {float} -- The macro-averaged precision score across C classes.
        """

        # Calculate all column sums, then for each column calculate (true +ve / total column)
        col_sums = confusion.sum(axis=0)
        precisions = np.array([confusion[i][i] / col_sums[i] if col_sums[i] else 0 for i in range(len(confusion))])

        return precisions, np.mean(precisions)

    @staticmethod
    def recall(confusion):
        """ Computes the recall score per class given a confusion matrix. Also returns macro-averaged recall.

        Arguments:
            confusion {np.array} --
                The confusion matrix (C by C, where C is the number of classes).
                Rows are ground truth per class, columns are predictions.
        Returns:
            {np.array} -- A C-dimensional numpy array, with the recall score for each class of the confusion matrix.
            {float} -- The macro-averaged recall score across C classes.
        """

        # Calculate all row sums, then for each row calculate (true +ve / total row)
        row_sums = confusion.sum(axis=1)
        recalls = np.array([confusion[i][i] / row_sums[i] if row_sums[i] else 0 for i in range(len(confusion))])

        return recalls, np.mean(recalls)

    @staticmethod
    def f1_score(confusion):
        """ Computes the f1 score per class given a confusion matrix. Also returns macro-averaged f1-score.

        Arguments:
            confusion {np.array} --
                The confusion matrix (C by C, where C is the number of classes).
                Rows are ground truth per class, columns are predictions.
        Returns:
            {np.array} -- A C-dimensional numpy array, with the f1 score for each class of the confusion matrix.
            {float} -- The macro-averaged f1 score across C classes.
        """

        # Initialise array to store recall for C classes
        f1s = np.zeros((len(confusion), ))

        precisions, _ = ConfusionEvaluator.precision(confusion)
        recalls, _ = ConfusionEvaluator.recall(confusion)

        for i in range(len(f1s)):
            numer = precisions[i] * recalls[i]
            denom = precisions[i] + recalls[i]
            f1s[i] = 2 * numer / denom if denom else 0

        return f1s, np.mean(f1s)
