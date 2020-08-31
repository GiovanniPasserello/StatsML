from math import exp

import numpy as np

# A small constant to avoid division by 0 in categorical cross entropy calculations
EPSILON = 1e-10


class LogisticRegressor:
    """
    A logistic regressor optimized through gradient descent
    """

    def __init__(self, dat, theta=None):
        # number of samples and features in the dataset
        self.num_samples = dat.shape[0]
        self.num_features = dat.shape[1]

        # For feature normalization
        self.mu = None
        self.sigma = None

        # row for the bias terms
        biases = np.ones(self.num_samples)
        normalized_features = self.feature_normalize(dat[:, :-1])
        self.x = np.column_stack((biases, normalized_features))
        self.y = np.reshape(dat[:, -1], (self.num_samples, 1))

        self.theta = theta if theta else np.zeros((self.num_features, 1))

    def feature_normalize(self, x):
        """
        Normalize the features in x
        After normalization the mean value of each feature is 0, and the standard deviation is 1

        Arguments:
            x {np.ndarray} -- An N x K dimensional numpy array of data samples
        Returns:
            {np.ndarray} -- An N x K dimensional numpy array of normalized data samples
        """

        # Initialize normalization params on training data set
        if self.mu is None or self.sigma is None:
            self.mu = np.mean(x, axis=0)
            self.sigma = np.std(x, axis=0)
            zero_mask = self.sigma == 0
            self.sigma += zero_mask  # ensure no division by zero (if == 0, set = 1)

        return (x - self.mu) / self.sigma

    def gradient_descent(self, alpha, num_iters, l1, l2):
        """ Perform gradient descent to fit theta to the data x, given y

        Arguments:
            alpha {float} -- the learning rate
            num_iters {int} -- the number of iterations to perform
            l1 {float} -- the l1 regularisation constant for gradient calculations
            l2 {float} -- the l2 regularisation constant for gradient calculations
        """

        scalar = alpha / len(self.y)
        for i in range(num_iters):
            gradient = self.compute_gradient(l1, l2)
            self.theta -= scalar * gradient

    def compute_gradient(self, l1, l2):
        """ Compute the elementwise regularised gradient of theta for logistic regression

        Arguments:
            l1 {float} -- the l1 regularisation constant for gradient calculations
            l2 {float} -- the l2 regularisation constant for gradient calculations
        Returns:
            {float} -- the elementwise gradient of theta for this logistic model
        """

        pred = self.x.dot(self.theta)
        sigmoid_pred = self.sigmoid(pred)
        err = sigmoid_pred - self.y

        gradients = self.x.transpose().dot(err)
        gradients[1:, :] += l1 * np.sign(self.theta[1:, :])  # l1 regularisation
        gradients[1:, :] += 2 * l2 * self.theta[1:, :]  # l2 regularisation

        return gradients

    def compute_cost(self, l1, l2):
        """ Compute the cost for logistic regression

        Returns:
            {float} -- the cross entropy cost of this logistic model
            l1 {float} -- the l1 regularisation constant for cost calculations
            l2 {float} -- the l2 regularisation constant for cost calculations
        """

        scalar = -1 / len(self.y)
        pred = self.x.dot(self.theta)
        sigmoid_pred = self.sigmoid(pred)

        # Add small EPSILON to each term to avoid division by zero
        first_term = self.y.transpose().dot(np.log(sigmoid_pred + EPSILON))
        second_term = (1 - self.y).transpose().dot(np.log(1 - sigmoid_pred + EPSILON))

        loss = scalar * (first_term + second_term).sum()
        loss += l1 * np.sign(self.theta).sum()  # l1 regularisation
        loss += l2 * np.square(self.theta).sum()  # l2 regularisation

        return loss

    def predict(self, x):
        """ Predict the output of the logistic model against sample data using learned parameters

        Arguments:
            x {np.ndarray} -- An N x K dimensional numpy array of data samples to predict
        Returns:
            {[float]} -- a list of the predicted outputs of the model, clips to 0 or 1
        """

        pred = x.dot(self.theta)
        return self.sigmoid(pred) >= 0.5

    @staticmethod
    def sigmoid(x):
        # Vectorized method is faster than broadcasted exponential
        vectorized_sigmoid = np.vectorize(LogisticRegressor.single_sigmoid)
        return vectorized_sigmoid(x)

    @staticmethod
    def single_sigmoid(x):
        return 1 / (1 + exp(-x))


def example_main(dat):
    # Hyper-parameters
    alpha = 0.1
    num_iters = 1500
    l1 = 0.1
    l2 = 0.01

    regressor = LogisticRegressor(dat)
    print("Starting Cost:", regressor.compute_cost(l1, l2))
    print("Training...")
    regressor.gradient_descent(alpha=alpha, num_iters=num_iters, l1=l1, l2=l2)
    print("Final Cost:", regressor.compute_cost(l1, l2))
    print("Final Theta:\n", regressor.theta)


if __name__ == "__main__":
    print("Beginning Multivariate Logistic Regression 1:\n")
    multivariate_dat1 = np.loadtxt("../../datasets/regression/logistic_multivariate1.txt", delimiter=",")
    example_main(multivariate_dat1)

    print("\nBeginning Multivariate Logistic Regression 2:\n")
    multivariate_dat2 = np.loadtxt("../../datasets/regression/logistic_multivariate2.txt", delimiter=",")
    example_main(multivariate_dat2)
