import numpy as np


class LinearRegressor:
    """
    A linear regressor optimized through gradient descent
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

    def gradient_descent(self, alpha, num_iters, lmda):
        """ Perform gradient descent to fit theta to the data x, given y

        Arguments:
            alpha {float} -- the learning rate
            num_iters {int} -- the number of iterations to perform
            lmda {float} -- the l1 regularisation constant for gradient calculations
        """

        scalar = alpha / len(self.x)
        for i in range(num_iters):
            gradients = self.x.transpose().dot((self.x.dot(self.theta) - self.y))
            gradients[1:, :] += lmda / len(self.x) * abs(self.theta[1:, :])
            self.theta -= scalar * gradients

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

    def compute_cost(self):
        """ Compute the cost for linear regression

        Returns:
            {float} -- the mean squared error of this linear model
        """

        diffs = self.x.dot(self.theta) - self.y
        return 1 / (2 * len(self.y)) * sum(diffs * diffs)

    def predict(self, x):
        """ Predict the output of the linear model against sample data using learned parameters

        Arguments:
            x {np.ndarray} -- An N x K dimensional numpy array of data samples to predict
        Returns:
            {[float]} -- a list of the predicted outputs of the model
        """

        return x.dot(self.theta)


def example_main(dat):
    regressor = LinearRegressor(dat)
    print("Starting Cost:", regressor.compute_cost())
    print("Training...")
    regressor.gradient_descent(alpha=0.1, num_iters=400, lmda=0.1)
    print("Final Cost:", regressor.compute_cost())
    print("Final Theta:\n", regressor.theta)


if __name__ == "__main__":
    print("Beginning Univariate Linear Regression:\n")
    univariate_dat = np.loadtxt("../../datasets/regression/linear_univariate.txt", delimiter=",")
    example_main(univariate_dat)

    print("\nBeginning Multivariate Linear Regression:\n")
    multivariate_dat = np.loadtxt("../../datasets/regression/linear_multivariate.txt", delimiter=",")
    example_main(multivariate_dat)
