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

        scalar = alpha / len(self.x)
        for i in range(num_iters):
            gradients = self.x.transpose().dot((self.x.dot(self.theta) - self.y))
            gradients[1:, :] += l1 * np.sign(self.theta[1:, :])  # l1 regularisation
            gradients[1:, :] += 2 * l2 * self.theta[1:, :]  # l2 regularisation
            self.theta -= scalar * gradients
            
    def compute_cost(self, l1, l2):
        """ Compute the cost for linear regression

        Returns:
            {float} -- the mean squared error of this linear model
            l1 {float} -- the l1 regularisation constant for cost calculations
            l2 {float} -- the l2 regularisation constant for cost calculations
        """

        diffs = self.x.dot(self.theta) - self.y
        loss = 1 / (2 * len(self.y)) * sum(diffs * diffs)

        loss += l1 * np.sign(self.theta).sum()  # l1 regularisation
        loss += l2 * np.square(self.theta).sum()  # l2 regularisation

        return loss

    def predict(self, x):
        """ Predict the output of the linear model against sample data using learned parameters

        Arguments:
            x {np.ndarray} -- An N x K dimensional numpy array of data samples to predict
        Returns:
            {[float]} -- a list of the predicted outputs of the model
        """

        return x.dot(self.theta)


def example_main(dat):
    # Hyper-parameters
    alpha = 0.1
    num_iters = 1000
    l1 = 0.01
    l2 = 0.001

    regressor = LinearRegressor(dat)
    print("Starting Cost:", regressor.compute_cost(l1, l2))
    print("Training...")
    regressor.gradient_descent(alpha=alpha, num_iters=num_iters, l1=l1, l2=l2)
    print("Final Cost:", regressor.compute_cost(l1, l2))
    print("Final Theta:\n", regressor.theta)


if __name__ == "__main__":
    print("Beginning Univariate Linear Regression:\n")
    univariate_dat = np.loadtxt("../../datasets/regression/linear_univariate.txt", delimiter=",")
    example_main(univariate_dat)

    print("\nBeginning Multivariate Linear Regression:\n")
    multivariate_dat = np.loadtxt("../../datasets/regression/linear_multivariate.txt", delimiter=",")
    example_main(multivariate_dat)
