import numpy as np

"""
Naming Conventions:
    x -> input to the first linear layer
    z -> output of a linear layer / input to a non-linear layer
    a -> output of a non-linear layer / input to the next linear layer
    grad_z -> gradient passing through to a linear layer
    grad_a -> gradient passing through to a non-linear layer
"""


class Layer:
    """
    Abstract layer class
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class LinearLayer(Layer):
    """
    LinearLayer: Perform affine transformation of input
    """

    @staticmethod
    def _xavier_init(size, gain=1.0):
        """
        Xavier initialization of network weights
          - Faster convergence than random initialization
          - Mitigates saturated or dead gradients
        """
        low = -gain * np.sqrt(6.0 / np.sum(size))
        high = gain * np.sqrt(6.0 / np.sum(size))
        return np.random.uniform(low, high, size)

    def __init__(self, n_in, n_out):
        """
        Arguments:
            n_in {int} -- dimension of input
            n_out {int} -- dimension of output
        """
        self.n_in = n_in
        self.n_out = n_out

        # Learnable Parameters - weights initialised by Xavier, biases set to 0
        self._W = self._xavier_init((n_in, n_out))
        self._b = np.zeros(n_out)

        # Backpropagation attributes
        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns Wx + b).
        Stores information needed to compute gradient at a later stage in _cache_current

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in)
        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        assert x.shape[1] == self._W.shape[0]

        # Store the input for later backpropagation
        self._cache_current = x
        # XW + b
        return np.dot(x, self._W) + self._b

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) w.r.t. the output of this layer, perform back pass
        through the layer (i.e. compute gradients of loss w.r.t. parameters of layer and inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out)
        Returns:
            {np.ndarray} -- Array containing gradients w.r.t. layer input of shape (batch_size, n_in).
        """
        assert self.n_out == grad_z.shape[1]

        # Store gradients for parameter updates
        self._grad_W_current = np.dot(self._cache_current.T, grad_z)
        self._grad_b_current = grad_z.sum(axis=0)

        # Pass through gradient loss
        return np.dot(grad_z, self._W.T)

    def update_params(self, learning_rate):
        """
        Perform one step of gradient descent with given learning rate on layer parameters using stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        self._W -= learning_rate * self._grad_W_current
        self._b -= learning_rate * self._grad_b_current


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Apply sigmoid function elementwise
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, z):
        a = 1 / (1 + np.exp(-z))
        self._cache_current = a
        return a

    def backward(self, grad_a):
        diff = self._cache_current * (1 - self._cache_current)
        return diff * grad_a


class ReluLayer(Layer):
    """
    ReluLayer: Apply Relu function elementwise
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, z):
        self._cache_current = z
        return np.maximum(0, z)

    # return 0 if the stored input is negative else grad_a
    #   -> x * true = x  ...  x * false = 0
    def backward(self, grad_a):
        return grad_a * (self._cache_current > 0)


class MSELossLayer(Layer):
    """
    MSELossLayer: Compute mean-squared error between y_pred and y_target
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, z, y_target):
        assert len(z) == len(y_target)

        self._cache_current = z, y_target
        return self._mse(z, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Compute softmax followed by negative log-likelihood loss
    """

    def __init__(self):
        self._cache_current = None

    # Subtract z.max to improve numerical stability and reduce exponential growth accumulating in error terms
    # Equivalent to standard softmax verifiable by exponential factorisation
    @staticmethod
    def _softmax(z):
        numer = np.exp(z - z.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, z, y_target):
        assert len(z) == len(y_target)

        y_pred = self._softmax(z)
        self._cache_current = y_pred, y_target

        return -1 / len(y_target) * np.sum(y_target * np.log(y_pred))

    def backward(self):
        y_pred, y_target = self._cache_current
        return -1 / len(y_target) * (y_target - y_pred)
