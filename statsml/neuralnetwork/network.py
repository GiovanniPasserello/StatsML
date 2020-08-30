import functools
import numpy as np
import pickle

from statsml.neuralnetwork.layer import Layer, LinearLayer, ReluLayer, SigmoidLayer


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and activation functions
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension)
            neurons {list} -- Number of neurons in each layer represented as a list (the length of the list determines
            the number of layers)
            activations {list} -- List of activation functions to use for each layer
        """

        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        # Create all pairs of (input, output) layer dimensions
        layer_neurons = [input_dim] + neurons
        dimensions = zip(layer_neurons, layer_neurons[1:])

        # Create all pairings of linear layers w/ activation layers
        layers = []
        for i, (n_in, n_out) in enumerate(dimensions):
            # Add linear layer with corresponding dimensions
            layers.append(LinearLayer(n_in, n_out))
            # Add activation layer (pass through identity layer)
            if activations[i] != "identity":
                layers.append(self._get_activation_layer(activations[i]))

        # Set as polymorphic layer numpy array
        self._layers = np.array(layers, dtype=Layer)

    @staticmethod
    def _get_activation_layer(activation):
        if activation == "sigmoid":
            return SigmoidLayer()
        elif activation == "relu":
            return ReluLayer()
        return None  # Only support sigmoid or relu

    def forward(self, x):
        """
        Perform forward pass through the network

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim)
        Returns:
            {np.ndarray} -- Output array of shape (batch_size, num_neurons_in_final_layer)
        """

        assert x.shape[1] == self._layers[0].n_in

        # Pass the input x through the whole network and return the output of the final layer.
        # This will internally store the inputs at each layer for gradient calculations.
        # We can reduce as a method of iteratively calculating and passing through values.
        # x is the starting value of inp, layer[0] is the starting value of layer.
        return functools.reduce(lambda inp, layer: layer.forward(inp), self._layers, x)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Perform backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1, num_neurons_in_final_layer)
        Returns:
            {np.ndarray} -- Array containing gradient w.r.t. layer input of shape (batch_size, input_dim)
        """

        assert isinstance(self._layers[-1], SigmoidLayer) or \
               isinstance(self._layers[-1], ReluLayer) or \
               grad_z.shape[1] == self._layers[-1].n_out

        # Pass the output grad_z back through the whole network to return the input grad.
        # This will internally store the gradients in each linear layer for parameter updates.
        # Same as above, grad is initially grad_z, layer is initially the final layer (reversed).
        return functools.reduce(lambda grad, layer: layer.backward(grad), reversed(self._layers), grad_z)

    def update_params(self, learning_rate):
        """
        Perform one step of gradient descent with given learning rate on parameters of all layers using stored gradients

        Arguments:
            learning_rate {float} -- Learning rate of update step
        """

        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """

    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """

    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network
