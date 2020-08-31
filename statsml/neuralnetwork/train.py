import numpy as np

from statsml.neuralnetwork.layer import MSELossLayer, CrossEntropyLossLayer


class Trainer(object):
    """
    Trainer: Object that manages the training of MultiLayerNetwork
    """

    def __init__(self, network, batch_size, nb_epoch, learning_rate, loss_fun, shuffle_flag):
        """
        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained
            batch_size {int} -- Training batch size
            nb_epoch {int} -- Number of training epochs
            learning_rate {float} -- SGD learning rate to be used in training
            loss_fun {str} -- Loss function to be used -> possible values: mse, cross_entropy
            shuffle_flag {bool} -- If True, training data is shuffled before training
        """

        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        assert loss_fun in ["mse", "cross_entropy"], \
            "loss_fun can only take values \"mse\" (mean-squared error) or \"cross_entropy\""

        self._loss_layer = MSELossLayer() if loss_fun == "mse" else CrossEntropyLossLayer()

    @staticmethod
    def _shuffle(input_dataset, target_dataset):
        """
        Return shuffled versions of the inputs

        Arguments:
            input_dataset {np.ndarray} -- Array of input features, of shape (num_data_points, n_features)
            target_dataset {np.ndarray} -- Array of corresponding targets, of shape (num_data_points, )
        Returns:
            2-tuple of np.ndarray: (shuffled inputs, shuffled_targets)
        """

        assert len(input_dataset) == len(target_dataset)

        permutation = np.random.permutation(len(input_dataset))
        return input_dataset[permutation], target_dataset[permutation]

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffle the input data (if `shuffle` is True)
            - Split the dataset into batches of size `batch_size`
            - For each batch:
                - Perform forward pass through the network given the current batch of inputs
                - Compute loss
                - Perform backward pass to compute gradients of loss with respect to parameters of network
                - Perform one step of gradient descent on the network parameters

        Arguments:
            input_dataset {np.ndarray} -- Array of input features, of shape (num_training_data_points, n_features)
            target_dataset {np.ndarray} -- Array of corresponding targets of shape (num_training_data_points, )
        """

        assert len(input_dataset) == len(target_dataset)

        for _ in range(self.nb_epoch):
            # Shuffle uniquely for every epoch
            if self.shuffle_flag:
                input_dataset, target_dataset = self._shuffle(input_dataset, target_dataset)

            input_batches, target_batches = self.create_batches(input_dataset, target_dataset)

            for i in range(len(input_batches)):
                input_batch = input_batches[i]
                target_batch = target_batches[i]

                prediction = self.network.forward(input_batch)  # Pass forward through the network
                self._loss_layer.forward(prediction, target_batch)  # Pass forward through the loss layer
                grad_loss = self._loss_layer.backward()  # Calculate the gradient of the loss
                self.network.backward(grad_loss)  # Perform back propagation using the loss gradient
                self.network.update_params(self.learning_rate)  # Update the network with the stored gradients

    # Create batch splits, taking into account non-even splits with larger batches.
    # We are not guaranteed that the dataset is evenly divisible by batch_size, if we were, just use np.split().
    # The method here is to even distribute the excess samples, as we want to maintain a fairly constant batch size,
    # rather than having a single smaller batch which would affect sample gradient weighting.
    def create_batches(self, input_dataset, target_dataset):
        input_batches, target_batches = [], []

        # Get the number of full batches and any remaining overflow
        num_batches, num_larger = divmod(len(input_dataset), self.batch_size)

        # Create batches with the correct size accounting for overflow
        covered_range = 0
        for i in range(num_batches):
            # For the first num_larger iterations, take an extra element
            span = self.batch_size + 1 if i < num_larger else self.batch_size
            # Take a batch index by the span over the dataset
            input_batches.append(input_dataset[covered_range:covered_range + span])
            target_batches.append(target_dataset[covered_range:covered_range + span])
            # Move on to guarantee sampling without replacement
            covered_range += span

        return np.array(input_batches), np.array(target_batches)

    def eval_loss(self, input_dataset, target_dataset):
        """
        Evaluate the loss function for given data
        Pass input data forward through the network and then generate predictions through the loss layer

        Arguments:
            input_dataset {np.ndarray} -- Array of input features of shape (num_evaluation_data_points, n_features)
            target_dataset {np.ndarray} -- Array of corresponding targets of shape (#_evaluation_data_points, )
        """

        predictions = self.network.forward(input_dataset)
        return self._loss_layer.forward(predictions, target_dataset)
