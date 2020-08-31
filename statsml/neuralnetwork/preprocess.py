class Preprocessor(object):
    """
    Preprocessor: Apply simple preprocessing operations to real-valued numerical datasets (min-max normalization)
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset

        Arguments:
            data {np.ndarray} --
                dataset used to determined the parameters for the normalization
                An N x K dimensional dataset (N samples, K features)
        """

        self._num_features = data.shape[1]  # Number of features per sample
        self._data_mins = data.min(axis=0)  # Minimum value per feature
        self._data_ranges = data.ptp(axis=0)  # Range per feature (peak-to-peak)

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset (min-max normalization over the range [0, 1])

        Arguments:
            data {np.ndarray} -- dataset to be normalized
        Returns:
            {np.ndarray} normalized dataset
        """

        assert data.shape[1] == self._num_features

        return (data - self._data_mins) / self._data_ranges

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve original dataset (min-max normalization over the range [0, 1])

        Arguments:
            data {np.ndarray} -- dataset for which to revert normalization
        Returns:
            {np.ndarray} reverted dataset
        """

        assert data.shape[1] == self._num_features

        return (data * self._data_ranges) + self._data_mins
