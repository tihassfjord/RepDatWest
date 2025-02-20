import random


class Network():
    def __init__():
        """

        General class

        """
        pass

    def make_batches_rnn(self, N, batch_size):
        """

        Function for splitting a dataset
        into mini-batches.

        N: Total number of samples
        batch_size: Size of each mini-batch

        """
        idx = range(0, N)

        for i in range(0, N, batch_size):
            yield idx[i:i+batch_size]

    def make_batches_cnn(self, N, batch_size):
        """

        Function for splitting a dataset
        into mini-batches.

        N: Total number of samples
        batch_size: Size of each mini-batch

        """
        idx = random.sample(range(0, N), N)

        for i in range(0, N, batch_size):
            yield idx[i:i+batch_size]
