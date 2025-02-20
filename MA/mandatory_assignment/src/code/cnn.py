#import layers_solution as layers
import layers
import networks
import numpy as np

fc_input = # TODO: INSERT NUMBER HERE

class CNN(networks.Network):
    def __init__(self):
        """
        Implementation of a LeNet-5-like architecture.
        This implementation uses a ReLU activation function and
        max-pooling layers.
        """
        self.filters = [(6, 1, 5, 5), (16, 6, 5, 5)]
        self.fc_layers = [fc_input, 120, 84, 10]
        self.layers = []
        for i in range(len(self.filters)):
            layer = layers.ConvolutionalLayer(self.filters[i])
            self.layers.append(layer)
            layer = layers.MaxPoolingLayer(2,2,2)
            self.layers.append(layer)
            layer = layers.ReluLayer()
            self.layers.append(layer)
        for i in range(len(self.fc_layers[:-1])):
            layer = layers.FullyConnectedLayer(self.fc_layers[i],
                                               self.fc_layers[i+1])
            self.layers.append(layer)
            layer = layers.ReluLayer()
            self.layers.append(layer)
        self.layers.pop(-1)  # Remove relu actiovation function in last layer.
        self.layers.append(layers.SoftmaxLossLayer())
        self.output = layers.SoftmaxLayer()

    def forward_pass(self, x, y=None, TRAIN=False):
        """
        Forward pass of the network. If TRAIN=False
        the network outputs the softmax activations.
        If TRAIN=True the networks outputs the loss and the loss derivatives.
        """
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        if TRAIN:
            loss, loss_derivative = self.layers[-1].forward(x, y)
            return loss, loss_derivative
        else:
            return self.output.forward(x)

    def backward_pass(self, loss):
        """
        Backward pass of the network. Loops
        through all layers of the network, starting from the
        output layer.
        """
        for layer in reversed(self.layers[:-1]):
            loss = layer.backward(loss)

    def train(self, x, y, epochs, batch_size):
        """
        Training function. Creates batches of size
        batch_size and trains for epochs number of epochs.
        """
        idx_list = list(self.make_batches_cnn(x.shape[0], batch_size))
        for epoch in range(epochs):
            loss_av = []
            for idx in idx_list:
                loss, loss_derivative = self.forward_pass(x[idx], y[idx], True)
                self.backward_pass(loss_derivative)
                loss_av.append(loss)
            print("Epoch:", epoch, "Loss:", np.mean(loss_av))

    def test(self, x):
        """
        Testing network.
        """
        return self.forward_pass(x)
