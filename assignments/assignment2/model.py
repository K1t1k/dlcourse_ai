import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """

        self.reg = reg
        self.layers = [FullyConnectedLayer(n_input, hidden_layer_size),
                       ReLULayer(),
                       FullyConnectedLayer(hidden_layer_size, n_output)]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """

        for layer in self.layers:
            layer.reset_grad()

        input = X.copy()
        for layer in self.layers:
            input = layer.forward(input)

        loss, grad = softmax_with_cross_entropy(input, y)

        output = grad.copy()
        for layer in reversed(self.layers):
            output = layer.backward(output)

        if self.reg:
            for num, layer in enumerate(self.layers):
                if isinstance(layer, FullyConnectedLayer):
                    l2_loss, d_reg = l2_regularization(layer.W.value, self.reg)
                    loss += l2_loss
                    layer.W.grad += d_reg

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

        input = X.copy()
        for layer in self.layers:
            input = layer.forward(input)

        probs = softmax(input)
        preds = np.argmax(probs, axis=1)

        return preds

    def params(self):
        result = {}

        for num_layer, layer in enumerate(self.layers):
            for key, value in layer.params().items():
                result[(num_layer, key)] = value

        return result
