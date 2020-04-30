import numpy as np
from array import array


def softmax(predictions):
    """
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions -
        probability for every class, 0..1
    """

    preds = np.array(predictions.copy())

    if preds.ndim > 1:
        preds -= np.max(preds, axis=1).reshape(-1, 1)
        proba = np.exp(preds) / np.sum(np.exp(preds), axis=1).reshape(-1, 1)
    else:
        preds -= np.max(preds)
        proba = np.exp(preds) / np.sum(np.exp(preds))

    return proba


def cross_entropy_loss(probs, target_index):
    """
    Computes cross-entropy loss

    Arguments:
      probs: np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    """

    if probs.ndim > 1:
        true_probs = probs[np.arange(len(target_index)), target_index.reshape(-1)]
    else:
        true_probs = probs[target_index]

    loss = np.sum(-np.log(true_probs))
    return loss


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """

    loss = np.sum(W * W) * reg_strength
    grad = reg_strength * W * 2

    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    if preds.ndim > 1:
        mask = np.zeros(preds.shape)
        mask[np.arange(len(target_index)), target_index.reshape(-1)] = 1
    else:
        mask = np.zeros(preds.shape)
        mask[target_index] = 1

    probs = softmax(preds.copy())
    loss = cross_entropy_loss(probs, target_index)
    dpreds = probs - mask

    return loss, dpreds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        self.fw = None

    def forward(self, X):
        self.fw = np.where(X > 0, X, 0)
        return self.fw

    def backward(self, d_out):
        d_result = d_out * np.where(self.fw > 0, 1, 0)
        return d_result

    def params(self):
        return {}

    def reset_grad(self):
        pass

class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        self.X = X.copy()
        return self.X.dot(self.W) + self.B

    def backward(self, d_out):
        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.ones(self.X.shape[0]).dot(d_out)
        return d_out.T.dot(self.W.value)

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )
        self.B = Param(np.zeros(out_channels))
        self.X = None
        self.X_pad = None
        self.padding = padding

    def forward(self, X):
        self.X = X
        self.orig_size_X = X.shape
        batch_size, height, width, channels = self.X.shape
        padding = self.padding

        if padding:
            self.X = np.zeros(shape=(batch_size,
                                         height + padding * 2,
                                         width + padding * 2,
                                         channels))

            self.X[:, padding: -padding, padding: -padding, :] = X

        batch_size, height, width, channels = self.X.shape

        out_height = height - self.filter_size + 1
        out_width = width - self.filter_size + 1

        output = np.zeros(shape=(batch_size, out_height, out_width, self.out_channels))

        for x in range(out_height):
            for y in range(out_width):
                focus = self.X[:, x: x + self.filter_size, y: y + self.filter_size, :]
                output[:, x, y, :] = np.dot(focus.reshape(batch_size, -1),
                                            self.W.value.reshape(-1, self.out_channels)) + self.B.value
        return output

    def backward(self, d_out):
        padding = self.padding
        in_batch_size, in_height, in_width, in_channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        grad_out = np.zeros(shape=self.X.shape)
        self.W.grad = np.zeros(shape=self.W.value.shape)
        self.B.grad = 0

        batch_size = in_batch_size

        for x in range(out_height):
            for y in range(out_width):

                focus_x = self.X[:, x: x + self.filter_size, y: y + self.filter_size, :]
                focus_grad = d_out[:, x, y, :]

                self.B.grad += 2 * np.mean(focus_grad, axis=0)
                self.W.grad += np.matmul(focus_x.reshape(batch_size, -1).T,
                                         focus_grad).reshape(self.W.value.shape)

                focus_grad_out = np.matmul(focus_grad, self.W.value.reshape(-1, batch_size).T)\
                                   .reshape(batch_size, self.filter_size, self.filter_size, in_channels)
                grad_out[:, x: x+self.filter_size, y: y+self.filter_size, :] += focus_grad_out

        return grad_out[:, padding: -padding, padding: -padding, :]

    def params(self):
        return { 'W': self.W, 'B': self.B }

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.max_idx = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        out_height = int((height - self.pool_size) / self.stride) + 1
        out_width = int((width - self.pool_size) / self.stride) + 1

        output = np.zeros(shape=(batch_size, out_height, out_width, channels))

        for x in range(0, out_height, self.stride):
            for y in range(0, out_width, self.stride):
                focus = X[:, x: x + self.pool_size, y: y + self.pool_size, :]
                output[:, x, y, :] = np.max(focus, axis=(1, 2))

        return output

    def backward(self, d_out):
        X_col = self.X
        dX_col = np.zeros_like(X_col)
        max_idx = np.argmax(X_col, axis=0)
        dout_flat = dout.transpose(2, 3, 0, 1).ravel()
        dX_col[max_idx, range(max_idx.size)] = dout_flat

        # We now have the stretched matrix of 4x9800, then undo it with col2im operation
        # dX would be 50x1x28x28
        dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)

        # Reshape back to match the input dimension: 5x10x28x28
        dX = dX.reshape(X.shape)
        return dX_col

    def params(self):
        return {}

    def reset_grad(self):
        pass

class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        self.X = X
        batch_size, height, width, channels = X.shape
        return X.reshape((batch_size, -1))

    def backward(self, d_out):
        return d_out.reshape(self.X.shape)

    def params(self):
        # No params!
        return {}
