import numpy as np


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
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """

        d_result = d_out * np.where(self.fw > 0, 1, 0)
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
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

        return self.X.dot(self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        self.W.grad = self.X.T.dot(d_out)
        self.B.grad = np.ones(shape=(1, self.X.shape[0])).dot(d_out)

        return d_out.dot(self.W.value.T)

    def params(self):
        return {'W': self.W, 'B': self.B}

    def reset_grad(self):
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
