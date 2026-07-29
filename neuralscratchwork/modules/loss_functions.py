from .activation_functions import Activation_Softmax
import numpy as np

# General loss class
class Loss:

    # Data loss
    def data_loss(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

    # Regularization Loss (L1, L2)
    def regularization_loss(self, layer):
    
            # default = 0
            regularization_loss = 0
    
            # L1 weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            # L1 biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            # L2 weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
            # L2 biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)
    
            return regularization_loss

# Binary Cross Entropy
class Loss_BinaryCrossEntropy(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=1)

        return sample_losses

    def backward(self, dvalues: np.ndarray, y_true: np.ndarray):
        n_samples = len(dvalues)
        n_outputs = len(dvalues[0])

        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        self.dinputs = -(y_true / dvalues_clipped - (1 - y_true) / (1 - dvalues_clipped)) / n_outputs
        self.dinputs = self.dinputs / n_samples
         
# Categorical Cross Entropy
class Loss_CategoricalCrossEntropy(Loss):

    def __init__(self):
        self.softmax = Activation_Softmax()

    def forward(self, inputs, y_true):
        self.softmax.forward(inputs)
        self.predictions = self.softmax.output

        n_samples = len(self.predictions)
        # Clipping to avoid log(0) (undefined)
        y_pred_clipped = np.clip(self.predictions, 1e-7, 1 - 1e-7)

        # For categorical labels
        if len(y_true.shape) == 1:
             correct_confidences = y_pred_clipped[range(n_samples), y_true]
        # For one-hot encoded labels
        elif len(y_true.shape) == 2:
             correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        sample_losses = -np.log(correct_confidences)
        data_loss = np.mean(sample_losses)
        return data_loss

    def backward(self, dvalues, y_true):
        n_samples = len(dvalues)
        # For hot-one encoded lables only
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs = self.dinputs / n_samples