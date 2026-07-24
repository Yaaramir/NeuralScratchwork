from activation_functions import Activation_Softmax
import numpy as np

class Loss:

    # data loss without L1, L2 regularization
    def data_loss(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

    # L1 and L2 regularization loss
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

# Categorical Cross Entropy to calculate on probability predictions. Integrated Softmax!
class Loss_CategoricalCrossEntropy(Loss):

    def __init__(self):
        self.softmax = Activation_Softmax

    def forward(self, inputs, y):
        self.softmax.forward(inputs)
        self.output = self.softmax.output
        return self.calculate(self.output, y)

    def backward(self, dvalues, y):
        n_samples = len(dvalues)

        # For one hot encoded labels
        if len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), y] -= 1
        self.dinputs = self.dinputs / n_samples