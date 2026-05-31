import nnfs
from nnfs import spiral_data
import numpy as np

# Genral settings
nnfs.init()

class Layer_Dense:

    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLu:
    
    def forward(self, inputs):
        # f(x) = x if x > 0, 0 if x <= 0
        self.output = np.maximum(0, inputs)

class Activation_Softmax:
    
    def forward(self, inputs):
        # Exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Create dataset
X, y = spiral_data(samples=100, classes=3)