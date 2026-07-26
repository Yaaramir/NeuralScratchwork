import numpy as np

# Dense / Fully-Connected Layer
class Layer_Dense:

    # Initialization
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0,
                 bias_regularizer_l1=0,
                 weight_regularizer_l2=0,
                 bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # L1 regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        # L2 regularization strength
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues):

        # Gradient on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradients on regularization
        # L1 on weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
        # L1 on biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1
        # L2 on weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        # L2 on biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

class Layer_Convolutional_2D:

    # Initialization
    def __init__(self, input_channels, input_size, kernel_size, output_channels, stride=1, padding=0):

        # Set kernel size from either int or tuple
        if isinstance(kernel_size, int):
            m_kern = n_kern = kernel_size
        else:
            m_kern, n_kern = kernel_size

        # Create kernels with wieghts and biases
        self.weights = 0.001 * np.random.randn(m_kern, n_kern, input_channels, output_channels)
        self.biases = np.zeros(1, output_channels)

        self.padding = padding
        self.stride = stride

        # Create feature maps
        if isinstance(input_size, int):
            m_in = n_in = input_size
        else:
            m_in, n_in = input_size
        m_out = (m_in - m_kern + 2 * padding) / stride + 1
        n_out = (n_in - m_kern + 2 * padding) / stride + 1
        self.feature_maps = np.zeros(m_out, n_out, output_channels)

# Dropout Layer
class Layer_Dropout:
    def __init__(self, dropout_rate):
        self.success_rate = 1 - dropout_rate

    # Carry out dropout and balance out the rate by multiplie successful outputs accordingly to rate
    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = np.random.binomial(n=1, p=self.success_rate, size=inputs.shape) / self.success_rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask