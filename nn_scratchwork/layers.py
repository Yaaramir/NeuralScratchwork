import numpy as np

# Dense / Fully-Connected Layer
class Layer_Dense:

    # Initialization
    def __init__(self, input_channels, n_neurons,
                 weight_regularizer_l1=0,
                 bias_regularizer_l1=0,
                 weight_regularizer_l2=0,
                 bias_regularizer_l2=0):
        self.weights = 0.01 * np.random.randn(input_channels, n_neurons)
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
    def __init__(self,
                 # Number of input channels
                 c_in: int,
                 # Kernel / filter size
                 kernel_size: int | tuple[int, int],
                 # Number of kernels / filters
                 c_out: int,
                 # Kernal behaviour
                 stride: int = 1, padding: int = 0):

        """Initializes a Convolutional Layer for CNN. Creates kernels and feature maps.
        
        Args:
            n_input_channels (int): Number of input channels
            kernel_size (int | (int, int)): Size h**2 or (h * w) of the kernels
            n_output_channels (int): Number of output channels (number of kernels)
            stride (int): Size of steps a filter moves on a matrix after completing one calculation. Default is 1.
            padding (int): Size of augmented picture frame for kernels to move over. Default is 0."""


        self.c_out = c_out

        # Set kernel size from either int or tuple
        if isinstance(kernel_size, int):
            self.h_kern = self.w_kern = kernel_size
        else:
            self.h_kern, self.w_kern = kernel_size

        # Create kernels with weights and biases
        self.weights = 0.001 * np.random.randn(self.h_kern, self.w_kern, c_in, c_out)
        self.biases = np.zeros((1, c_out))

        self.padding = padding
        self.stride = stride

    def forward(self, input_matrix):

        # Get input matrix' properties
        h_in, w_in, c_in = input_matrix.shape

        # Make sure input matrix is large enough / kernel is small enough
        assert (h_in + 2 * self.padding >= self.h_kern) and (w_in + 2 * self.padding >= self.w_kern), "ERROR: Input matrix is too small for kernel to move over."

        # Create feature maps as output
        h_out = (h_in - self.h_kern + 2 * self.padding) / self.stride + 1
        w_out = (w_in - self.w_kern + 2 * self.padding) / self.stride + 1
        self.feature_maps = np.zeros((h_out, w_out, self.c_out))

        # For feature map
        for feature_map in range(self.c_out):

            # For every feature
            for i in range(h_out):
                for j in range(w_out):

                    # Get patch for kernel calculations
                    i_start = i * self.stride
                    i_end = i_start + self.w_kern
                    j_start = j * self.stride
                    j_end = j_start + self.w_kern

                    patch = input_matrix[i_start:i_end, j_start:j_end]

                    # Get the right kernel for this feature map
                    kernel = self.weights[:, :, :, feature_map]

                    # Calculation
                    self.feature_maps[i, j, feature_map] = np.sum(patch * kernel) + self.biases[0, feature_map]


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