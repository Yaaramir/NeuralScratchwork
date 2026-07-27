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

# Convolutional Layer for 2-dimensional multi-channel inputs
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

        """
        Initializes a Convolutional Layer for CNN. Creates kernels and feature maps.
        
        Args:
            n_input_channels (int): Number of input channels
            kernel_size (int | (int, int)): Size h**2 or (h * w) of the kernels
            n_output_channels (int): Number of output channels (number of kernels)
            stride (int): Size of steps a filter moves on a matrix after completing one calculation. Default is 1.
            padding (int): Size of augmented picture frame for kernels to move over. Default is 0.
        """


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

    def forward(self, input_matrix: np.ndarray):

        """
        Calculates a feature map ready for pooling for each kernel.

        Args:
            input_matrix (np.ndarray): A 2D matrix with a number of channels
        """

        # Get input matrix properties
        if input_matrix.ndim == 2:
            input_matrix = np.expand_dims(input_matrix, axis=1)
        h_in, w_in, c_in = input_matrix.shape

        # Make sure input matrix is large enough / kernel is small enough
        assert (h_in + 2 * self.padding >= self.h_kern) and (w_in + 2 * self.padding >= self.w_kern), "ERROR: Input matrix is too small for kernel to move over."

        # Create feature maps as output
        h_out = (h_in - self.h_kern + 2 * self.padding) // self.stride + 1
        w_out = (w_in - self.w_kern + 2 * self.padding) // self.stride + 1
        self.output = np.zeros((h_out, w_out, self.c_in, self.c_out))

        # For each output channel / feature map
        for c_out in range(self.c_out):

            for h in range(h_out):
                for w in range(w_out):

                    # Get patch for kernel calculations from input_matrix
                    h_start = h * self.stride
                    h_end = h_start + self.h_kern
                    w_start = w * self.stride
                    w_end = w_start + self.w_kern
                    patch = input_matrix[h_start:h_end, w_start:w_end]

                    # Get the right kernel for this output channel / feature map
                    kernel = self.weights[:, :, :, c_out]

                    # Calculation
                    self.output[h, w, c_out] = np.sum(patch * kernel) + self.biases[0, c_out]

    def backward(self, dvalues: np.ndarray):

      dweights = np.zeros_like(self.weights)
      dbiases = np.zeros_like(self.biases)
      dinputs = np.zeros_like(self.input)

      h_out, w_out, _, c_out = self.output.shape

      for c_o in range(c_out):
        dbiases[0, c_o] = np.sum(dvalues[:, :, c_o])

        for h in range(h_out):
          for w in range(w_out):
            h_start = h * self.stride
            h_end = h_start + self.h_kern
            w_start = w * self.stride
            w_end = w_start + self.w_kern

            patch = self.input[h_start:h_end, w_start:w_end, :]

            dweights[:, :, :, c_o] += patch * dvalues[h, w, c_o]

            kernel = self.weights[:, :, :, c_o]
            dinputs[h_start:h_end, w_start:w_end, :] += (
                kernel * dvalues[h, w, c_o]
            )

      self.dweights = dweights
      self.dbiases = dbiases
      self.dinputs = dinputs

      return self.dinputs

class Layer_Pooling:
    
    def __init__(self, kernel_size: int | tuple[int, int], stride: int):

        # Get kernel properies
        if isinstance(kernel_size, int):
            self.h_kern = self.w_kern = kernel_size
        else:
            self.h_kern, self.w_kern = kernel_size

        self.stride = stride

    def pool(self, feature_maps: np.ndarray):

        # Create pooled feature maps
        h_in, w_in, self.c = feature_maps.shape
        self.h_out = (h_in - self.h_kern) // self.stride + 1
        self.w_out = (w_in - self.w_kern) // self.stride + 1
        self.output = np.zeros((self.h_out, self.w_out, self.c))
        self.forward(feature_maps)

# Average Pooling Layer
class Layer_Pooling_Average(Layer_Pooling):

    def forward(self, feature_maps: np.ndarray):

        for c in range(self.c):
            for h in range(self.h_out):
                for w in range(self.w_out):

                    # Get patch for kernel calculations from feature_maps
                    h_start = h * self.stride
                    h_end = h_start + self.h_kern
                    w_start = w * self.stride
                    w_end = w_start + self.w_kern
                    patch = feature_maps[h_start:h_end, w_start:w_end, c]

                    self.output[h, w, c] = np.mean(patch)

    def backward(self, dvalues: np.ndarray):
        h_in = (self.h_out - 1) * self.stride + self.h_kern
        w_in = (self.w_out - 1) * self.stride + self.w_kern
        dvalues_prev = np.zeros((h_in, w_in, self.c))

        patch = self.h_kern * self.w_kern

        for c in range(self.c):
            for h in range(self.h_out):
                for w in range(self.w_out):
                    h_start = h * self.stride
                    h_end = h_start + self.h_kern
                    w_start = w * self.stride
                    w_end = w_start + self.w_kern

                    dist_grad = dvalues[h, w, c] / patch
                    dvalues_prev[h_start:h_end, w_start:w_end, c] += dist_grad

        self.dinputs = dvalues_prev

# Maximum Pooling Layer
class Layer_Pooling_Maximum(Layer_Pooling):

    def forward(self, feature_maps: np.ndarray):

        for c in range(self.c):
            for h in range(self.h_out):
                for w in range(self.w_out):

                    # Get patch for kernel calculations from feature_maps
                    h_start = h * self.stride
                    h_end = h_start + self.h_kern
                    w_start = w * self.stride
                    w_end = w_start + self.w_kern
                    patch = feature_maps[h_start:h_end, w_start:w_end, c]

                    self.output[h, w, c] = np.max(patch)

    def backward(self, dvalues: np.ndarray):
        h_in = (self.h_out - 1) * self.stride + self.h_kern
        w_in = (self.w_out - 1) * self.stride + self.w_kern
        dvalues_prev = np.zeros((h_in, w_in, self.c))

        for c in range(self.c):
            for h in range(self.h_out):
                for w in range(self.w_out):
                    h_start = h * self.stride
                    h_end = h_start + self.h_kern
                    w_start = w * self.stride
                    w_end = w_start + self.w_kern

                    patch = self.input[h_start:h_end, w_start:w_end, c]
                    mask = patch == np.max(patch)

                    dvalues_prev[h_start:h_end, w_start:w_end, c] += (mask * dvalues[h, w, c])

        self.dinputs = dvalues_prev

# Flattens matrices to 1D tensor
class Layer_Flattener():

    # Save input_shape and reshape to 1D tensor
    def forward(self, input):
        self.input_shape = input.shape
        self.output = self.input.flatten()

    # Reshape data to input_shape saved while forwarding
    def backward(self, dvalues):
        self.dvalues = dvalues.reshape(self.input_shape)

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