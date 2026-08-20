import numpy as np

# Dense Layer
class Layer_Dense:
    """
    Dense Layer or Fully Connected Layer

    Every neuron is connected to each neuron of the previous layer.

    Parameters
    ----------
    ``input_channels`` : int
        The number of input channels from data or the previous layer.
    ``n_neurons`` : int
        The number of neurons in this layer.
    ``weight_regularizer_l1`` : float
        L1 regularization value for weights.
    ``bias_regularizer_l1`` : float
        L1 regularization value for biases.
    ``weight_regularizer_l2`` : float
            L2 regularization value for weights.
    ``bias_regularizer_l2`` : float
        L2 regularization value for biases.

    Methods
    ----------
    - ``forward()``: This layer's forward pass
    - ``backward()``: This layer's backward pass

    Examples
    ----------
    >>> dense_layer = Layer_Dense(input_channels=4,
    >>>                           n_neurons=3,
    >>>                           weight_regularizer_l2=0.02,
    >>>                           bias_regularizer_l2=0.002)
    """

    # Initialization
    def __init__(self, input_channels: int,
                 n_neurons: int,
                 weight_regularizer_l1: float = 0,
                 bias_regularizer_l1: float = 0,
                 weight_regularizer_l2: float = 0,
                 bias_regularizer_l2: float = 0):

        """
        Parameters
        ----------
        ``input_channels`` : int
            The number of input channels from data or the previous layer.
        ``n_neurons`` : int
            The number of neurons in this layer.
        ``weight_regularizer_l1`` : float
            L1 regularization value for weights.
        ``bias_regularizer_l1`` : float
            L1 regularization value for biases.
        ``weight_regularizer_l2`` : float
                L2 regularization value for weights.
        ``bias_regularizer_l2`` : float
            L2 regularization value for biases.
        """
        
        self.weights = 0.01 * np.random.randn(input_channels, n_neurons)
        self.biases = np.zeros((1, n_neurons))

        # L1 regularization strength
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        # L2 regularization strength
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    # Forward pass
    def forward(self, inputs: np.ndarray):
        """
        Pass the data forward and save results as output.

        Inputs are multiplied with weights and added by biases. The results are saved in ``output``
        so that other layers and functions can access it. Inputs are saved as well, so the backward
        pass can access it to calculate the derivative.

        Parameters
        ----------
        ``inputs`` : np.ndarray
            An array of values of the previous layer or function.

        Examples
        ----------
        >>> # In a forward pass our dense layer forwards data
        >>> predecessor.forward(data)
        >>> dense_layer.forward(predecessor.output)
        >>> successor.forward(dense_layer.output)
        """
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        """
        Save the derivates as ``dinputs``.
        
        Since the inputs are saved while forwarding through the activation function, the backward pass can access
        them and with that calculate the derivative.

        Parameters
        ----------
        ``dvalues`` : np.ndarray
            An array of derivate values of the successor layer or function.

        Example
        ----------
        >>> # In a backward pass our dense layer backwards data
        >>> successor.backward(data)
        >>> dense_layer.backward(successor.dinputs)
        >>> predecessor.backward(dense_layer.dinputs)
        """

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

# Dropout Layer
class Layer_Dropout:
    """
    A regularization tool to prevend the network to memorize data.

    A Dropout Layer turns off a specific rate of neurons in each training step. The network will have to
    learn making decisions with only a fracture of all available data. This way memorizing data and
    coadaption can be avoided.

    Parameters
    ----------
    ``dropout_rate`` : (int | float)
        The rate of neurons to be turned off with ``(0 <= dropout_rate < 1)``.

    Methods
    ----------
    - ``forward()``: This layer's forward pass
    - ``backward()``: This layer's backward pass

    Example
    ----------
    >>> dropout_layer = Layer_Dropout(dropout_rate=0.3)
    """
    def __init__(self, dropout_rate: int | float):
        """
        Parameters
        ----------
        ``dropout_rate`` : int | float
            The rate of neurons to be turned off with ``(0 <= dropout_rate < 1)``
        """
        # Check for valid parameters
        if not isinstance(dropout_rate, (int, float)):
            raise TypeError("Dropout rate must be a number of type int or float!")
        if not (0.0 <= dropout_rate < 1.0):
            raise ValueError("Dropout rate must be between 0 (inclusive) and 1 (exclusive)!")

        self.dropout_rate = dropout_rate
        self.success_rate = 1 - dropout_rate

    # Carry out dropout and balance out the rate by multiplie successful outputs accordingly to rate
    def forward(self, inputs: np.ndarray):
        """
        Forward data through Dropout Layer and save results as ``output``.

        Pass ``inputs`` through the Dropout Layer that the specified rate of neurons will
        be turned off. The active neuron's results are then saved as ``output`` so that other layers and
        functions can access it. The ``inputs`` are saved as well, so the backward pass can access it to
        calculate the derivative.

        Parameters
        ----------
        ``inputs`` : np.ndarray
            An array of values of the predecessor layer or function.

        Examples
        ----------
        >>> # In a forward pass our Dropout Layer forwards data.
        >>> predecessor.forward(data)
        >>> dropout_layer.forward(predecessor.output)
        >>> successor.forward(dropout_layer.output)
        """
        self.inputs = inputs
        self.binary_mask = np.random.binomial(n=1, p=self.success_rate, size=(inputs.shape)) / self.success_rate
        self.output = inputs * self.binary_mask

    def backward(self, dvalues: np.ndarray):
        """
        Save the derivates as ``dinputs``.
        
        Since the inputs are saved while forwarding through the activation function, the backward pass can access
        them and with that calculate the derivative.

        Parameters
        ----------
        ``dvalues`` : np.ndarray
            An array of derivate values of the successor layer or function.

        Example
        ----------
        >>> # In a backward pass our Dropout Layer backwards data
        >>> successor.backward(data)
        >>> dropout_layer.backward(successor.dinputs)
        >>> predecessor.backward(dropout_layer.dinputs)
        """
        self.dinputs = dvalues * self.binary_mask