import numpy as np

class Activation_ReLu:
    """
    Rectified Linear Unit activation function.

    This is a commonly used activation function in NNs to decide weather a neuron's output is fowarded
    further.

    - forward(): This Layer's forward pass
    - backward(): This Layer's backward pass

    Example
    ----------
    >>> relu = Activation_ReLU()
    """

    # Forward pass
    def forward(self, inputs: np.ndarray):
        """
        Save ``ReLU(inputs)`` as ``output``.

        Pass ``inputs (np.ndarray)`` through the ReLU so the calculation is applied to each value. The
        results are then saved as ``output (np.ndarray)`` so that other layers and functions can access
        it. The inputs are saved aswell, so the backward pass can access it to calculate the derivative.

        Parameters
        ----------
        inputs : np.ndarray
            An array of values of the predecessor layer or function.

        Examples
        ----------
        >>> relu = Activation_ReLU()
        >>> predecessor.forward(data)
        >>> relu.forward(predecessor.output)
        >>> successor.forward(relu.output)
        """
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        """
        Save the derivate of ``ReLU(inputs)`` as ``dinputs``.
        
        Since the inputs are saved while forwarding through the ReLU, the backward pass can access
        them and with that calculate the derivative.

        Parameters
        ----------
        dvalues : np.ndarray
            An array of derivate values of the successor layer or function.

        Example
        ----------
        >>> relu = Activation_ReLU()
        >>> successor.backward(data)
        >>> relu.backward(successor.dinputs)
        >>> predecessor.backward(relu.dinputs)
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:
    """
    Softmax / Softargmax / Normalized Exponential activation function.

    This is a commonly used activation function in NNs to transform inputs into an vector with
    components in (0, 1), while their sum is 1. With that each component represents a probability
    and Softmax is therefore often used as an output for single-lable classification tasks,
    especially when combined with CCE (Categorical Cross Entropy).

    Methods:
    ----------
    - forward(): This Layer's forward pass
    - backward(): This Layer's backward pass

    Example
    ----------
    >>> softmax = Activation_Softmax()
    """
    
    # Forward pass
    def forward(self, inputs: np.ndarray):
        """
        Save ``Softmax(inputs)`` as ``output``.

        Pass ``inputs (np.ndarray)`` through Softmax so the calculation formula is applied to each value.
        The results are then saved as ``output (np.ndarray)`` so that other layers and functions can
        access it. The inputs are saved aswell, so the backward pass can access it to calculate the
        derivative.

        Parameters
        ----------
        inputs : np.ndarray
            An array of logits of the predecessor layer or function.

        Examples
        ----------
        >>> softmax = Activation_Softmax()
        >>> predecessor.forward(data)
        >>> softmax.forward(predecessor.output)
        >>> successor.forward(softmax.output)
        """
        self.inputs = inputs
        # Exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        """
        Save the derivate of ``Softmax(inputs)`` as ``dinputs``.
        
        Since the inputs are saved while forwarding through Softmax, the backward pass can access
        them and with that calculate the derivative.

        Parameters
        ----------
        dvalues : np.ndarray
            An array of derivate values of the successor layer or function.

        Example
        ----------
        >>> softmax = Activation_Softmax()
        >>> successor.backward(data)
        >>> softmax.backward(successor.dinputs)
        >>> predecessor.backward(softmax.dinputs)
        """
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
    """
    Sigmoid activation function.

    This is a commonly used activation function in NNs to map an input value to a range between 0 and 1.
    It is frequently used to tirn raw model outputs into clear probabilities.

    Methods:
    ----------
    - forward(): This Layer's forward pass
    - backward(): This Layer's backward pass

    Example
    ----------
    >>> sigmoid = Activation_Sigmoid()
    """
    # Forward pass
    def forward(self, inputs: np.ndarray):
        """
        Save ``Sigmoid(inputs)`` as ``output``.

        Pass ``inputs (np.ndarray)`` through Sigmoid so the calculation formula is applied to each value.
        The results are then saved as ``output (np.ndarray)`` so that other layers and functions can
        access it. The inputs are saved aswell, so the backward pass can access it to calculate the
        derivative.

        Parameters
        ----------
        inputs : np.ndarray
            An array of values of the predecessor layer or function.

        Examples
        ----------
        >>> sigmoid = Activation_Sigmoid()
        >>> predecessor.forward(data)
        >>> sigmoid.forward(predecessor.output)
        >>> successor.forward(sigmoid.output)
        """
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues: np.ndarray):
        """
        Save the derivate of ``Sigmoid(inputs)`` as ``dinputs``.
        
        Since the inputs are saved while forwarding through Sigmoid, the backward pass can access
        them and with that calculate the derivative.

        Parameters
        ----------
        dvalues : np.ndarray
            An array of derivate values of the successor layer or function.

        Example
        ----------
        >>> sigmoid = Activation_Sigmoid()
        >>> successor.backward(data)
        >>> sigmoid.backward(successor.dinputs)
        >>> predecessor.backward(sigmoid.dinputs)
        """
        self.dinputs = dvalues * (1 - self.output) * self.output