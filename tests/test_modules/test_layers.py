import numpy as np
import pytest
import re

from neuralscratchwork.modules.layers import *

# Layer_Dense
def test_layer_dense_init_dimensions():
    """Checks if initializing produces correct dimensions."""

    # Arrange
    input_channels = 12
    n_neurons = 23

    # Act
    layer = Layer_Dense(input_channels, n_neurons, 0, 0, 0, 0)

    # Assert
    expected_weights_dimensions = (input_channels, n_neurons)
    expected_biases_dimensions = (1, n_neurons)

    assert layer.weights.shape == expected_weights_dimensions, \
        "Dense Layer weights have incorrect dimensions!"
    assert layer.biases.shape == expected_biases_dimensions, \
        "Dense Layer biases have incorrect dimensions!"

def test_layer_dense_init_regularizer_assignments():
    """Checks if regularization values are assigned correctly"""

    # Arrange
    input_channels = 12
    n_neurons = 23
    weight_regularizer_l1 = 0.1
    bias_regularizer_l1 = 0.2
    weight_regularizer_l2 = 0.3
    bias_regularizer_l2 = 0.4

    # Act
    layer = Layer_Dense(input_channels, n_neurons,
                        weight_regularizer_l1, bias_regularizer_l1,
                        weight_regularizer_l2, bias_regularizer_l2)

    # Assert
    assert layer.weight_regularizer_l1 == weight_regularizer_l1, \
        "Dense Layer weight regularizer L1 is not correctly assigned!"
    assert layer.bias_regularizer_l1 == bias_regularizer_l1, \
        "Dense Layer bias regularizer L1 is not correctly assigned!"
    assert layer.weight_regularizer_l2 == weight_regularizer_l2, \
        "Dense Layer weight regularizer L2 is not correctly assigned!"
    assert layer.bias_regularizer_l2 == bias_regularizer_l2, \
        "Dense Layer bias regularizer L2 is not correctly assigned!"

def test_layer_dense_init_regularizer_partly_assignment():
    """Checks is partly assignment of regularizers works correctly."""
    # Arrange
    input_channels = 12
    n_neurons = 23
    bias_regularizer_l1 = 0.2
    weight_regularizer_l2 = 0.3

    # Act
    layer = Layer_Dense(input_channels, n_neurons,
                        bias_regularizer_l1=bias_regularizer_l1,
                        weight_regularizer_l2=weight_regularizer_l2)

    # Assert
    assert layer.weight_regularizer_l1 == 0, \
        "Dense Layer weight regularizer L1 is not correctly assigned!"
    assert layer.bias_regularizer_l1 == bias_regularizer_l1, \
        "Dense Layer bias regularizer L1 is not correctly assigned!"
    assert layer.weight_regularizer_l2 == weight_regularizer_l2, \
        "Dense Layer weight regularizer L2 is not correctly assigned!"
    assert layer.bias_regularizer_l2 == 0, \
        "Dense Layer bias regularizer L2 is not correctly assigned!"

def test_layer_dense_init_random_weights_not_all_zero():
    """Checks if all randomly initialized weights are not zero"""

    # Arrange
    input_channels = 4
    n_neurons = 2

    # Act
    layer = Layer_Dense(input_channels, n_neurons)

    # Assert
    assert not np.all(layer.weights == 0), \
        "Dense Layer weights should not be all zeros!"

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dense_forward_dimensions(batch_size):
    """Checks if forwarding produces correct dimensions."""

    # Arrange
    input_channels = 3
    n_neurons = 5
    layer = Layer_Dense(input_channels, n_neurons)
    inputs = np.random.randn(batch_size, input_channels)

    # Act
    layer.forward(inputs)

    # Assert
    expected_dimensions = (batch_size, n_neurons)

    assert (layer.output.shape == expected_dimensions), \
        "Dense Layer forwarding produces incorrect output dimensions!"

def test_layer_dense_forward_calculation():
    """Checks if forwarding produces correct results."""

    # Arrange
    input_channel = 2
    n_neurons = 3
    layer = Layer_Dense(input_channel, n_neurons)
    layer.weights = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    layer.biases = np.array([10.0, 20.0, 30.0])
    inputs = np.array([[1.0, 0.5], [-1.0, 2.0]])

    # Act
    layer.forward(inputs)

    # Assert
    expected_output = np.array([[13.0, 24.5, 36.0], [17.0, 28.0, 39.0]])
    assert layer.output == pytest.approx(expected_output), \
        "Dense Layer forwarding produces incorrect output results!"

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dense_backward_dimensions(batch_size):
    """Checks if backwarding produces correct dimensions."""
    # Arrange
    input_channels = 3
    n_neurons = 5
    layer = Layer_Dense(input_channels, n_neurons)
    layer.inputs = np.zeros((batch_size, input_channels))
    dvalues = np.zeros((batch_size, n_neurons))

    # Act
    layer.backward(dvalues)

    # Assert
    expected_dweights_shape = (input_channels, n_neurons)
    expected_dbiases_shape = (1, n_neurons)
    expected_dinputs_shape = (batch_size, input_channels)

    assert layer.dweights.shape == expected_dweights_shape, \
        "Dense Layer backwarding produces incorrect dweights dimensions!"
    assert layer.dbiases.shape == expected_dbiases_shape, \
        "Dense Layer backwarding produces incorrect dbiases dimensions!"
    assert layer.dinputs.shape == expected_dinputs_shape, \
        "Dense Layer backwarding produces incorrect dinputs dimensions!"

def test_layer_dense_backward_calculation():
    """Checks if backwarding produces correct results."""

    # Arrange
    input_channels = 2
    n_neurons = 2
    layer = Layer_Dense(input_channels, n_neurons, 0, 0, 0, 0)
    layer.weights = np.array([[1.0, 2.0], [3.0, 4.0]])
    layer.inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    dvalues = np.array([[0.5, -1.0], [1.0, 2.0]])

    # Act
    layer.backward(dvalues)

    # Assert
    expected_dweights = np.array([[3.5, 5.0], [5.0, 6.0]])
    expected_dbiases = np.array([[1.5, 1.0]])
    expected_dinputs = np.array([[-1.5, -2.5], [5.0, 11.0]])

    assert layer.dweights == pytest.approx(expected_dweights), \
        "Layer backwarding produces incorrect dweights results!"
    assert layer.dbiases == pytest.approx(expected_dbiases), \
        "Layer backwarding produces incorrect dbiases results!"
    assert layer.dinputs == pytest.approx(expected_dinputs), \
        "Layer backwarding produces incorrect dinputs results!"

# Layer_Dropout
def test_layer_dropout_init_parameters():
    """Check if incorrect parameters raise correct errors."""
    # Assert
    with pytest.raises (TypeError, match=re.escape(
        "Dropout rate must be a number of type int or float!"
        )):
        layer = Layer_Dropout("3")
    with pytest.raises (TypeError, match=re.escape(
        "Dropout rate must be a number of type int or float!"
        )):
        layer = Layer_Dropout(None)
    with pytest.raises (ValueError, match=re.escape(
        "Dropout rate must be between 0 (inclusive) and 1 (exclusive)!"
        )):
        layer = Layer_Dropout(2)
    with pytest.raises (ValueError, match=re.escape(
        "Dropout rate must be between 0 (inclusive) and 1 (exclusive)!"
        )):
        layer = Layer_Dropout(-0.5)

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dropout_forward_dimensions(batch_size):
    """Checks if forwarding produces correct dimensions."""

    # Arrange
    layer = Layer_Dropout(dropout_rate=0.5)
    input_shape = (3, 5)
    inputs = np.ones(input_shape)

    # Act
    layer.forward(inputs)

    # Assert
    assert layer.output.shape == input_shape, \
        "Dropout Layer forwarding produces incorrect output dimensions!"
    assert layer.binary_mask.shape == input_shape, \
        "Dropout Layer forwarding produces incorrect binary mask dimensions!"