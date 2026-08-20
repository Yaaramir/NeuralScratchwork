import pytest
import numpy as np

from neuralscratchwork.modules.layers import *

# Layer_Dense
def test_layer_dense_init_dimensions():
    """Checks if initialization outputs a layer with correct dimensions."""

    # Arrange
    input_channels = 12
    n_neurons = 23

    # Act
    layer = Layer_Dense(input_channels, n_neurons, 0, 0, 0, 0)

    # Assert
    expected_weights_dimensions = (input_channels, n_neurons)
    expected_biases_dimensions = (1, n_neurons)

    assert layer.weights.shape == expected_weights_dimensions, (
        f"layer.weights.shape was expected to be {expected_weights_dimensions} "
        f"but was {layer.weights.shape}"
    )
    assert layer.biases.shape == expected_biases_dimensions, (
        f"layer.biases.shape was expected to be {expected_biases_dimensions} "
        f"but was {layer.biases.shape}"
    )

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
    assert layer.weight_regularizer_l1 == weight_regularizer_l1, (
        f"layer.weight_regularizer_l1 was expected to be {weight_regularizer_l1} "
        f"but was {layer.weight_regularizer_l1}"
    )
    assert layer.bias_regularizer_l1 == bias_regularizer_l1, (
        f"layer.bias_regularizer_l1 was expected to be {bias_regularizer_l1} "
        f"but was {layer.bias_regularizer_l1}"
    )
    assert layer.weight_regularizer_l2 == weight_regularizer_l2, (
        f"layer.weight_regularizer_l2 was expected to be {weight_regularizer_l2} "
        f"but was {layer.weight_regularizer_l2}"
    )
    assert layer.bias_regularizer_l2 == bias_regularizer_l2, (
        f"layer.bias_regularizer_l2 was expected to be {bias_regularizer_l2} "
        f"but was {layer.bias_regularizer_l2}"
    )

def test_layer_dense_init_regularizer_partly_assignment():
    """Checks is party assignment of regularizers works correctly."""
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
    assert layer.weight_regularizer_l1 == 0, (
        f"layer.weight_regularizer_l1 was expected to be 0 "
        f"but was {layer.weight_regularizer_l1}"
    )
    assert layer.bias_regularizer_l1 == bias_regularizer_l1, (
        f"layer.bias_regularizer_l1 was expected to be {bias_regularizer_l1} "
        f"but was {layer.bias_regularizer_l1}"
    )
    assert layer.weight_regularizer_l2 == weight_regularizer_l2, (
        f"layer.weight_regularizer_l2 was expected to be {weight_regularizer_l2} "
        f"but was {layer.weight_regularizer_l2}"
    )
    assert layer.bias_regularizer_l2 == 0, (
        f"layer.bias_regularizer_l2 was expected to be 0 "
        f"but was {layer.bias_regularizer_l2}"
    )

def test_layer_dense_init_random_weights_not_all_zero():
    """Checks if all randomly initialized weights are not zero"""

    # Arrange
    input_channels = 4
    n_neurons = 2

    # Act
    layer = Layer_Dense(input_channels, n_neurons)

    # Assert
    assert not np.all(layer.weights == 0), f"Weights should not be all zeros"

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dense_forward_dimensions(batch_size):
    """Checks if forward output has correct dimensions."""

    # Arrange
    input_channels = 3
    n_neurons = 5
    layer = Layer_Dense(input_channels, n_neurons)
    inputs = np.random.randn(batch_size, input_channels)

    # Act
    layer.forward(inputs)

    # Assert
    expected_dimensions = (batch_size, n_neurons)

    assert (
        layer.output.shape == expected_dimensions
    ), (
        f"layer.output.shape expected to be {expected_dimensions} "
        f"but was {layer.output.shape}"
    )

def test_layer_dense_forward_calculation():
    """Checks if forward calculates its output correctly."""

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
    assert layer.output == pytest.approx(expected_output), (
        f"layer.output expected to be {expected_output} "
        f"but was {layer.output}"
    )

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dense_backward_dimensions(batch_size):
    """Checks if backward output has correct dimensions."""
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

    assert layer.dweights.shape == expected_dweights_shape, (
        f"layer.dweights.shape expected to be {expected_dweights_shape} "
        f"but was {layer.dweights.shape}"
    )
    assert layer.dbiases.shape == expected_dbiases_shape, (
            f"layer.dweights.shape expected to be {expected_dbiases_shape} "
            f"but was {layer.dbiases.shape}"
        )
    assert layer.dinputs.shape == expected_dinputs_shape, (
            f"layer.dinputs.shape expected to be {expected_dinputs_shape} "
            f"but was {layer.dinputs.shape}"
        )

def test_layer_dense_backward_calculation():
    """Checks if backward calculates its output correctly."""

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

    np.testing.assert_allclose(layer.dweights, expected_dweights)
    np.testing.assert_allclose(layer.dbiases, expected_dbiases)
    np.testing.assert_allclose(layer.dinputs, expected_dinputs)