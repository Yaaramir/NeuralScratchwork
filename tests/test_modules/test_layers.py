import pytest
import numpy as np

from neuralscratchwork.modules.layers import *

# Layer_Dense
def test_layer_dense_init_dimensions():
    """Checks if initialization outputs a layer with correct dimensions."""

    # Create layer
    input_channels = 12
    n_neurons = 23
    layer = Layer_Dense(input_channels, n_neurons)

    assert layer.weights.shape == (input_channels, n_neurons), \
        f"layer.weights.shape was expected to be ({input_channels, n_neurons}) but was {layer.weights.shape}"
    assert layer.biases.shape == (1, n_neurons), \
        f"layer.biases.shape was expected to be ({1, n_neurons}) but was {layer.biases.shape}"

def test_layer_dense_init_regularizer_assignments():
    """Checks if regularization values are assigned correctly"""

    # Create layer
    input_channels = 12
    n_neurons = 23
    weight_regularizer_l1 = 0.1
    bias_regularizer_l1 = 0.2
    weight_regularizer_l2 = 0.3
    bias_regularizer_l2 = 0.4
    layer = Layer_Dense(input_channels, n_neurons,
                        weight_regularizer_l1, bias_regularizer_l1,
                        weight_regularizer_l2, bias_regularizer_l2)

    assert layer.weight_regularizer_l1 == weight_regularizer_l1, \
        f"layer.weight_regularizer_l1 was expected to be {weight_regularizer_l1} but was {layer.weight_regularizer_l1}"
    assert layer.bias_regularizer_l1 == bias_regularizer_l1, \
        f"layer.bias_regularizer_l1 was expected to be {bias_regularizer_l1} but was {layer.bias_regularizer_l1}"
    assert layer.weight_regularizer_l2 == weight_regularizer_l2, \
        f"layer.weight_regularizer_l2 was expected to be {weight_regularizer_l2} but was {layer.weight_regularizer_l2}"
    assert layer.bias_regularizer_l2 == bias_regularizer_l2, \
        f"layer.bias_regularizer_l2 was expected to be {bias_regularizer_l2} but was {layer.bias_regularizer_l2}"

def test_layer_dense_init_random_weights_not_all_zero():
    """Checks if all randomly initialized weights are not zero"""
    input_channels = 4
    n_neurons = 2
    layer = Layer_Dense(input_channels, n_neurons)

    assert not np.all(layer.weights == 0), \
        f"Weights should not be all zeros"

@pytest.mark.parametrize("batch_size", [1, 4, 32, 128, 256, 512, 1024])
def test_layer_dense_forward_dimensions(batch_size):
    """Checks if forward outputs a lyer with correct dimensions."""

    # Create layer and inputs
    input_channels = 3
    n_neurons = 5
    layer = Layer_Dense(input_channels, n_neurons)
    inputs = np.random.randn(batch_size, input_channels)

    layer.forward(inputs)

    expected_shape = (batch_size, n_neurons)

    assert (
        layer.output.shape == expected_shape
    ), f"layer.output expected to be {expected_shape} but was {layer.output.shape}"