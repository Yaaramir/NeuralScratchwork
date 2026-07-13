import nnfs
from nnfs.datasets import spiral_data
import numpy as np

# Genral settings
nnfs.init()

# The network's layers consist of neurons, each with a set of weights and a bias. Weights are
# multiplied by the inputs, a bias is added and the result is passed to an activation function,
# which determines wether a neuron fires or not.
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
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

# Activation functions process a neuron's output and decide what value to forward or whether the
# neuron should fire. The ReLu (Rectified Linear Unit) forwards all outputs greater than 0.
class Activation_ReLu:
    
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# The Softmax activation function consists of two steps: First, exponentiating the inputs, and
# secon, normalizing these values. The results represent the model's probability predictions.
class Activation_Softmax:
    
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        # Exponentiate
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

# The two basic evaluation metrics are accuracy (comparing predictions to true values) and loss.
# Different loss functions handle different types of results, so choosing a well-suited function is
# essential for the model.
class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # default = 0
        regularization_loss = 0

        # L1
        # weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

        # L2
        # weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
        # biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)

        return regularization_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Cross Entropy works based on probability predictions. It penalizes uncertainty and
# its derivative can be computed efficiently. For those reasons, it is the default loss function
# for classification tasks.
class Loss_CategoricalCrossEntropy(Loss):

    # Forward pass 
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)

        # Clip y_pred since log(0) is not defined
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Categorical labels only
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]

        # One-hot encoded labels only
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    # Backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])
        # For categorical labels only
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / n_samples

# Combining Softmax and Categorical Cross Entropy enables a faster backward pass.
class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.calculate(self.output, y_true)
    
    # Backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues)
        # For hot-one encoded lables only
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
        self.dinputs = dvalues.copy()
        self.dinputs[range(n_samples), y_true] -= 1
        self.dinputs = self.dinputs / n_samples

# After each epoch (a full iteration over the dataset), the weights and biases of each layer and
# neuron must be updated to improve results - the optimization process. Different optimizers use
# different approaches and hyperparameters, leading to varying results and efficiency. While Adam
# has become a default optimizer, it is important to test and consider different types to achieve
# the best possible results.

# Stochastic Gradient Descent (SGD): Simple and efficient, but can get stuck in local minima and
# is not adaptive.
class Optimizer_SGD:
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # If used with momentum
        if self.momentum:

            # Create momentum for layer if not existend yet
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Weight updates
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Bias updates
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        else:

            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        layer.weights += weight_updates
        layer.biases += bias_updates

    def post_update_params(self):
        self.iterations += 1

# Adaptive Gradient Descent (Adagrad): Adaptive version of SGD, but more complex and tends to
# reduce learning rates too much over time. Better suited for smaller networks.
class Optimizer_AdaGrad:

    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))

    def update_params(self, layer):

        # Create caches for layer if not existend yet
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache=np.zeros_like(layer.biases)

        # Update cache
        layer.weight_cache += layer.dweights**2
        layer.bias_cache += layer.dbiases**2

        # Update parameters
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Root Mean Square Propagation (RMSprop): Improves upon Adagrad by avoiding excessively small
# learning rates, but requires hyperparameter tuning and can loose stability or start oscillating.
class Optimizer_RMSprop:
    def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. / 1. + self.decay * self.iterations)

    def update_params(self, layer):

        # Create caches for layer if not existend yet
        if not hasattr(layer, "weight_cache"):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update cache
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2

        # Update params
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# Adam: Combines the advantages of RMSprop and momentum by adjusting its approximations in early
# epochs. Adaptive, efficient and robust, but can sometimes reduce learning rate too aggressively.
# Default Optimizer.
class Optimizer_Adam:
    
    def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self, layer):
        # Create caches and momentums for layer if not existend yet
        if not hasattr(layer, "weight_cache"):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentums
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
        # Correct momentums
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update caches
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
        # Correct caches
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # Update parameters
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1

# TRAINING

# Create dataset
X, y = spiral_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 64)
activation1 = Activation_ReLu()
dense2 = Layer_Dense(64, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
#optimizer = Optimizer_SGD(decay=8e-8, momentum=0.9)
#optimizer = Optimizer_AdaGrad(decay=1e-4)
optimizer = Optimizer_Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):

    # Forward pass, loss and accuracy
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    # For hot-one encoded labels only
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    acc = np.mean(predictions == y)

    if not epoch % 100:
        print(f"epoch: {epoch}, accuracy: {acc:.3f}, loss: {loss:.3f}, learning rate: {optimizer.current_learning_rate}")

    # Backpropagation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

# VALIDATION

# Create dataset
X_test, y_test = spiral_data(samples=100, classes=3)

# Forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss_test = loss_activation.forward(dense2.output, y_test)

# Calculate accuracy
predictions = np.argmax(loss_activation.output, axis=1)
# For hot-oneencoded labels only
if len(y_test.shape) == 2:
    y_test = np.argmax(y_test, axis=1)
acc_test = np.mean(predictions == y_test)

print(f"validation, acc: {acc_test:.3f}, loss: {loss_test:.3f}")