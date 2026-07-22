import nnfs
from nnfs.datasets import spiral_data
import numpy as np

# Genral settings
nnfs.init()

# Dense layer with weights and biases to process data flows
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

# Droput layer stabilizes the network and increases degree of generalization
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


# ReLU activation function to forward positive values only
class Activation_ReLu:
    
    # Forward pass
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

# Softmax for normalizing data flows and create probability predictions
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

# General loss class for data and regularization loss
class Loss:

    # Regularization loss calculation
    def regularization_loss(self, layer):

        # default = 0
        regularization_loss = 0

        # L1 weights
        if layer.weight_regularizer_l1 > 0:
            regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
        # L1 biases
        if layer.bias_regularizer_l1 > 0:
            regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
        # L2 weights
        if layer.weight_regularizer_l2 > 0:
            regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
        # L2 biases
        if layer.bias_regularizer_l2 > 0:
            regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)

        return regularization_loss

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Categorical Cross Entropy to calculate on probability predictions
class Loss_CategoricalCrossEntropy(Loss):

    # Forward pass 
    def forward(self, y_pred, y_true):

        # number of samples in a batch
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

# Adam optimizer combines the advantages of RMSprop and momentum by adjusting its approximations in early epochs.
class Optimizer_Adam:
    
    def __init__(self, learning_rate=1e-2, decay=5e-7, epsilon=1e-8, beta_1=0.9, beta_2=0.999):
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
X, y = spiral_data(samples=1000, classes=3)

# Create model
dense1 = Layer_Dense(2, 512,
                     weight_regularizer_l2=1e-3,
                     bias_regularizer_l2=1e-3)
activation1 = Activation_ReLu()
dropout1 = Layer_Dropout(0.1)
dense2 = Layer_Dense(512, 512,
                     weight_regularizer_l2=1e-3,
                     bias_regularizer_l2=1e-3 )
activation2 = Activation_ReLu()
dropout2 = Layer_Dropout(0.1)
dense3 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()
optimizer = Optimizer_Adam()

loss_train = -1
acc_train = -1
for epoch in range(10001):

    # Forward pass, loss and accuracy
    dense1.forward(X)
    activation1.forward(dense1.output)
    dropout1.forward(activation1.output)
    dense2.forward(dropout1.output)
    activation2.forward(dense2.output)
    dropout2.forward(activation2.output)
    dense3.forward(dropout2.output)
    data_loss = loss_activation.forward(dense3.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2) + loss_activation.loss.regularization_loss(dense3)
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    # For hot-one encoded labels only
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    acc = np.mean(predictions == y)
    loss_train = loss
    acc_train = acc

    if not epoch % 1000:
        print(f"epoch: {epoch}, " +
              f"accuracy: {acc:.3f}, " +
              f"loss: {loss:.3f} " +
              f"(data_loss: {data_loss:.3f}, " +
              f"regularization_loss: {regularization_loss:.3f}), " +
              f"learning rate: {optimizer.current_learning_rate}")

    # Backpropagation
    loss_activation.backward(loss_activation.output, y)
    dense3.backward(loss_activation.dinputs)
    dropout2.backward(dense3.dinputs)
    activation2.backward(dropout2.dinputs)
    dense2.backward(activation2.dinputs)
    dropout1.backward(dense2.dinputs)
    activation1.backward(dropout1.dinputs)
    dense1.backward(activation1.dinputs)

    # Update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

# VALIDATION
# Create dataset
X_val, y_val = spiral_data(samples=100, classes=3)

# Forward pass
dense1.forward(X_val)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)
dense3.forward(activation2.output)
loss_val = loss_activation.forward(dense3.output, y_val)

# Calculate accuracy
predictions = np.argmax(loss_activation.output, axis=1)
# For hot-one encoded labels only
if len(y_val.shape) == 2:
    y_val = np.argmax(y_val, axis=1)
acc_val = np.mean(predictions == y_val)

# Printing final results
print(f"\n{loss_train:.3f} TRAINING LOSS")
print(f"{loss_val:.3f} VALIDATION LOSS ({(loss_val - loss_train):.3f})\n")

print(f"{acc_train:.3f} TRAINING ACCURACY")
print(f"{acc_val:.3f} VALIDATION ACCURACY ({(acc_val - acc_train):.3f})\n")