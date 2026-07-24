import nn_scratchwork as scratch
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

# Settings
np.random.seed(42)
#nnfs.init()

# Create modules
model = scratch.Model()
dense_1 = scratch.Layer_Dense(2, 512, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3)
activation_1 = scratch.Activation_ReLu()
dropout_1 = scratch.Layer_Dropout(0.1)
dense_2 = scratch.Layer_Dense(512, 512, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3 )
activation_2 = scratch.Activation_ReLu()
dropout_2 = scratch.Layer_Dropout(0.1)
dense_3 = scratch.Layer_Dense(512, 3)
cce = scratch.Loss_CategoricalCrossEntropy()
adam = scratch.Optimizer_Adam()

# Create model
model.add_module(dense_1)
model.add_module(activation_1)
model.add_module(dropout_1)
model.add_module(dense_2)
model.add_module(activation_2)
model.add_module(dropout_2)
model.add_module(dense_3)
model.add_loss_function(cce)
model.add_optimizer(adam)

# Create data
X_train, y_train = spiral_data(samples=1000, classes=3)
X_val, y_val = spiral_data(samples=100, classes=3)

# Training
model.activate_training_mode()

train_loss = None
train_accuracy = None
for epoch in range(10001):

    # Forward pass
    (train_accuracy, train_loss, data_loss, reg_loss) = model.forward(X_train, y_train)

    # Print progress
    if not epoch % 1000:
        print(f"Epoch: {epoch}, Accuracy: {train_accuracy:.3f}, Loss: {train_loss:.3f} (Data Loss: {data_loss:.3f}, Regularization Loss: {reg_loss:.3f}), Learning Rate: {model.optimizer.current_learning_rate:.5f}")

    # Backward pass
    model.backward(cce.predictions, y_train)

    # Optimize
    model.optimize()

# Validation
model.activate_default_mode()

(val_accuracy, val_loss, data_loss, reg_loss) = model.forward(X_val, y_val)

# Print evaluation
print(f"{train_accuracy:.3f} Training Accuracy")
print(f"{val_accuracy:.3f} Validation Accuracy ({(val_accuracy - train_accuracy):.3f})\n")

print(f"{train_loss:.3f} Training Loss")
print(f"{val_loss:.3f} Validation Loss ({(val_loss - train_loss):.3f})\n")