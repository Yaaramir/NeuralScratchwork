import nn_scratchwork as scratch
import nnfs
from nnfs.datasets import spiral_data

# Create modules
model = scratch.Model()
dense_1 = scratch.Layer_Dense(2, 64, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3)
activation_1 = scratch.Activation_ReLu()
dropout_1 = scratch.Layer_Dropout(0.1)
dense_2 = scratch.Layer_Dense(64, 3, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3 )
cce = scratch.Loss_CategoricalCrossEntropy()
adam = scratch.Optimizer_Adam()

# Create model
model.add_module(dense_1)
model.add_module(activation_1)
model.add_module(dropout_1)
model.add_module(dense_2)
model.add_loss_function(cce)
model.add_optimizer(adam)

# Create data
X_train, y_train = spiral_data(samples=100, classes=3)
X_val, y_val = spiral_data(samples=100, classes=3)

# Training
train_loss = None
model.training_mode()
for epoch in range(10001):

    # Forward pass
    (logits, loss, data_loss, reg_loss) = model.forward(X_train, y_train)
    train_loss = loss

    # Print progress
    if not epoch % 1000:
        print(f"Epoch: {epoch}, Loss: {loss:.3f} (Data Loss: {data_loss:.3f}, Regularization Loss: {reg_loss:.3f}), Learning Rate: {model.optimizer.current_learning_rate:.5f}")

    # Backward pass
    model.backward(cce.predictions, y_train)

    # Optimize
    model.optimize()

# Validation
val_loss = None
model.default_mode()
(output, loss, data_loss, reg_loss) = model.forward(X_val, y_val)
val_loss = loss

# Print evaluation
print(f"\n{train_loss:.3f} Training Loss")
print(f"{val_loss:.3f} Validation Loss ({(val_loss - train_loss):.3f})\n")

    