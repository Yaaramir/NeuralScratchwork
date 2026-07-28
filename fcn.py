import neuralscratchwork as scratch
from nnfs.datasets import spiral_data
import numpy as np

# General settings
dot_precision_workaround: bool = True
default_dtype: str = 'float64'
random_seed: int = 0

# Create data
n_samples_per_class = 100
n_classes = 3

X_train, y_train = spiral_data(n_samples_per_class, n_classes)
acc_train = loss_train = None

X_val, y_val = spiral_data(n_samples_per_class, n_classes)

# Create modules
dense_1 = scratch.Layer_Dense(2, 512, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3)
activation_1 = scratch.Activation_ReLu()
dropout_1 = scratch.Layer_Dropout(0.1)
dense_2 = scratch.Layer_Dense(512, 512, weight_regularizer_l2=1e-3, bias_regularizer_l2=1e-3 )
activation_2 = scratch.Activation_ReLu()
dropout_2 = scratch.Layer_Dropout(0.1)
dense_3 = scratch.Layer_Dense(512, n_classes)
cce = scratch.Loss_CategoricalCrossEntropy()
adam = scratch.Optimizer_Adam()

# Training
epochs: int = 10000
#train_acc, train_loss = model.train(epochs, X_train, y_train)
for epoch in range(epochs + 1):

    # Forward Pass
    dense_1.forward(X_train)
    activation_1.forward(dense_1.output)
    dropout_1.forward(activation_1.output)
    dense_2.forward(dropout_1.output)
    activation_2.forward(dense_2.output)
    dropout_2.forward(activation_2.output)
    dense_3.forward(dropout_2.output)
    data_loss = cce.forward(dense_3.output, y_train)
    reg_loss = cce.regularization_loss(dense_1) + cce.regularization_loss(dense_2) + cce.regularization_loss(dense_3)
    loss_train = loss = data_loss + reg_loss

    # Epoch evaluation
    predictions = np.argmax(cce.predictions, axis=1)
    if len(y_train.shape) == 2:
        y_train = np.argmax(y_train, axis=1)
    acc_train = accuracy = np.mean(predictions == y_train)

    if not epoch % 100:
        print(f"epoch: {epoch}, " +
              f"acc: {accuracy:.3f}, " +
              f"loss: {loss:.3f} (data_loss: {data_loss:.3f}, reg_loss: {reg_loss:.3f}), " +
              f"lr: {adam.current_learning_rate:.6f}")

    # Backward Pass
    cce.backward(cce.predictions, y_train)
    dense_3.backward(cce.dinputs)
    dropout_2.backward(dense_3.dinputs)
    activation_2.backward(dropout_2.dinputs)
    dense_2.backward(activation_2.dinputs)
    dropout_1.backward(dense_2.dinputs)
    activation_1.backward(dropout_1.dinputs)
    dense_1.backward(activation_1.dinputs)

    # Optimizing
    adam.pre_update_params()
    adam.update_params(dense_1)
    adam.update_params(dense_2)
    adam.update_params(dense_3)
    adam.post_update_params()

# Validation
dense_1.forward(X_val)
activation_1.forward(dense_1.output)
dropout_1.forward(activation_1.output)
dense_2.forward(dropout_1.output)
activation_2.forward(dense_2.output)
dropout_2.forward(activation_2.output)
dense_3.forward(dropout_2.output)
data_loss = cce.forward(dense_3.output, y_val)
reg_loss = cce.regularization_loss(dense_1) + cce.regularization_loss(dense_2) + cce.regularization_loss(dense_3)
loss_val = data_loss + reg_loss

# Evaluation
predictions = np.argmax(cce.predictions, axis=1)
if len(y_val.shape) == 2:
    y_val = np.argmax(y_val, axis=1)
acc_val = accuracy = np.mean(predictions == y_val)

print(f"\n{acc_train:.3f} Training Accuracy")
print(f"{acc_val:.3f} Validation Accuracy ({(acc_val - acc_train):.3f})\n")

print(f"{loss_train:.3f} Training Loss")
print(f"{loss_val:.3f} Validation Loss ({(loss_val - loss_train):.3f})\n")

print(f"Trained {n_samples_per_class} samples per class with {n_classes} classes in {epochs} epochs.\n")