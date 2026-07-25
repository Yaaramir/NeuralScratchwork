import nn_scratchwork as scratch
from nnfs.datasets import spiral_data

# General settings
dot_precision_workaround: bool = True
default_dtype: str = 'float64'
random_seed: int = 0

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
n_samples = 1000
n_classes = 3
X_train, y_train = spiral_data(n_samples, n_classes)
X_val, y_val = spiral_data(100, 3)

# Training
epochs = 10000
train_acc, train_loss = model.train(epochs, X_train, y_train)

# Validation
(val_acc, val_loss, data_loss, reg_loss) = model.forward(X_val, y_val)

# Print evaluation
print(f"{train_acc:.3f} Training Accuracy")
print(f"{val_acc:.3f} Validation Accuracy ({(val_acc - train_acc):.3f})\n")

print(f"{train_loss:.3f} Training Loss")
print(f"{val_loss:.3f} Validation Loss ({(val_loss - train_loss):.3f})\n")

print(f"Training: {n_samples} samples with {n_classes} classes trained in {epochs} epochs.\n")