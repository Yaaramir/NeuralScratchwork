import neuralscratchwork as scratch

# General settings
dot_precision_workaround: bool = True
default_dtype: str = 'float64'
random_seed: int = 0

# Create modules
model = scratch.Model()
# CNN
conv1 = scratch.Layer_Convolutional_2D(3, 6, 5)
relu1 = scratch.Activation_ReLu()
pool1 = scratch.Layer_Pooling_Maximum(2, 2)
conv2 = scratch.Layer_Convolutional_2D(6, 16, 5)
relu2 = scratch.Activation_ReLu()
pool2 = scratch.Layer_Pooling_Maximum(2, 2)
flat = scratch.Layer_Flattener()
# FCN
dense1 = scratch.Layer_Dense(16*5*5, 120)
relu3 = scratch.Activation_ReLu()
dense2 = scratch.Layer_Dense(120, 84)
relu4 = scratch.Activation_ReLu()
dense3 = scratch.Layer_Dense(84, 10)
# Loss and Optimizer
cce = scratch.Loss_CategoricalCrossEntropy()
adam = scratch.Optimizer_SGD()

# Create model
# CNN
model.add_module(conv1)
model.add_module(relu1)
model.add_module(pool1)
model.add_module(conv2)
model.add_module(relu2)
model.add_module(pool2)
model.add_module(flat)
# FCN
model.add_module(dense1)
model.add_module(relu3)
model.add_module(dense2)
model.add_module(relu4)
model.add_module(dense3)
# Loss and Optimizer
model.add_loss_function(cce)
model.add_optimizer(adam)

# Create data
X_train, y_train = 4, 2
X_val, y_val = 4, 2

# Training
epochs = 10
train_acc, train_loss = model.train(epochs, X_train, y_train)

# Validation
(val_acc, val_loss, data_loss, reg_loss) = model.forward(X_val, y_val)

# Print evaluation
print(f"{train_acc:.3f} Training Accuracy")
print(f"{val_acc:.3f} Validation Accuracy ({(val_acc - train_acc):.3f})\n")

print(f"{train_loss:.3f} Training Loss")
print(f"{val_loss:.3f} Validation Loss ({(val_loss - train_loss):.3f})\n")

# print(f"Training: {n_samples} samples with {n_classes} classes trained in {epochs} epochs.\n")