import neuralscratchwork as scratch

# General settings
dot_precision_workaround: bool = True
default_dtype: str = 'float64'
random_seed: int = 0

# Create modules
model = scratch.Model()
conv1 = scratch.Layer_Convolutional_2D(3, 6, 5)
pool1 = scratch.Layer_Pooling_Maximum(2, 2)
conv2 = scratch.Layer_Convolutional_2D(6, 16, 5)
pool2 = scratch.Layer_Pooling_Average(2, 2)
dense1 = scratch.Layer_Dense()
relu1 = scratch.Activation_ReLu()
drop = scratch.Layer_Dropout(0.1)
dense2 = scratch.Layer_Dense(bla, 10)
cce = scratch.Loss_CategoricalCrossEntropy()
sgd = scratch.Optimizer_SGD()

# Create model
model.add_module(conv1)
model.add_module(pool1)
model.add_module(conv2)
model.add_module(pool2)
model.add_module(dense1)
model.add_module(relu1)
model.add_module(drop)
model.add_module(dense2)
model.add_loss_function(cce)
model.add_optimizer(sgd)

# Create data
# TODO

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

print(f"Training: {n_samples} samples with {n_classes} classes trained in {epochs} epochs.\n")