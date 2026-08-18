import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import spiral_data
import numpy as np

# Neuralscratchwork modules
from ..modules import (
    Activation_ReLu,
    Activation_Sigmoid,
    Layer_Dense,
    Loss_BinaryCrossEntropy,
    Optimizer_Adam,
)

class FCN_BinaryRegression:

    def run(self):

        print("Starting preset: BINARY REGRESSION")

        # General settings
        nnfs.init()

        # Create data
        n_samples_per_class = 1000
        n_classes = 2

        X_train, y_train_raw = spiral_data(n_samples_per_class, n_classes)
        y_train = y_train_raw.reshape(-1, 1)
        acc_train = loss_train = None

        X_test, y_test_raw = spiral_data(n_samples_per_class, n_classes)
        y_test = y_test_raw.reshape(-1, 1)

        # Plot training data
        plt.scatter(X_train[:,0], X_train[:,1])
        plt.suptitle("Training data")
        plt.show()

        # Create modules
        dense_1 = Layer_Dense(2, 512, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
        activation_1 = Activation_ReLu()
        dense_2 = Layer_Dense(512, 1)
        activation_2 = Activation_Sigmoid()
        bce = Loss_BinaryCrossEntropy()
        adam = Optimizer_Adam(decay=5e-7)


        # Training
        epochs: int = 1000
        for epoch in range(epochs + 1):

            # Forward Pass
            dense_1.forward(X_train)
            activation_1.forward(dense_1.output)
            dense_2.forward(activation_1.output)
            activation_2.forward(dense_2.output)
            data_loss = bce.data_loss(activation_2.output, y_train)
            reg_loss = bce.regularization_loss(dense_1) + bce.regularization_loss(dense_2)
            loss_train = data_loss + reg_loss

            # Evaluate epoch and calculate loss
            predictions = (activation_2.output > 0.5) * 1
            acc_train = np.mean(predictions == y_train)

            if not epoch % 100:
                print(f"epoch: {epoch}, " +
                    f"acc: {acc_train:.3f}, " +
                    f"loss: {loss_train:.3f} (data_loss: {data_loss:.3f}, reg_loss: {reg_loss:.3f}), " +
                    f"lr: {adam.current_learning_rate:.6f}")

            # Backward Pass
            bce.backward(activation_2.output, y_train)
            activation_2.backward(bce.dinputs)
            dense_2.backward(activation_2.dinputs)
            activation_1.backward(dense_2.dinputs)
            dense_1.backward(activation_1.dinputs)

            # Optimizing
            adam.pre_update_params()
            adam.update_params(dense_1)
            adam.update_params(dense_2)
            adam.post_update_params()

        # Testing
        dense_1.forward(X_test)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)

        # Calculate loss and accuracy
        loss_test = data_loss = bce.data_loss(activation_2.output, y_test)
        predictions = (activation_2.output > 0.5) * 1
        acc_test = np.mean(predictions == y_test)

        # Plot testing results
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.scatter(X_test[:,0], X_test[:,1], c=predictions)
        ax1.set_title("Testing Predictions")
        ax2.scatter(X_test[:,0], X_test[:,1], c=y_test)
        ax2.set_title("Testing Targets")
        plt.show()

        print(f"\n{acc_train:.3f} Training Accuracy")
        print(f"{acc_test:.3f} Testing Accuracy ({(acc_test - acc_train):.3f})\n")

        print(f"{loss_train:.3f} Training Loss")
        print(f"{loss_test:.3f} Testing Loss ({(loss_test - loss_train):.3f})\n")

        print(f"Trained {n_samples_per_class} samples per class with {n_classes} classes in {epochs} epochs.\n")