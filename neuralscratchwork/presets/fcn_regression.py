import matplotlib.pyplot as plt
import neuralscratchwork as scratch
import nnfs
from nnfs.datasets import sine_data
import numpy as np

class FCN_Regression:

    def run(self):

        # General settings
        nnfs.init()

        # Create dataset
        X_train, y_train = sine_data()
        X_test, y_test = sine_data()

        # Plot training data
        plt.plot(X_train, y_train)
        plt.suptitle("Training Data")
        plt.show()

        # Create modules
        dense_1 = scratch.Layer_Dense(1, 64)
        activation_1 = scratch.Activation_ReLu()
        dense_2 = scratch.Layer_Dense(64, 64)
        activation_2 = scratch.Activation_ReLu()
        dense_3 = scratch.Layer_Dense(64, 1)
        activation_3 = scratch.Activation_Linear()
        mse = scratch.Loss_MeanSquaredError()
        adam = scratch.Optimizer_Adam(learning_rate=0.01, decay=1e-3)

        accuracy_precision = np.std(y_train) / 250

        # Training
        loss_train = acc_train = None
        n_epochs = 10000
        for epoch in range(n_epochs + 1):

            # Forward pass
            dense_1.forward(X_train)
            activation_1.forward(dense_1.output)
            dense_2.forward(activation_1.output)
            activation_2.forward(dense_2.output)
            dense_3.forward(activation_2.output)
            activation_3.forward(dense_3.output)
            data_loss = mse.data_loss(activation_3.output, y_train)
            reg_loss = mse.regularization_loss(dense_1) + mse.regularization_loss(dense_2) + mse.regularization_loss(dense_3)
            loss_train = data_loss + reg_loss

            # Accuracy
            predictions = activation_3.output
            acc_train = np.mean(np.absolute(predictions - y_train) < accuracy_precision)

            # Print training progress
            if not epoch % 1000:
                print(f"epoch: {epoch}, " +
                    f"accuracy: {acc_train:.3f}, " +
                    f"loss: {loss_train:.7f} (data loss: {data_loss:.7f}, regularization loss: {reg_loss:.7f}), " +
                    f"lr: {adam.current_learning_rate:.5f}")

            # Backward pass
            mse.backward(activation_3.output, y_train)
            activation_3.backward(mse.dinputs)
            dense_3.backward(activation_3.dinputs)
            activation_2.backward(dense_3.dinputs)
            dense_2.backward(activation_2.dinputs)
            activation_1.backward(dense_2.dinputs)
            dense_1.backward(activation_1.dinputs)

            # Optimization
            adam.pre_update_params()
            adam.update_params(dense_1)
            adam.update_params(dense_2)
            adam.update_params(dense_3)
            adam.post_update_params()

        # Testing
        dense_1.forward(X_test)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)
        dense_3.forward(activation_2.output)
        activation_3.forward(dense_3.output)

        # Calculate loss and accuracy
        loss_test = mse.data_loss(activation_3.output, y_test)
        acc_test = np.mean(np.absolute(activation_3.output - y_test) < accuracy_precision)

        print(f"\n{loss_train:.3f} Training Loss")
        print(f"{loss_test:.3f} Testing Loss ({(loss_test - loss_train):.3f})\n")
        print(f"{acc_train:.3f} Training Accuracy")
        print(f"{acc_test:.3f} Testing Accuracy ({(acc_test - acc_train):.3f})\n")

        plt.plot(X_test, y_test)
        plt.plot(X_test, activation_3.output)
        plt.suptitle("Test Data and Predictions")
        plt.show()