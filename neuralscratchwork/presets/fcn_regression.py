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

        plt.plot(X_train, y_train)
        plt.suptitle("Training Data")
        plt.show()

        # Create modules
        dense_1 = scratch.Layer_Dense(1, 64)
        activation_1 = scratch.Activation_ReLu()
        dense_2 = scratch.Layer_Dense(64, 1)
        activation_2 = scratch.Activation_Linear()
        mse = scratch.Loss_MeanSquaredError()
        adam = scratch.Optimizer_Adam()

        accuracy_precision = np.std(y_train) / 250

        # Training
        n_epochs = 10000
        for epoch in range(n_epochs + 1):

            # Forward pass
            dense_1.forward(X_train)
            activation_1.forward(dense_1.output)
            dense_2.forward(activation_1.output)
            activation_2.forward(dense_2.output)
            data_loss = mse.data_loss(activation_2.output, y_train)
            reg_loss = mse.regularization_loss(dense_1) + mse.regularization_loss(dense_2)
            loss = data_loss + reg_loss

            # Accuracy
            predictions = activation_2.output
            acc = np.mean(np.absolute(predictions - y_train) < accuracy_precision)

            # Print training progress
            if not epoch % 100:
                print(f"epoch: {epoch}, " +
                    f"accuracy: {acc}, " +
                    f"loss: {loss} (data loss: {data_loss}, regularization loss: {reg_loss}), " +
                    f"lr: {adam.current_learning_rate}")

            # Backward pass
            mse.backward(activation_2.output, y_train)
            activation_2.backward(mse.dinputs)
            dense_2.backward(activation_2.dinputs)
            activation_1.backward(dense_2.dinputs)
            dense_1.backward(activation_1.dinputs)

            # Optimization
            adam.pre_update_params()
            adam.update_params(dense_1)
            adam.update_params(dense_2)
            adam.post_update_params()

        # Validation
        dense_1.forward(X_test)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)

        plt.plot(X_test, y_test)
        plt.plot(X_test, activation_2.output)
        plt.suptitle("Test Data and Predictions")
        plt.show()