import neuralscratchwork as scratch
from nnfs.datasets import spiral_data
import numpy as np

class FCN_BinaryRegression:

    def run(self):

        # General settings
        dot_precision_workaround: bool = True
        default_dtype: str = 'float64'
        random_seed: int = 0

        # Create data
        n_samples_per_class = 100
        n_classes = 2

        X_train, y_train = spiral_data(n_samples_per_class, n_classes)
        y_train = y_train.reshape(-1, 1)
        acc_train = loss_train = None

        X_val, y_val = spiral_data(n_samples_per_class, n_classes)
        y_val = y_val.reshape(-1, 1)

        # Create modules
        dense_1 = scratch.Layer_Dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4)
        activation_1 = scratch.Activation_ReLu()
        dense_2 = scratch.Layer_Dense(64, 1)
        activation_2 = scratch.Activation_Sigmoid()
        bce = scratch.Loss_BinaryCrossEntropy()
        adam = scratch.Optimizer_Adam(decay=5e-7)


        # Training
        epochs: int = 10000
        #train_acc, train_loss = model.train(epochs, X_train, y_train)
        for epoch in range(epochs + 1):

            # Forward Pass
            dense_1.forward(X_train)
            activation_1.forward(dense_1.output)
            dense_2.forward(activation_1.output)
            activation_2.forward(dense_2.output)
            data_loss = bce.data_loss(activation_2.output, y_train)
            reg_loss = bce.regularization_loss(dense_1) + bce.regularization_loss(dense_2)
            loss_train = data_loss + reg_loss

            # Epoch evaluation
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

        # Validation
        dense_1.forward(X_val)
        activation_1.forward(dense_1.output)
        dense_2.forward(activation_1.output)
        activation_2.forward(dense_2.output)
        loss_val = data_loss = bce.data_loss(activation_2.output, y_val)

        # Evaluation
        predictions = (activation_2.output > 0.5) * 1
        acc_val = np.mean(predictions == y_val)

        print(f"\n{acc_train:.3f} Training Accuracy")
        print(f"{acc_val:.3f} Validation Accuracy ({(acc_val - acc_train):.3f})\n")

        print(f"{loss_train:.3f} Training Loss")
        print(f"{loss_val:.3f} Validation Loss ({(loss_val - loss_train):.3f})\n")

        print(f"Trained {n_samples_per_class} samples per class with {n_classes} classes in {epochs} epochs.\n")