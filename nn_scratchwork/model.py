# TODO: Modell zusammenführen, backward pass für Softmax + CCE: Entweder TRENNEN oder Softmax festen teil von CCE werden lassen
import numpy as np

class Model:

    def __init__(self):
        self.modules = []
        self.has_loss_function = False
        self.has_optimizer = False

    # In training mode, training modules like dropout layers are allowed to forward data
    def activate_training_mode(self):
        self.training_mode = True
        print("\nSetting model to training mode. Data will be forwarded through training modules.\n")

    # In training mode, training modules like dropout layers are not allowed to forward data
    def activate_default_mode(self):
        self.training_mode = False
        print("\nSetting model to default mode. Data will not be forwarded through training modules.\n")

    # Add a module to the model
    def add_module(self, module):
        self.modules.append(module)

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function
        self.has_loss_function = True
        

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.has_optimizer = True

    # Returns tuple (logits, TODO: accuracy, loss, data loss, regularization loss)
    def forward(self, X, y):
        if self.has_optimizer and self.has_loss_function:
            logits = X
            reg_loss = 0

            for module in self.modules:
                if not (getattr(module, "training_module", False) and not self.training_mode):
                    module.forward(logits)
                    logits = module.output
                if (hasattr(module, "weights") and self.training_mode):
                    reg_loss += self.loss_function.regularization_loss(module)

            data_loss = self.loss_function.forward(logits, y)
            loss = data_loss + reg_loss

            y_hat = np.argmax(self.loss_function.predictions, axis=1)
            if len(y.shape) == 2:
                y = np.argmax(y, axis=1)
            accuracy = np.mean(y_hat == y)

            return(accuracy, loss, data_loss, reg_loss)
        else:
            print("ERROR: Optimizer and/or Loss Function missing.")

    def backward(self, output, y):
        # Start backward pass in loss function
        self.loss_function.backward(output, y)
        dinputs = self.loss_function.dinputs

        for module in reversed(self.modules):
            module.backward(dinputs)
            dinputs = module.dinputs

    def optimize(self):
        self.optimizer.pre_update_params()
        for module in self.modules:
            if hasattr(module, "weights"):
                self.optimizer.update_params(module)
        self.optimizer.post_update_params()

    def train(self, epochs, X_train, y_train):
        self.activate_training_mode()
        print("NeuralScratchwork starts training.\n")

        for epoch in range(epochs+1):
            # Forward pass
            (train_acc, train_loss, data_loss, reg_loss) = self.forward(X_train, y_train)

            # Print progress
            if not epoch % 1000:
                print(f"Epoch: {epoch}, " +
                      f"Accuracy: {train_acc:.3f}, " +
                      f"Loss: {train_loss:.3f} " +
                      f"(Data Loss: {data_loss:.3f}, " +
                      f"Regularization Loss: {reg_loss:.3f}), " +
                      f"Learning Rate: {self.optimizer.current_learning_rate:.5f}")

            # Backward pass
            self.backward(self.loss_function.predictions, y_train)

            # Optimize
            self.optimize()

            self.activate_default_mode

        return train_acc, train_loss