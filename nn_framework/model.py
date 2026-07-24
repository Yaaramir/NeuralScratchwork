# TODO: Modell zusammenführen, backward pass für Softmax + CCE: Entweder TRENNEN oder Softmax festen teil von CCE werden lassen
import numpy as np

class Model:

    def __init__(self):
        self.modules = []
        self.has_loss_function = False
        self.has_optimizer = False

    # In training mode, training modules like dropout layers are allowed to forward data
    def training_mode(self):
        self.training_mode = True
        print("Setting model to training mode. Data will be forwarded by training modules.")

    # In training mode, training modules like dropout layers are not allowed to forward data
    def default_mode(self):
        self.training_mode = False
        print("Setting model to default mode. Data will not be forwarded by training modules.")

    # Add a module to the model
    def add_module(self, module):
        self.modules.append(module)

    def add_loss_function(self, loss_function):
        self.loss_function = loss_function
        self.has_loss_function = True
        

    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.has_optimizer = True

    # Returns tuple (output, TODO: accuracy, loss, data loss, regularization loss)
    def forward(self, X, y):
        if self.has_optimizer and self.has_loss_function:
            x = X
            reg_loss = 0

            for module in self.modules:
                if not (getattr(module, "training_module", False) and not self.training_mode):
                    module.forward(x)
                    x = module.output
                    if hasattr(module, "weights"):
                        reg_loss += self.loss_function.regularization_loss(self, module)
            data_loss = self.loss_function.forward(self, x, y)
            loss = data_loss + reg_loss

            return(x, loss, data_loss, reg_loss)
        else:
            print("ERROR: Optimizer and/or Loss Function missing.")

    def backward(self, output, y):
        # Start backward pass in loss function
        self.loss_function.backward(output, y)
        dinputs = self.loss_function.dinputs

        for module in reversed(self.modules):
            if not (getattr(module, "training_module", False) and not self.training_mode):
                 module.backward(self, dinputs)
                 dinputs = module.dinputs

    def optimize(self):
        self.optimizer.pre_update_params()
        for module in self.modules:
            if hasattr(module, "weights"):
                self.optimizer.update_params(module)
        self.optimizer.post_update_params()