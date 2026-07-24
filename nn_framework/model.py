# TODO: Modell zusammenführen, backward pass für Softmax + CCE: Entweder TRENNEN oder Softmax festen teil von CCE werden lassen
import numpy as np

class model:

    def __init__(self):
        self.modules = []

    def training_mode(self):
        self.training_mode = True
        print("Setting model to training mode. Data will be forwarded by training modules.")

    def default_mode(self):
        self.training_mode = False
        print("Setting model to default mode. Data will not be forwarded by training modules.")

    def forward(self, inputs):
        x = inputs
        for module in self.modules:
            if not (getattr(module, "training_module", False) and not self.training_mode):
                module.forward(x)
                x = module.output