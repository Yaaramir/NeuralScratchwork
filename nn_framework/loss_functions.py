import numpy as np

class Loss:

    # data loss without L1, L2 regularization
    def data_loss(self, output, y):
            sample_losses = self.forward(output, y)
            data_loss = np.mean(sample_losses)
            return data_loss

    # L1 and L2 regularization loss
    def regularization_loss(self, layer):
    
            # default = 0
            regularization_loss = 0
    
            # L1 weights
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
            # L1 biases
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            # L2 weights
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights**2)
            # L2 biases
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases**2)
    
            return regularization_loss

# Categorical Cross Entropy to calculate on probability predictions
class Loss_CategoricalCrossEntropy(Loss):

    def module_info(self):
            return("CCE")

    # Forward pass 
    def forward(self, dvalues, y_true):

        n_samples = len(dvalues)

        # Clip y_pred since log(0) is not defined
        clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # For categorical labels
        if len(y_true.shape) == 1:
             n_labels = len(dvalues[0])
             y_true_one_hot = np.eye(n_labels)[y_true]
        # For one-hot encoded labels
        else:
             y_true_one_hot = y_true
            
        self.dinputs = -y_true_one_hot / clipped_dvalues
        self.dinputs = self.dinputs / n_samples
    
    # Backward pass
    def backward(self, dvalues, y_true):
        n_samples = len(dvalues)
        n_labels = len(dvalues[0])
        # For categorical labels only
        if len(y_true.shape) == 1:
            y_true = np.eye(n_labels)[y_true]
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / n_samples