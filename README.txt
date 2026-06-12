This is an Artificial Neural Network created with "Neural Networks from Scratch in Python" (Kinsley H., Kukiela D., 2020).
The Neural Scratchwork uses raw python for the most part with only numpy to aid in array operations.
The whole network is created using only one file for both visualizing its simple approach and keeping the code as close to easy python syntax as possible.

Status Quo:
This network takes a dataset SPIRAL_DATA from nnfs.datasets and, using an Adam Optimizer, trains itself to set its weights and biases right to fit the data.
By using Adam we achieve a loss of 0.074 with an accuracy of 0.967 in 10000 epochs.