# NeuralScratchwork
An artificial neural network developed with raw Python and NumPy only.

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)]()

---

## Table of Contents
- [About the project](#goals)
- [Status Quo](#statusquo)
- [Whats's next?](#todo)

## About the project
NeuralScratchwork is one of three simple neural networks created for classification exercises. Each is coded using a different set of frameworks:

- [NeuralScratchwork](https://github.com/Yaaramir/NeuralScratchwork): This network is created with raw Python and only implements NumPy to organize and utilize data in arrays. This repository dictates the speed and content of the other two, as it serves as the template for both.
- [NeuralTorchwork](https://github.com/Yaaramir/NeuralTorchwork): Based on NeuralScratchwork this project makes use of the [PyTorch framework](https://pytorch.org/) developed by Meta's AI Research lab.
- [NeuralFlowwork](https://github.com/Yaaramir/NeuralFlowwork): Based on NeuralScratchwork this project makes use of the [TensorFlow framework](https://www.tensorflow.org/) developed by Alphabet Inc.'s Google Brain Team.

The first goal is to implement a complete network from scratch in ***NeuralScratchwork*** that can be trained and used for simple classification exercises while implementing the PyTorch and TensorFlow solutions simultaneously.

Once that stage is completed, ***NeuralTorchwork*** will be further developed to be deployed for scientific usage within the [OpenFlexure](https://openflexure.org/) project, while ***NeuralFlowwork*** will transformed in an office and smart home scenario.

Since understanding how neural networks work at its core and learning how to use them successfully is and has been the main goal of this project, development does not necessarily follow the fastes or most efficient way, but often takes a detour to fully capture the edges, boundaries, challenges and oportunities the frameworks and underlying paradigms offer.

Idea and architecture of the NeuralScratchwork are conceived and heavily inspired by [Neural Networks from Scratch](https://nnfs.io/) (Kinsley H., Kukiela D., 2020).

## Status Quo
- A simple model with two linear dense layers, ReLU, and Softmax activation functions has been implemented. CCE has been chosen for loss calculation and Adam as the optimizer.
- L1 and L2 regularization has been implemented and experimenting with different hyperparameters is beginning to find best settings possible
- A 2D dataset with three classes of dots spiraling around a center point is implemented.

![Spiral Data](./assets/spiral_data.png)
- The network trains for 10k epochs by performing forward passes, backward passes, gradient calculation, and parameter updating.
- A test dataset is used to evaluate model performance after training.

![Test Results](./assets/train_and_val_results.png)

## What's next?
Next step will be to implement a dropout layer to further stabilize the network. Afterwards other output layers and regression will be considered, beore the network will be opened for other and unknown types of data.