import torch
import random
import numpy as np  # numpy
import torch.nn as nn  # nn objects
import torch.optim as optim  # nn optimizers
import matplotlib.pyplot as plt

class XORNet(nn.Module):
    # input_size = # of attributes put into the input of the nn - creates first layer
    # hidden_size = # nodes to have in the 1st hidden layer - too many leads to overfitting
    # output_size = # number of neurons to have in the output layer
    # need to add hidden layers because there is no linear seperability in single-layer perceptron network

    def __init__(self, input_size, hidden_size, output_size):  # in this constructor method we'll define the fully connected layers
        super().__init__()  # initializes superclass (nn.Module); could also use super(XORNet, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)  # nn.Linear is used when you're doing a fc layers
        self.hidden_layer = nn.Linear(hidden_size, output_size)  # nn = input -> hidden layer -> output

    def forward(self, x):  # here we have a feedforward nn in which data (i.e. x) goes forward from input to output
        x = torch.tanh(self.input_layer(x))  # first, x passes through input layer to hidden layer w/ sigmoid activ
        x = self.hidden_layer(x)  # then, x passes through the hidden layer to output layer
        return x