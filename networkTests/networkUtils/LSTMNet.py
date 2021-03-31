import torch
import random
import numpy as np  # numpy
import torch.nn as nn  # nn objects
import torch.optim as optim  # nn optimizers
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# NN model and forward method
class LSTMXORNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, random_h0=False, random_c0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.random_c0 = random_c0
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output lyaer
        # input must have the sahpe: batch_size, sequence_length, input_size

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_h0==True:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_c0==True:
            c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out