import torch
import random
import numpy as np  # numpy
import torch.nn as nn  # nn objects
import torch.optim as optim  # nn optimizers
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set device to gpu if possible


#### NETWORK RELATED FUNCTIONS ####


# feedforward XOR-suited NN
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
    

# recurrent XOR-suited NN
class RecurrentXORNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, random_h0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output lyaer
        # input must have the sahpe: batch_size, sequence_length, input_size

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_h0==True:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out
    
# recurrent XOR-suited NN
class GRUXORNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, random_h0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output lyaer
        # input must have the sahpe: batch_size, sequence_length, input_size

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_h0==True:
            h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.gru(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out
    

# LSTM XOR-suited NN
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
    

    
#### TASK RELATED FUNCTIONS ####


"""
Generates a data sample (i.e. matrix) filled with n (n = seqlen1 + seqlen2 + 2) 
vectors of size input_size. The sample contains 2 non-distraction vectors (i.e. 
"input vectors") located at input1_location and input2_location and the rest of
the sample contains "distraction vectors" (i.e. random vectors filled with
numbers between 0 and 1) located everywhere else. Depending on the sequence
type, the generated sample will have equivalent or different non-distraction
vectors
"""
def generate_sample(sequence_type, input_size, seqlen1, seqlen2, seqlen3):
    sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
    X_sample = torch.zeros(sequence_length, input_size)
    Y_sample = 0
    X0 = torch.tensor(np.eye(input_size)[0])
    X1 = torch.tensor(np.eye(input_size)[1])

    # input vectors
    if sequence_type == 0:
        X_sample[seqlen1] = X0
        X_sample[seqlen1 + seqlen2 + 1] = X0
        Y_sample = [0]
    if sequence_type == 1:
        X_sample[seqlen1] = X0
        X_sample[seqlen1 + seqlen2 + 1] = X1
        Y_sample = [1]
    if sequence_type == 2:
        X_sample[seqlen1] = X1
        X_sample[seqlen1 + seqlen2 + 1] = X1
        Y_sample = [0]
    if sequence_type == 3:
        X_sample[seqlen1] = X1
        X_sample[seqlen1 + seqlen2 + 1] = X0
        Y_sample = [1]

    # distraction vectors
    for i in range(sequence_length):
        if i != seqlen1 and i != (seqlen1 + seqlen2 + 1):
            X_sample[i] = torch.rand(input_size)

    return X_sample, Y_sample

"""
Generates a dataset of 4 different versions of the XOR problem based on the
matricies made with the generate_sample function. This dataset will contain
the equivalent of matricies corresponding to the [0, 0, 0], [0, 1, 1], 
[1, 1, 0], and [1, 0, 1] examples of the XOR problem in that order
"""
def generate_dataset(same_distractions, input_size, seqlen1, seqlen2, seqlen3, random=False):
    sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
    dataset = torch.zeros(4, sequence_length, input_size)

    if random:
        sample_1 = generate_sample(0, input_size, seqlen1, seqlen2, seqlen3)
        sample_2 = generate_sample(1, input_size, seqlen1, seqlen2, seqlen3)
        sample_3 = generate_sample(2, input_size, seqlen1, seqlen2, seqlen3)
        sample_4 = generate_sample(3, input_size, seqlen1, seqlen2, seqlen3)
        sample_set = [sample_1, sample_2, sample_3, sample_4]
        np.random.shuffle(sample_set)
        dataset[0], Y0 = sample_set[0]
        dataset[1], Y1 = sample_set[1]
        dataset[2], Y2 = sample_set[2]
        dataset[3], Y3 = sample_set[3]
    
    if not random:
        dataset[0], Y0 = generate_sample(0, input_size, seqlen1, seqlen2, seqlen3)
        dataset[1], Y1 = generate_sample(1, input_size, seqlen1, seqlen2, seqlen3)
        dataset[2], Y2 = generate_sample(2, input_size, seqlen1, seqlen2, seqlen3)
        dataset[3], Y3 = generate_sample(3, input_size, seqlen1, seqlen2, seqlen3)
    
    # when true sets all dataset samples to the have same distraction vectors
    if same_distractions:
        for i in range(sequence_length):
            if i != seqlen1 and i != (seqlen1 + seqlen2 + 1):
                dataset[1][i] = dataset[0][i]
                dataset[2][i] = dataset[0][i]
                dataset[3][i] = dataset[0][i]

    targets = torch.tensor([Y0, Y1, Y2, Y3])
    return dataset, targets, sequence_length


#### TRAINING RELATED FUNCTIONS ####


# trains network
def train_network(network, dataset, targets, sequence_length, input_size, batch_size, epochs, optimizer, criterion, sheduler, generate_new=False, generate_random=False, same_distractions=False, condition=None, verbose=True):     
    mean_losses = []
    for epoch in range(epochs):
        losses = []
        
        if generate_new and condition is not None:
            seqlen1, seqlen2, seqlen3 = condition[0], condition[1], condition[2]
            dataset, targets, sequence_length = generate_dataset(same_distractions, input_size, seqlen1, seqlen2, seqlen3, random=generate_random)

        for sample, target in zip(dataset, targets):
            optimizer.zero_grad() 
            sample = sample.view(batch_size, sequence_length, input_size)

            # forward propagation
            output = network(sample)  # pass each sample into the network
            loss = criterion.forward(output, target)  # forward propagates loss
            losses.append(loss)

            # backpropagation
            loss.backward()  # backpropagates the loss

            # gradient descent/adam step
            optimizer.step()

        mean_loss = sum(losses) / len(losses)  # calculates mean loss for epoch
        mean_losses.append(mean_loss)
        if verbose:
            # sheduler.step(mean_loss)
            if epoch in [0, epochs/4, epochs/2, 3*epochs/4, epochs-1]:
              print(f'Cost at epoch {epoch} is {mean_loss}')

    return np.array(mean_losses)


#### TESTING RELATED FUNCTIONS ####


# Tests rounded network outputs against correct network outputs based on sample
def test_network(sample_number, dataset, targets, network, input_size, batch_size, sequence_length):
    # Test network
    test_sample = sample_number
    test_data = dataset[test_sample].view(batch_size, sequence_length, 
                                          input_size)
    test_targets = torch.tensor([0.0, 0.0])
    if targets[test_sample] != 0:
        test_targets = torch.tensor([1.0, 0.0])
    else:
        test_targets = torch.tensor([0.0, 1.0])
    test = network(test_data)
    test = torch.softmax(test, dim=1)

    print("\n")
    print("Test of network: ")
    print("input is {}".format(test_data.detach().numpy()))
    print('out is {}'.format(torch.round(test).detach().numpy()))
    print('expected out is {}'.format(test_targets.detach().numpy()))
    
# Compares and plots the loss of four different networks
def plot_four_losses(title, loss1, loss2=None, loss3=None, loss4=None):
    # Plot mean losses
    # plt.figure()
    plt.suptitle(title)
    plt.plot(loss1, color='red')
    if loss2 is not None:
        plt.plot(loss2, color='blue')
    if loss3 is not None:
        plt.plot(loss3, color='orange')
    if loss4 is not None:
        plt.plot(loss4, color='green')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    
# Plots the mean loss of up to four different networks
def plot_mean_losses(title, loss1, loss2=None, loss3=None, loss4=None, *args, **kwargs):
    # Plot mean losses
    # plt.figure()
    plt.suptitle(title)
    plt.plot(loss1, color='red', *args, **kwargs)
    if loss2 is not None:
        plt.plot(loss2, color='blue', *args, **kwargs)
    if loss3 is not None:
        plt.plot(loss3, color='orange', *args, **kwargs)
    if loss4 is not None:
        plt.plot(loss4, color='green', *args, **kwargs)
    plt.xlabel('epoch')
    plt.ylabel('mean loss')
    
# Plots the individual loss of all the networks of a specific type
def plot_individual_losses(losses, *args, **kwargs):
    for loss in losses:
        plt.plot(loss, *args, **kwargs)