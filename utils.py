import torch
import random
import numpy as np  # numpy
import torch.nn as nn  # nn objects
import torch.optim as optim  # nn optimizers
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #set device to gpu if possible


#### NETWORK RELATED FUNCTIONS ####

### Match to Sample RNNs ###

# Vanilla Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class VanillaMTSNet(nn.Module):
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
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out
    
# GRU Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class GRUMTSNet(nn.Module):
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
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.gru(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out
    

# LSTM Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class LSTMMTSNet(nn.Module):
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
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_c0==True:
            c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        out = self.fc(out)  # passes last RNN layer to fc output layer
        return out

# RNNCell Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class RNNCellMTSNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.rnncell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # BUT in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): 
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
        
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx = self.rnncell(x[i], hx)

        out = self.fc(hx.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
        return out

# GRUCell Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class GRUCellMTSNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.grucell = nn.GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # BUT in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): 
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)

        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx = self.grucell(x[i], hx)

        out = self.fc(hx.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
        return out

# LSTMCell Match to Sample-suited NN (using XOR problem as basis) - Determine whether or not you've seen the first vector before
class LSTMCellMTSNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.lstmcell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # BUT in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): 
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
        cx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)

        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
        if self.random_cx==True:
            cx = torch.randn(self.batch_size, self.hidden_size).to(device)
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx, cx = self.lstmcell(x[i], (hx, cx))

        out = self.fc(hx.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
        return out

### Dual Retro-cue RNNs ###
    
# Vanilla Dual Retrocue-suited NN - The network makes two guesses based on different probes
class VanillaDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, delay1, delay2, random_h0=False, random_c0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.random_c0 = random_c0
        self.delay1 = delay1
        self.delay2 = delay2
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size

    def forward(self, x): 
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
        
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        if self.random_h0==True:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        out, hx = self.rnn(x, h0)  # first, x passes through the recurrent layer
        
        hx1 = out[:, probe1_layer, :] # grab the hidden state created after being shown the first cue
        hx2 = out[:, probe2_layer, :] # grab the hidden state created after being shown the second cue
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
        
        return out    
    
# GRU Dual Retrocue-suited NN - The network makes two guesses based on different probes
class GRUDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, delay1, delay2, random_h0=False, random_c0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.random_c0 = random_c0
        self.delay1 = delay1
        self.delay2 = delay2
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size

    def forward(self, x): 
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
        
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        if self.random_h0==True:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        out, hx = self.gru(x, h0)  # first, x passes through the recurrent layer
        
        hx1 = out[:, probe1_layer, :] # grab the hidden state created after being shown the first cue
        hx2 = out[:, probe2_layer, :] # grab the hidden state created after being shown the second cue
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
        
        return out
    
# LSTMCell Dual Retrocue-suited NN - The network makes two guesses based on different probes
class LSTMDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 num_classes, batch_size, delay1, delay2, random_h0=False, random_c0=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_h0 = random_h0
        self.random_c0 = random_c0
        self.delay1 = delay1
        self.delay2 = delay2
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size

    def forward(self, x): 
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
        
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        if self.random_h0==True:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        if self.random_c0==True:
            c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        
        out, (hx, cx) = self.lstm(x, (h0, c0))  # first, x passes through the recurrent layer
        
        hx1 = out[:, probe1_layer, :] # grab the hidden state created after being shown the first cue
        hx2 = out[:, probe2_layer, :] # grab the hidden state created after being shown the second cue
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
        
        return out
    
# RNNCell Dual Retrocue-suited NN - The network makes two guesses based on different probes
class RNNCellDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, delay1, delay2, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.delay1 = delay1
        self.delay2 = delay2
        self.rnncell = nn.RNNCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): # note that x is the initial vector input (i.e., the dual retro-cue vectors for a single sample)
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
      
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
        
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
                
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx = self.rnncell(x[i], hx)    
            
            if i == probe1_layer:
                hx1 = hx   
            
            if i == probe2_layer:
                hx2 = hx
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
                                                        
        return out

# GRUCell Dual Retrocue-suited NN - The network makes two guesses based on different probes
class GRUCellDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, delay1, delay2, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.delay1 = delay1
        self.delay2 = delay2
        self.grucell = nn.GRUCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): # note that x is the initial vector input (i.e., the dual retro-cue vectors for a single sample)
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
      
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
        
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
                
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx = self.grucell(x[i], hx)    
            
            if i == probe1_layer:
                hx1 = hx   
            
            if i == probe2_layer:
                hx2 = hx
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
                                        
        return out
    
# LSTMCell XOR-suited NN - The network makes two guesses based on different probes
class LSTMCellDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, delay1, delay2, random_hx=False, random_cx=False):  # define fc/reccurent layers
        super().__init__()  # initializes superclass (nn.Module)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.random_hx = random_hx
        self.random_cx = random_cx
        self.delay1 = delay1
        self.delay2 = delay2
        self.lstmcell = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the shape: batch_size, sequence_length, input_size normally
        # in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): # note that x is the initial vector input (i.e., the dual retro-cue vectors for a single sample) 
        
        probe1_layer = 2 + self.delay1 + 1 + 1 - 1 # two stimuli + delay + first cue + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + self.delay2 + 1 + 1 # first probe's length + second delay + second cue + second probe
        
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size) 
        cx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
                
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
            
        if self.random_cx==True:
            cx = torch.randn(self.batch_size, self.hidden_size).to(device)
        
        for i in range(x.size()[0]): # goes through each time step (i.e. sequence length)
            hx, cx = self.lstmcell(x[i], (hx, cx)) 
            
            if i == probe1_layer:
                hx1 = hx
                cx1 = cx
            
            if i == probe2_layer:
                hx2 = hx
                cx2 = cx
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
                                                
        return out

### Memory-Augmented Dual Retro-cue RNNs ### 

## This still needs to be added
    
#### TASK RELATED FUNCTIONS ####


"""
Generates a data sample (i.e. matrix) filled with n (n = seqlen1 + seqlen2 + 2) 
vectors of size input_size. The sample contains 2 non-distraction vectors (i.e. 
"input vectors") located at input1_location and input2_location and the rest of
the sample contains "distraction vectors" (i.e. random vectors filled with
numbers between 0 and 1) located everywhere else. Depending on the sequence
type, the generated sample will have equivalent or different non-distraction
vectors. Can run for match-to-sample test (default) or for dual retro-cue
"""
def generate_sample(sequence_type, input_size, seqlen1, seqlen2, seqlen3=0, test_type='match_to_sample'):
    
    if test_type == 'match_to_sample':
        
        # total sequence length = 2 input vectors + distractions
        sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
        X_sample = torch.zeros(sequence_length, input_size)
        Y_sample = 0

        # corner cases
        if input_size == 1:
            X0 = torch.tensor([1])
            X1 = torch.tensor([0])

        else:
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
    
    if test_type == 'dual_retro_cue':
        
        # total sequence length = input vectors + 2 delays + 2 cues + 2 probes
        '''
        This is similar to two simultaneous match-to-sample tasks. You are shown two 
        stimuli, then cued on which stimuli will be tested and then probed on the stimuli
        (i.e., asked if the probed stimuli matched the cue and the original stimuli
        
        Think of it like a person being shown two types of images on a screen. Image A
        is shown on the right and Image B is shown on the left. The cue is a left-pointing
        arrow and a right-pointing arrow. If you're cued a left arrow then your goal is to
        determine whether or not whatever is shown on the screen to the left matches what
        was shown at the beginning (the "0" case) or not (the "1" case)
        ''' 
        
        sequence_length = 6 + seqlen1 + seqlen2
        X_sample = torch.zeros(sequence_length, input_size)
        Y_sample = [[1,0], [1,0]]
        cue_X0 = torch.ones(input_size) * 2
        cue_X1 = torch.ones(input_size) * 3
        
        # corner cases
        if input_size == 1:
            X0 = torch.tensor([1])
            X1 = torch.tensor([0])
            probe_X0 = X0
            probe_X1 = X1

        else:
            X0 = torch.tensor(np.eye(input_size)[0])
            X1 = torch.tensor(np.eye(input_size)[1])
            probe_X0 = X0
            probe_X1 = X1
        
        X_sample[0] = X0 # first stimulus
        X_sample[1] = X1 # second stimulus
        
        ### completely matching cases ###
                
        # the normal 1a case: different cues, probes match, start with cue A
        if sequence_type == 0:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 0] 
            
        # the repeat 1b case: same cues, probes match (stay match), start with cue A
        if sequence_type == 1:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 0]
            
        # the normal 2a case: different cues, probes match, start with cue B
        if sequence_type == 2:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 0] 
            
        # the repeat 2b case: same cues, probes match (stay match), start with cue B
        if sequence_type == 3:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 0]
            
        
        ### "switch" cases ###
            
        # the mismatch 1a case: different cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches the second stimulus
        if sequence_type == 4: 
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 1b case: different cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches a random stimulus
        if sequence_type == 5:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1,0]
            
        # the mismatch 1c case: different cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches the second stimulus
        if sequence_type == 6: 
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 1d case: different cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches a random stimulus
        if sequence_type == 7:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1,0]
            
        # the mismatch 2a case: different cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches the first stimulus
        if sequence_type == 8:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 2b case: different cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches a random stimulus
        if sequence_type == 9:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 2c case: different cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches the first stimulus
        if sequence_type == 10:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 2d case: different cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches a random stimulus
        if sequence_type == 11:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 3a case: same cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches the second stimulus
        if sequence_type == 12: 
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 3b case: same cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches a random stimulus
        if sequence_type == 13:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
            
        # the mismatch 3c case: same cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches the second stimulus
        if sequence_type == 14: 
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 3d case: same cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches a random stimulus
        if sequence_type == 15:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
            
        # the mismatch 4a case: same cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches the second stimulus
        if sequence_type == 16:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 4b case: same cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches a random stimulus
        if sequence_type == 17:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 4c case: same cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches the second stimulus
        if sequence_type == 18:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 4d case: same cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches a random stimulus
        if sequence_type == 19:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the complete mismatch 1a case: different cues, probes match opposite stimuli, start with cue A
        if sequence_type == 20:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 1b case: different cues, probes match random stimuli, start with cue A
        if sequence_type == 21:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
            
        # the complete mismatch 1c case: different cues, probes match opposite stimuli, start with cue B
        if sequence_type == 22:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 1d case: different cues, probes match random stimuli, start with cue B
        if sequence_type == 23:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2a case: same cues, probes match opposite stimuli, start with cue A
        if sequence_type == 24:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2b case: same cues, probes match random stimuli, start with cue A
        if sequence_type == 25:
            X_sample[seqlen1 + 2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
            
        # the complete mismatch 2c case: same cues, probes match opposite stimuli, start with cue B
        if sequence_type == 26:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2d case: same cues, probes match random stimuli, start with cue B
        if sequence_type == 27:
            X_sample[seqlen1 + 2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[sequence_length - 2] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
        
    return X_sample, Y_sample

"""
Generates a dataset of 4 different versions of the XOR problem 
(in the match-to-sample [MTS] case) or  28 different versions 
of a class-type identification problem (in the dual-retro cue [DRC] 
case) based on the matricies made with the generate_sample function. 
The parameters for the sequence length/delay, distraction vectors, 
and randomization of order of samples shown can be controlled based
on the input parameters. Default test is MTS
"""
def generate_dataset(input_size, seqlen1, seqlen2, seqlen3=0, same_distractions=False, random=False, test_type='match_to_sample'):

    if test_type == 'match_to_sample':
        
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

    if test_type == 'dual_retro_cue':
        
        sequence_length = 6 + seqlen1 + seqlen2
        dataset = torch.zeros(28, sequence_length, input_size)
        
        if random:
            sample_1 = generate_sample(0, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_2 = generate_sample(1, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_3 = generate_sample(2, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_4 = generate_sample(3, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_5 = generate_sample(4, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_6 = generate_sample(5, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_7 = generate_sample(6, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_8 = generate_sample(7, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_9 = generate_sample(8, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_10 = generate_sample(9, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_11 = generate_sample(10, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_12 = generate_sample(11, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_13 = generate_sample(12, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_14 = generate_sample(13, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_15 = generate_sample(14, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_16 = generate_sample(15, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_17 = generate_sample(16, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_18 = generate_sample(17, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_19 = generate_sample(18, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_20 = generate_sample(19, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_21 = generate_sample(20, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_22 = generate_sample(21, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_23 = generate_sample(22, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_24 = generate_sample(23, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_25 = generate_sample(24, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_26 = generate_sample(25, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_27 = generate_sample(26, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_28 = generate_sample(27, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            sample_set = [sample_1, sample_2, sample_3, sample_4, sample_5, sample_6, sample_7, sample_8, 
                          sample_9, sample_10, sample_11, sample_12, sample_13, sample_14, sample_15, sample_16, 
                          sample_17, sample_18, sample_19, sample_20, sample_21, sample_22, sample_23, sample_24, 
                          sample_25, sample_26, sample_27, sample_28]
            np.random.shuffle(sample_set)
            dataset[0], Y0 = sample_set[0]
            dataset[1], Y1 = sample_set[1]
            dataset[2], Y2 = sample_set[2]
            dataset[3], Y3 = sample_set[3]
            dataset[4], Y4 = sample_set[4]
            dataset[5], Y5 = sample_set[5]
            dataset[6], Y6 = sample_set[6]
            dataset[7], Y7 = sample_set[7]
            dataset[8], Y8 = sample_set[8]
            dataset[9], Y9 = sample_set[9]
            dataset[10], Y10 = sample_set[10]
            dataset[11], Y11 = sample_set[11]
            dataset[12], Y12 = sample_set[12]
            dataset[13], Y13 = sample_set[13]
            dataset[14], Y14 = sample_set[14]
            dataset[15], Y15 = sample_set[15]
            dataset[16], Y16 = sample_set[16]
            dataset[17], Y17 = sample_set[17]
            dataset[18], Y18 = sample_set[18]
            dataset[19], Y19 = sample_set[19]
            dataset[20], Y20 = sample_set[20]
            dataset[21], Y21 = sample_set[21]
            dataset[22], Y22 = sample_set[22]
            dataset[23], Y23 = sample_set[23]
            dataset[24], Y24 = sample_set[24]
            dataset[25], Y25 = sample_set[25]
            dataset[26], Y26 = sample_set[26]
            dataset[27], Y27 = sample_set[27]
            
        if not random:
            dataset[0], Y0 = generate_sample(0, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[1], Y1 = generate_sample(1, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[2], Y2 = generate_sample(2, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[3], Y3 = generate_sample(3, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[4], Y4 = generate_sample(4, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[5], Y5 = generate_sample(5, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[6], Y6 = generate_sample(6, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[7], Y7 = generate_sample(7, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[8], Y8 = generate_sample(8, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[9], Y9 = generate_sample(9, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[10], Y10 = generate_sample(10, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[11], Y11 = generate_sample(11, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[12], Y12 = generate_sample(12, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[13], Y13 = generate_sample(13, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[14], Y14 = generate_sample(14, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[15], Y15 = generate_sample(15, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[16], Y16 = generate_sample(16, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[17], Y17 = generate_sample(17, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[18], Y18 = generate_sample(18, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[19], Y19 = generate_sample(19, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[20], Y20 = generate_sample(20, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[21], Y21 = generate_sample(21, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[22], Y22 = generate_sample(22, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[23], Y23 = generate_sample(23, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[24], Y24 = generate_sample(24, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[25], Y25 = generate_sample(25, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[26], Y26 = generate_sample(26, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            dataset[27], Y27 = generate_sample(27, input_size, seqlen1, seqlen2, test_type='dual_retro_cue')
            
        targets = torch.tensor([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9, Y10, Y11, Y12, Y13, Y14, Y15, Y16, Y17, 
                                Y18, Y19, Y20, Y21, Y22, Y23, Y24, Y25, Y26, Y27])
    
    return dataset, targets, sequence_length


#### TRAINING RELATED FUNCTIONS ####


# trains network
# generate_new -- generates new datasets and targets (of the same type of task) for the network to train on
# generate_random -- shuffles the generated target and dataset order so the network doesn't simply learn the order of predictions to make but the type
# same_distractions -- makes the smame distractions as part of the distraction vector as opposed to different ones
def train_network(network, dataset, targets, sequence_length, input_size, batch_size, epochs, optimizer, criterion, sheduler, generate_new=True, generate_random=True, same_distractions=False, condition=None, verbosity=2, cell=True, test_type="match_to_sample"):     
    mean_losses = []
    
#     dataset = dataset[0:2]  # give it a smaller initial training set before it moves on to more data (dual retro-cue)
#     targets = targets[0:2]  # give it a smaller initial training set before it moves on to more data (daul retro-cue)
            
    for epoch in range(epochs):
        losses = []
        
        if generate_new and condition is not None:
                        
            seqlen1, seqlen2, seqlen3 = condition[0], condition[1], 0
            
            if test_type == "match_to_sample":
                seqlen3 = condition[2]
            
            dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, same_distractions=same_distractions, random=generate_random, test_type=test_type)
                        
        if str(criterion) == "BCELoss()":
            targets = targets.to(torch.float)
        
        for sample, target in zip(dataset, targets):
            
            optimizer.zero_grad() 
                        
            if cell == False: # for regular RNN using batch first
                sample = sample.view(batch_size, sequence_length, input_size)
            if cell == True: # for RNNCell using sequence length first
                sample = sample.view(sequence_length, batch_size, input_size)
                                            
            # forward propagation
            output = network(sample)  # pass each sample into the network
                      
            loss = criterion.forward(output, target)  # forward propagates loss
            losses.append(loss.detach())

            # backpropagation
            loss.backward()  # backpropagates the loss

            # gradient descent/adam step
            optimizer.step()
            
        mean_loss = sum(losses) / len(losses)  # calculates mean loss for epoch
        mean_losses.append(mean_loss)
        sheduler.step(mean_loss)
        
        if verbosity > 0:
            if epoch in [0, epochs/4, epochs/2, 3*epochs/4, epochs-1]:
                print(f'Cost at epoch {epoch} is {mean_loss}')
                if verbosity > 1:
                    print()
                    print(f'Input = {sample.detach().reshape(sequence_length, input_size)}')
                    print(f'Targets = {target.detach()}')
                    print(f'Predictions = {output.detach()}')
                    print()

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