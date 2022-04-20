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
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the sahpe: batch_size, sequence_length, input_size

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_h0==True:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        
        out = self.fc(out.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
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
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
        # input must have the sahpe: batch_size, sequence_length, input_size

    def forward(self, x): 
        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)
        if self.random_h0==True:
            h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size).to(device)
        out, _ = self.gru(x, h0)  # first, x passes through the recurrent layer
        out = out[:, -1, :]  # only last layer of the RNN
        
        out = self.fc(out.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
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
        self.fc = nn.Linear(hidden_size, num_classes)  # fc output layer
        self.sigmoid = nn.Sigmoid()
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
        
        out = self.fc(out.squeeze())  # passes last RNN layer to fc output layer
        out = self.sigmoid(out)
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
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device)
        
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
        
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
        
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
        
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
      
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
      
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
        
        probe1_layer = 2 + 1 + self.delay1 + 1 - 1 # two stimuli + first cue + first delay + probe - 1 (for 0 based indexing)
        probe2_layer = probe1_layer + 1 + self.delay2 + 1 # first probe's length + second cue + second delay + second probe
        
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

# Differentiable nerual dictionary class which will allow for episodic memory within a neural network
# inspired by qihong lu's work on dnds ~ https://github.com/qihongl/dnd-lstm
class DND():
    
    '''
    Here the dictionary length is the size of the number of memories to save and the
    dimension is the state for which memory will be saved. Think of it like a ever-growing
    list of states and memories of that state
    
    We can modify the maximum number of states that can be saved and the maximum number of
    memories that can be saved
    '''
    def __init__(self):
        
        # dictonary representing episodic memory 
        self.dnd = {}
        
        # dynamic state
        self.encoding_off = False
        self.retrieval_off = False
       
        # allocate space for memories
        self.reset_memory()
    
    # resets the memory for a particular state's dnd
    def reset_memory(self):
        self.dnd.clear()

    # saves the memory for a particular state's dnd
    def save_memory(self, memory_key, memory_val):
       
        if self.encoding_off:
            return
        
        # add new memory to the the dictionary
        self.dnd.update({memory_key : memory_val})
        
#         # remove the oldest memory, if overflow
#         if len(self.keys) > self.dict_len:
#             self.keys.pop(0)
#             self.vals.pop(0)
    
    def get_memory(self, query_key):
        return self.dnd.get(query_key)
    
# Context representations that allow important features from WM to be used in task
class CR():
    def __init__(self):
        
        # context representations used in WMEM
        self.position_code_1 = 0 # stimulus shown first
        self.position_code_2 = 1 # stimulus shown second
        self.trial_tag = None # what trial type is this (i.e., what trial number is being used)
        self.temporal_drift = 0 # a representation of how much time has passed in general
    
    # updates/sets the trial tag
    def update_trial_tag(self, new_trial_tag):
        self.trial_tag = new_trial_tag
    
    # updates/sets the temporal drift
    def update_temporal_drift(self):
        self.temporal_drift += 1
    
    # updates/sets the trial tag
    def reset_trial_tag(self):
        self.trial_tag = None
    
    # updates/sets the temporal drift
    def reset_temporal_drift(self):
        self.temporal_drift = 0
        
    def get_position_code_1(self):
        return self.position_code_1
    
    def get_position_code_2(self):
        return self.position_code_2
    
    def get_trial_tag(self):
        return self.trial_tag
    
    def get_temporal_drift(self):
        return self.temporal_drift

# Memeory Augmented LSTMCell XOR-suited NN with Differentiable Neural Dictionary - The network makes two guesses based on different probes
class DNDLSTMCellDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, delay1, delay2, memory_rule, random_hx=False, random_cx=False):
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
        self.dnd = DND() # differentiable neural dictionary (dictionary of cell states)
        self.memory_rule = memory_rule
        self.sigmoid = nn.Sigmoid() # sigmoidal output function
        # input must have the shape: batch_size, sequence_length, input_size normally
        # in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): # note that x is the initial vector input (i.e., the dual retro-cue vectors for a single sample)
        
        num_states = x.size()[0]
        states = x
        
        stimulus1_layer = 0
        stimulus1 = states[0]
        stimulus2_layer = 1
        stimulus2 = states[1]
        cue1_layer = 2 + 1 - 1 # two stimuli + delay + first cue - 1 (for 0 based indexing)
        probe1_layer = cue1_layer + self.delay1 + 1 # first probe comes right after first cue + first delay
        cue2_layer = probe1_layer + 1 # first probe's length + second cue
        probe2_layer = cue2_layer + self.delay2 + 1 # second probe comes right after second cue + second delay
        
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size) 
        cx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
                
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
            
        if self.random_cx==True:
            cx = torch.randn(self.batch_size, self.hidden_size).to(device)
        
        for state, timestamp in zip(states, range(num_states)): # goes through each time step (i.e. sequence length)
            
            if (timestamp != probe1_layer) and (timestamp != probe2_layer):
                hx, cx = self.lstmcell(x[timestamp], (hx, cx)) 
                
                # memory rule to remember the cell states of the stimuli 
                if self.memory_rule == 'remember_stimuli_states':
                    if timestamp == stimulus1_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is stimulus 1) 
                        self.dnd.save_memory(str(state), cx) 
                    if timestamp == stimulus2_layer:
                        # save the current stimluli's cell state into a dnd; (here the stimuli is stimulus 2)
                        self.dnd.save_memory(str(state), cx) 
                        
                # memory rule to remember the cell states of the cues and their timestamp
                if self.memory_rule == 'remember_cue_directions':
                    if timestamp == cue1_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is the cue) 
                        self.dnd.save_memory(str(state), cx) 
                    if timestamp == cue2_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is the cue) 
                        self.dnd.save_memory(str(state), cx) 
            
            if timestamp == probe1_layer:
                
                # memory rules states to check if you've seen this before
                # if it's in your long term memory, pull it up, otherwise view it as completely random
                if self.memory_rule == 'remember_stimuli_states':
                    # check if the probe matches a stimlulus we've seen before then pull it out of memory
                    if str(state) in [str(stimulus1), str(stimulus2)]:
                        cx_saved = self.dnd.get_memory(str(state))      
                    else:
                        cx_saved = torch.randn(self.batch_size, self.hidden_size).to(device)
                
                # memory rule states just to pull up your memory of what the cue direction was
                if self.memory_rule == 'remember_cue_directions':
                    cx_saved = self.dnd.get_memory(str(states[cue1_layer]))
                
                hx, cx = self.lstmcell(x[timestamp], (hx, cx_saved))
                hx1 = hx
            
            if timestamp == probe2_layer:
               
                # memory rules states to check if you've seen this before 
                # if it's in your long term memory, pull it up, otherwise view it as completely random
                if self.memory_rule == 'remember_stimuli_states':
                    # check if the probe matches a stimlulus we've seen before then pull it out of memory
                    if str(state) in [str(stimulus1), str(stimulus2)]:
                        cx_saved = self.dnd.get_memory(str(state))   
                    else:
                        cx_saved = torch.randn(self.batch_size, self.hidden_size).to(device)
                
                # memory rule states just to pull up your memory of what the cue direction was
                if self.memory_rule == 'remember_cue_directions':
                    cx_saved = self.dnd.get_memory(str(states[cue2_layer]))
                
                hx, cx = self.lstmcell(x[timestamp], (hx, cx_saved))
                hx2 = hx
        
        out1 = self.fc(hx1).squeeze() # passes RNN after cue1 layer (flattened) to fc layer
        out1 = self.sigmoid(out1)
        
        out2 = self.fc(hx2).squeeze() # passes RNN after cue2 layer (flattened) to fc layer
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
                                                
        return out
    
    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()
            
# Working Memory Episodic Memory Augmented LSTMCell XOR-suited NN with Differentiable Neural Dictionary - The network makes two guesses based on different probes
class WMEMLSTMCellDRCNet(nn.Module):
    def __init__(self, input_size, hidden_size,
                 num_classes, batch_size, delay1, delay2, memory_rule, random_hx=False, random_cx=False):
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
        self.sequence_length = 6 + delay1 + delay2 # 2 stimuli + 2 cues + 2 probes + delays
        self.num_position_codes = 2
        self.num_trial_tags = 28 * 2 # unique samples in drc task (28 per dataset) & 2 different datasets used for switch stimuli/reusing stimuli
        self.num_temporal_drifts = self.num_trial_tags * self.sequence_length
        self.context_represenation_size = self.num_position_codes + self.num_trial_tags + 1
        self.fc = nn.Linear(hidden_size + self.context_represenation_size, num_classes)
        self.dnd = DND() # differentiable neural dictionary (dictionary of cell states)
        self.cr = CR() # context represtations of position codes, trial tags, and temporal drift
        self.memory_rule = memory_rule
        self.sigmoid = nn.Sigmoid() # sigmoidal output function
        # input must have the shape: batch_size, sequence_length, input_size normally
        # in this case we want our input to be of shape sequence_length, batch_size, input_size

    def forward(self, x): # note that x is the initial vector input (i.e., the dual retro-cue vectors for a single sample)
        
        num_states = x.size()[0]
        states = x
        
        stimulus1_layer = 0
        stimulus1 = states[0]
        stimulus2_layer = 1
        stimulus2 = states[1]
        cue1_layer = 2 + 1 - 1 # two stimuli + delay + first cue - 1 (for 0 based indexing)
        probe1_layer = cue1_layer + self.delay1 + 1 # first probe comes right after first cue + first delay
        cue2_layer = probe1_layer + 1 # first probe's length + second cue
        probe2_layer = cue2_layer + self.delay2 + 1 # second probe comes right after second cue + second delay
        
        hx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size) 
        cx = torch.zeros(self.batch_size, self.hidden_size).to(device) # (batch, hidden_size)
                
        if self.random_hx==True:
            hx = torch.randn(self.batch_size, self.hidden_size).to(device)
            
        if self.random_cx==True:
            cx = torch.randn(self.batch_size, self.hidden_size).to(device)
        
        for state, timestamp in zip(states, range(num_states)): # goes through each time step (i.e. sequence length)
            
            if (timestamp != probe1_layer) and (timestamp != probe2_layer):
                hx, cx = self.lstmcell(x[timestamp], (hx, cx)) 
                
                # memory rule to remember the cell states of the stimuli 
                if self.memory_rule == 'remember_stimuli_states':
                    if timestamp == stimulus1_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is stimulus 1) 
                        self.dnd.save_memory(str(state), cx) 
                    if timestamp == stimulus2_layer:
                        # save the current stimluli's cell state into a dnd; (here the stimuli is stimulus 2)
                        self.dnd.save_memory(str(state), cx) 
                        
                # memory rule to remember the cell states of the cues and their timestamp
                if self.memory_rule == 'remember_cue_directions':
                    if timestamp == cue1_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is the cue) 
                        self.dnd.save_memory(str(state), cx) 
                    if timestamp == cue2_layer:
                        # save the current stimuli's cell state into a dnd; (here the stimuli is the cue) 
                        self.dnd.save_memory(str(state), cx) 
            
            if timestamp == probe1_layer:
                
                # context representations
                if states[cue1_layer][0].mean() == 2.0:
                    probe1_position_code = torch.eye(self.num_position_codes)[self.cr.get_position_code_1()] # one hot encoded position code
                else:
                    probe1_position_code = torch.eye(self.num_position_codes)[self.cr.get_position_code_2()] # one hot encoded position code
                probe1_trial_tag = torch.eye(self.num_trial_tags)[self.cr.get_trial_tag()]
                probe1_temporal_drift = torch.tensor(self.cr.get_temporal_drift() / self.num_temporal_drifts).view(1)
                
                # em representations
                
                # memory rules states to check if you've seen this before
                # if it's in your long term memory, pull it up, otherwise view it as completely random
                if self.memory_rule == 'remember_stimuli_states':
                    # check if the probe matches a stimlulus we've seen before then pull it out of memory
                    if str(state) in [str(stimulus1), str(stimulus2)]:
                        cx_saved = self.dnd.get_memory(str(state))      
                    else:
                        cx_saved = torch.randn(self.batch_size, self.hidden_size).to(device)
                
                # memory rule states just to pull up your memory of what the cue direction was
                if self.memory_rule == 'remember_cue_directions':
                    cx_saved = self.dnd.get_memory(str(states[cue1_layer]))
                
                hx, cx = self.lstmcell(x[timestamp], (hx, cx_saved))
                hx1 = hx
            
            if timestamp == probe2_layer:
                
                # context representations
                if states[cue1_layer][0].mean() == 2.0:
                    probe2_position_code = torch.eye(self.num_position_codes)[self.cr.get_position_code_1()] # one hot encoded position code
                else:
                    probe2_position_code = torch.eye(self.num_position_codes)[self.cr.get_position_code_2()] # one hot encoded position code
                probe2_trial_tag = torch.eye(self.num_trial_tags)[self.cr.get_trial_tag()]
                probe2_temporal_drift = torch.tensor(self.cr.get_temporal_drift() / self.num_temporal_drifts).view(1) # normalize so temporal drift size isn't enormous
               
                # memory rules states to check if you've seen this before 
                # if it's in your long term memory, pull it up, otherwise view it as completely random
                if self.memory_rule == 'remember_stimuli_states':
                    # check if the probe matches a stimlulus we've seen before then pull it out of memory
                    if str(state) in [str(stimulus1), str(stimulus2)]:
                        cx_saved = self.dnd.get_memory(str(state))   
                    else:
                        cx_saved = torch.randn(self.batch_size, self.hidden_size).to(device)
                
                # memory rule states just to pull up your memory of what the cue direction was
                if self.memory_rule == 'remember_cue_directions':
                    cx_saved = self.dnd.get_memory(str(states[cue2_layer]))
                
                hx, cx = self.lstmcell(x[timestamp], (hx, cx_saved))
                hx2 = hx
                
            self.cr.update_temporal_drift() # increase the temporal drift by 1
        
        em1_representation = hx1.squeeze()
        context1_representation = torch.cat([probe1_position_code, probe1_trial_tag, probe1_temporal_drift]).to(device)
        wmem1_representation = torch.cat([em1_representation, context1_representation])
        
        out1 = self.fc(wmem1_representation).squeeze() # passes RNN after cue1 layer (flattened) to fc layer (EM stimuli representation)
        out1 = self.sigmoid(out1)
        
        em2_representation = hx2.squeeze()
        context2_representation = torch.cat([probe2_position_code, probe2_trial_tag, probe2_temporal_drift]).to(device)
        wmem2_representation = torch.cat([em2_representation, context2_representation])
        
        out2 = self.fc(wmem2_representation).squeeze() # passes RNN after cue1 layer (flattened) to fc layer (EM stimuli representation)
        out2 = self.sigmoid(out2)
        
        out = torch.stack([out1, out2])
                                                
        return out
    
    def turn_off_encoding(self):
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()
    
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
def generate_sample(sequence_type, input_size, seqlen1, seqlen2, seqlen3=0, test_type='match_to_sample', stimulus1=None, stimulus2=None):
    
    if test_type == 'match_to_sample':
        
        # total sequence length = 2 input vectors + distractions
        sequence_length = 2 + seqlen1 + seqlen2 + seqlen3
        X_sample = torch.zeros(sequence_length, input_size)
        Y_sample = 0

        # corner cases
        if input_size == 1:
            
            if (stimulus1 is not None):
                X0 = stimulus1
            else:
                X0 = torch.tensor([1])
            
            if (stimulus2 is not None):
                X1 = stimulus2
            else:
                X1 = torch.tensor([0])

        else:
            if (stimulus1 is not None):
                X0 = stimulus1
            else:
                X0 = torch.tensor(np.eye(input_size)[0])
            
            if (stimulus2 is not None): 
                X1 = stimulus2
            else:
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
        
#         cue_X0 = torch.ones(input_size) * 2
#         cue_X1 = torch.ones(input_size) * 3
        
        # best to represent the cues on their own dimension
        cue_X0 = torch.tensor(np.eye(input_size)[input_size - 2])
        cue_X1 = torch.tensor(np.eye(input_size)[input_size - 1])
        
        # corner cases
        if input_size == 1:

            if (stimulus1 is not None):
                X0 = stimulus1
            else:
                X0 = torch.tensor([1])
            
            if (stimulus2 is not None):
                X1 = stimulus2
            else:
                X1 = torch.tensor([0])
            
            probe_X0 = X0
            probe_X1 = X1

        else:
            
            if (stimulus1 is not None):
                X0 = stimulus1
            else:
                X0 = torch.tensor(np.eye(input_size)[0])
            
            if (stimulus2 is not None): 
                X1 = stimulus2
            else:
                X1 = torch.tensor(np.eye(input_size)[1])
            
            probe_X0 = X0
            probe_X1 = X1
        
        X_sample[0] = X0 # first stimulus
        X_sample[1] = X1 # second stimulus
        
        ### completely matching cases ###
                
        # the normal 1a case: different cues, probes match, start with cue A
        if sequence_type == 0:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 0] 
            
        # the repeat 1b case: same cues, probes match (stay match), start with cue A
        if sequence_type == 1:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 0]
            
        # the normal 2a case: different cues, probes match, start with cue B
        if sequence_type == 2:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 0] 
            
        # the repeat 2b case: same cues, probes match (stay match), start with cue B
        if sequence_type == 3:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 0]
            
        
        ### "switch" cases ###
            
        # the mismatch 1a case: different cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches the second stimulus
        if sequence_type == 4: 
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 1b case: different cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches a random stimulus
        if sequence_type == 5:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1,0]
            
        # the mismatch 1c case: different cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches the second stimulus
        if sequence_type == 6: 
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 1d case: different cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches a random stimulus
        if sequence_type == 7:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1,0]
            
        # the mismatch 2a case: different cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches the first stimulus
        if sequence_type == 8:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 2b case: different cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches a random stimulus
        if sequence_type == 9:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 2c case: different cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches the first stimulus
        if sequence_type == 10:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 2d case: different cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches a random stimulus
        if sequence_type == 11:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 3a case: same cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches the second stimulus
        if sequence_type == 12: 
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 3b case: same cues, first probe doesn't match first cue, start with cue A
        # here the first probe matches a random stimulus
        if sequence_type == 13:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 0]
            
        # the mismatch 3c case: same cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches the second stimulus
        if sequence_type == 14: 
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
        
        # the mismatch 3d case: same cues, first probe doesn't match first cue, start with cue B
        # here the first probe matches a random stimulus
        if sequence_type == 15:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 0]
            
        # the mismatch 4a case: same cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches the second stimulus
        if sequence_type == 16:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 4b case: same cues, second probe doesn't match second cue, start with cue A
        # here the second probe matches a random stimulus
        if sequence_type == 17:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the mismatch 4c case: same cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches the second stimulus
        if sequence_type == 18:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [0, 1]
        
        # the mismatch 4d case: same cues, second probe doesn't match second cue, start with cue B
        # here the second probe matches a random stimulus
        if sequence_type == 19:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [0, 1]
            
        # the complete mismatch 1a case: different cues, probes match opposite stimuli, start with cue A
        if sequence_type == 20:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 1b case: different cues, probes match random stimuli, start with cue A
        if sequence_type == 21:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
            
        # the complete mismatch 1c case: different cues, probes match opposite stimuli, start with cue B
        if sequence_type == 22:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 1d case: different cues, probes match random stimuli, start with cue B
        if sequence_type == 23:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2a case: same cues, probes match opposite stimuli, start with cue A
        if sequence_type == 24:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = probe_X1 # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = probe_X1 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2b case: same cues, probes match random stimuli, start with cue A
        if sequence_type == 25:
            X_sample[2] = cue_X0 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X0 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
            
        # the complete mismatch 2c case: same cues, probes match opposite stimuli, start with cue B
        if sequence_type == 26:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = probe_X0 # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = probe_X0 # second probe
            Y_sample = [1, 1]
        
        # the complete mismatch 2d case: same cues, probes match random stimuli, start with cue B
        if sequence_type == 27:
            X_sample[2] = cue_X1 # first cue
            X_sample[seqlen1 + 3] = torch.rand(input_size) # first probe
            X_sample[seqlen1 + 4] = cue_X1 # second cue
            X_sample[sequence_length - 1] = torch.rand(input_size) # second probe
            Y_sample = [1, 1]
        
    return X_sample, Y_sample

"""
Generates a dataset of 4 different versions of the XOR problem 
(in the match-to-sample [MTS] case) or 28 different versions 
of a class-type identification problem (in the dual-retro cue [DRC] 
case) based on the matricies made with the generate_sample function. 
The parameters for the sequence length/delay, distraction vectors, 
and randomization of order of samples shown can be controlled based
on the input parameters. Default test is MTS
"""
def generate_dataset(input_size, seqlen1, seqlen2, seqlen3=0, random=False, test_type='match_to_sample', stimulus1=None, stimulus2=None):
    
    if test_type == 'match_to_sample':
        num_samples = 4
        sequence_length = 2 + seqlen1 + seqlen2 + seqlen3

    if test_type == 'dual_retro_cue':
        num_samples = 28
        sequence_length = 6 + seqlen1 + seqlen2
    
    dataset = torch.zeros(num_samples, sequence_length, input_size).to(device)
    sample_set = [] # set of samples and targets for task
    targets = [] # set of targets for task
        
    if random:

        if random:
            for i in range(num_samples):
                sample_set.append(generate_sample(i, input_size, seqlen1, seqlen2, seqlen3, test_type=test_type, stimulus1=stimulus1, stimulus2=stimulus2))

            np.random.shuffle(sample_set) # shuffle the data and targets

            for i in range(num_samples): # assign the the shuffled data and targets
                dataset[i], target = sample_set[i]
                targets.append(target)

    if not random:
        for i in range(num_samples):
            dataset[i], target = generate_sample(i, input_size, seqlen1, seqlen2, seqlen3, test_type=test_type, stimulus1=stimulus1, stimulus2=stimulus2)
            targets.append(target)
    
    targets = torch.tensor(targets).to(device)
    
    return dataset, targets, sequence_length


#### TRAINING RELATED FUNCTIONS ####

# trains network
# generate_new -- generates new datasets and targets (of the same type of task) for the network to train on
# generate_random -- shuffles the generated target and dataset order so the network doesn't simply learn the order of predictions to make but the type
def train_network(network, dataset, targets, sequence_length, input_size, batch_size, epochs, optimizer, criterion, sheduler, generate_new=True, generate_random=True, condition=None, verbosity=2, cell=True, test_type="match_to_sample", multiple_tests=False, multiple_tests_type=None, stimulus_set=None, return_sample_losses=False):     
    mean_losses = []
    
    if return_sample_losses == True:
        sample_losses = []
            
    for epoch in range(1, epochs + 1):
        losses = [] # this is the loss for each sample set (i.e., the 4 different samples sets in the mts task and the 14 different sample sets in the drc task)
                
        # generate new distraction vectors during each epoch
        if generate_new and condition is not None:
                        
            seqlen1, seqlen2, seqlen3 = condition[0], condition[1], 0
            
            if test_type == "match_to_sample":
                seqlen3 = condition[2]
            
            
            # check to see if you want to train on multiple tests of the drc/mts task (i.e., changing what the input stimuli are)
            if (multiple_tests == True) and (multiple_tests_type is not None) and (stimulus_set is not None):
                
                if multiple_tests_type == "switch_stimuli":

                    # give vanilla test initially
                    if epoch % 2 != 0:
                        dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[0], stimulus2=stimulus_set[1])

                    # at the halfway point, give second test (switch first and second stimulus)
                    if epoch % 2 == 0:
                        dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[1], stimulus2=stimulus_set[0])

                if multiple_tests_type == "resue_one_stimulus":

                    # give vanilla test initially
                    if epoch % 2 != 0:
                        dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[0], stimulus2=stimulus_set[1])

                    # at the halfway point, give second test (switch first and second stimulus)
                    if epoch % 2 == 0:
                        dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[2], stimulus2=stimulus_set[1])
            else:
                dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type)
        
        # check to see if you want to train on multiple tests of the drc/mts task (i.e., changing what the input stimuli are)
        if (multiple_tests == True) and (multiple_tests_type is not None) and (stimulus_set is not None):
            
            seqlen1, seqlen2, seqlen3 = condition[0], condition[1], 0
            
            if test_type == "match_to_sample":
                seqlen3 = condition[2]
                
            if multiple_tests_type == "switch_stimuli":
                
                # give vanilla test initially
                if epoch % 2 != 0:
                    dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[0], stimulus2=stimulus_set[1])
                
                # at the halfway point, give second test (switch first and second stimulus)
                if epoch % 2 == 0:
                    dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[1], stimulus2=stimulus_set[0])
            
            if multiple_tests_type == "resue_one_stimulus":
                
                # give vanilla test initially
                if epoch % 2 != 0:
                    dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[0], stimulus2=stimulus_set[1])
                
                # at the halfway point, give second test (switch first and second stimulus)
                if epoch % 2 == 0:
                    dataset, targets, sequence_length = generate_dataset(input_size, seqlen1, seqlen2, seqlen3, random=generate_random, test_type=test_type, stimulus1=stimulus_set[2], stimulus2=stimulus_set[1])
                
        if str(criterion) == "BCELoss()":
            dataset = dataset.to(torch.float)
            targets = targets.to(torch.float)
                
        for sample_num, sample, target in zip(range(len(dataset)), dataset, targets):
                        
            optimizer.zero_grad() 
                        
            if cell == False: # for regular RNN using batch first
                sample = sample.view(batch_size, sequence_length, input_size)
            if cell == True: # for RNNCell using sequence length first
                sample = sample.view(sequence_length, batch_size, input_size)
            
            if "WMEMLSTMCellDRCNet" in str(type(network)):
                if multiple_tests == False:
                    network.cr.update_trial_tag(sample_num)
                else:
                    if epoch % 2 != 0:
                        network.cr.update_trial_tag(sample_num)
                    if epoch % 2 == 0:
                        network.cr.update_trial_tag(sample_num + len(dataset)) # update the trial tags to include every other sample
                
            # forward propagation
            output = network(sample)  # pass each sample into the network
            
            loss = criterion.forward(output, target)  # forward propagates loss
            losses.append(loss.detach().cpu())
            
            if return_sample_losses == True:
                sample_losses.append(loss.detach().cpu())
        
            # backpropagation (occurs per sample)
            loss.backward()

            # gradient descent/adam step
            optimizer.step()
            
        # reset the temporal drift once you've gone through all the trials
        if "WMEMLSTMCellDRCNet" in str(type(network)):
            if multiple_tests == False:
                # reset temporal drift after every sample
                network.cr.reset_temporal_drift()
            else:
                if epoch % 2 == 0:
                    # reset the temporal drift after every other set of samples (do factor in the second set of samples)
                    network.cr.reset_temporal_drift()
            
        mean_loss = sum(losses) / len(losses)  # calculates mean loss for epoch
        mean_losses.append(mean_loss)
        sheduler.step(mean_loss)
        
        # only the last sample of an epoch will be output but all samples will be trained
        if verbosity > 2:
            print(f'Cost at epoch {epoch} is {mean_loss}')
            print()
            print(f'Input = {sample.detach().reshape(sequence_length, input_size)}')
            print(f'Targets = {target.detach()}')
            print(f'Predictions = {output.detach()}')
            print()
            
        if verbosity > 0:
            if epoch in [1, round(epochs/3), round(epochs/2), round(2*epochs/3), epochs]:
                print(f'Cost at epoch {epoch} is {mean_loss}')
                if verbosity > 1:
                    
                    # verbosely note the halfway point
                    if (multiple_tests == True) and (epoch == epochs/2):
                        print()
                        print("Test switch using " + multiple_tests_type)
                    
                    print()
                    print(f'Input = {sample.detach().reshape(sequence_length, input_size)}')
                    print(f'Targets = {target.detach()}')
                    print(f'Predictions = {output.detach()}')
                    print()

    if return_sample_losses == True:
        return np.array(mean_losses), np.array(sample_losses)
   
    else:
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