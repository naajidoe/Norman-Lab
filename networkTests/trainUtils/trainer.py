import numpy as np
from taskUtils import generate as gen

# trains network
def train_network(network, dataset, targets, sequence_length, input_size, batch_size, epochs, optimizer, criterion, sheduler, generate_new=False, same_distractions=False, condition=None, verbose=True):     
    mean_losses = []
    for epoch in range(epochs):
        losses = []
        
        if generate_new and condition is not None:
            seqlen1, seqlen2, seqlen3 = condition[0], condition[1], condition[2]
            dataset, targets, sequence_length = gen.generate_dataset(same_distractions, input_size, seqlen1, seqlen2, seqlen3)

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