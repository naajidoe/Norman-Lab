import numpy as np

# trains network
def train_network(network, dataset, targets, sequence_length, input_size, batch_size, epochs, optimizer, criterion, sheduler, verbose=True):     
    mean_losses = []
    for epoch in range(epochs):
        losses = []

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