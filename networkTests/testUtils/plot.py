import matplotlib.pyplot as plt

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