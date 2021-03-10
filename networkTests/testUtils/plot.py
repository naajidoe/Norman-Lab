import matplotlib.pyplot as plt

# Compares and plots the loss of four different networks (for testing)
def plot_four_losses(title, loss1, loss2, loss3, loss4):
    # Plot mean losses
    # plt.figure()
    plt.suptitle(title)
    plt.plot(loss1, color='red')
    plt.plot(loss2, color='blue')
    plt.plot(loss3, color='orange')
    plt.plot(loss4, color='green')
    plt.xlabel('epoch')
    plt.ylabel('mean loss')