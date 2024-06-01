import matplotlib.pyplot as plt

def plot_loss_vs_epoch(losses, epochs):
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, marker='o', color='r', linestyle='-', linewidth = 1, label='Loss per Epoch')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    losses = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.25, 0.2]
    epochs = list(range(1, len(losses) + 1))
    plot_loss_vs_epoch(losses, epochs)