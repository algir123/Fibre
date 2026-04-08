import numpy as np
import matplotlib.pyplot as plt

def plot_File(datFilePath, labels):
    data = np.loadtxt(datFilePath)
    plt.figure(figsize=(10, 6))
    plt.plot(data[:, 0], data[:, 1])
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.show()


if __name__ == "__main__":
    datFilePath = 'data.dat'
    labels = ['Longueur d\'onde [nm]', 'Densité spectrale de puissance [dBm/nm]']
    plot_File(datFilePath, labels)