import numpy as np
import matplotlib.pyplot as plt

def ShowHeatmap(arr: np.array, title: str, minimum = 0.0, maximum = 1.0):
    plt.imshow(arr, cmap='plasma', interpolation='nearest', origin='lower', vmin=minimum, vmax=maximum)
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.clf()

def SaveHeatmap(arr: np.array, title: str, filepath: str, minimum = 0.0, maximum = 1.0):
    plt.imshow(arr, cmap='plasma', interpolation='nearest', origin='lower', vmin=minimum, vmax=maximum)
    plt.colorbar()
    plt.title(title)
    plt.savefig(filepath)
    plt.clf()

def ShowVectorField(val_x: np.array, val_y: np.array, title: str):
    plt.quiver(val_x, val_y)
    plt.title(title)
    plt.show()
    plt.clf()