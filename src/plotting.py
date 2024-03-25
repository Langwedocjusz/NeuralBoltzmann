import numpy as np
import matplotlib.pyplot as plt

def BasicHeatmap(arr: np.array, title: str):
    plt.imshow(arr, cmap='plasma', interpolation='nearest', origin='lower', vmin=0, vmax=1)
    plt.colorbar()
    plt.title(title)
    plt.show()

def VectorField(val_x: np.array, val_y: np.array, title: str):
    plt.quiver(val_x, val_y)
    plt.title(title)
    plt.show()