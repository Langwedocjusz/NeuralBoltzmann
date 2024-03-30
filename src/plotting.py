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

def SaveVectorField(val_x: np.array, val_y: np.array, title: str, filepath: str):
    plt.quiver(val_x, val_y)
    plt.title(title)
    plt.savefig(filepath)
    plt.clf()

def ShowFunction1d(val: np.array):
    plt.plot(val)
    plt.show()
    plt.clf()

def ShowFunctions1d(arrays, names, title):
    for array in arrays:
        plt.plot(array)
    plt.legend(names)
    plt.title(title)
    plt.show()
    plt.clf()

def ShowFlowLines(val_x: np.array, val_y: np.array):
    x = np.arange(np.size(val_x, 0))
    y = np.arange(np.size(val_x, 1))

    #print(val_x.shape)
    #print(x.shape)
    #print(y.shape)

    plt.streamplot(y, x, val_x, val_y)

    plt.show()
    plt.clf()