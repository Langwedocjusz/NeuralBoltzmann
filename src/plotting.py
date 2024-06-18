"""This module implements a bunch of helper functions for visualizing data with pyplot"""

import numpy as np
import matplotlib.pyplot as plt

def show_heatmap(arr: np.array, title: str, minimum = 0.0, maximum = 1.0):
    """Displays provided data as a heatmap."""

    plt.imshow(
        arr,
        cmap='plasma', interpolation='nearest', origin='lower',
        vmin=minimum, vmax=maximum
    )
    plt.colorbar()
    plt.title(title)
    plt.show()
    plt.clf()

def save_heatmap(arr: np.array, title: str, filepath: str, minimum = 0.0, maximum = 1.0):
    """Saves image of a heatmap generated from the provided data at a given filepath."""

    plt.imshow(
        arr,
        cmap='plasma', interpolation='nearest', origin='lower',
        vmin=minimum, vmax=maximum
    )
    plt.colorbar()
    plt.title(title)
    plt.savefig(filepath)
    plt.clf()

def show_vector_field(val_x: np.array, val_y: np.array, title: str):
    """Displays vector field created from the passed arrays (x and y components)."""

    plt.quiver(val_x, val_y)
    plt.title(title)
    plt.show()
    plt.clf()

def save_vector_field(val_x: np.array, val_y: np.array, title: str, filepath: str):
    """
    Plots the provided data (x and y components)
    as a vector field and saves it a given filepath.
    """

    plt.quiver(val_x, val_y)
    plt.title(title)
    plt.savefig(filepath)
    plt.clf()

def show_function_1d(val: np.array):
    """Displays the provided data as a function."""

    plt.plot(val)
    plt.show()
    plt.clf()

def show_functions_1d(arrays, names, title):
    """Displays each element of the provided collection as a function on the same plot."""

    for array in arrays:
        plt.plot(array)
    plt.legend(names)
    plt.title(title)
    plt.show()
    plt.clf()

def show_flowlines(val_x: np.array, val_y: np.array):
    """Displays flowlines of a vector field created from the inputs (x,y components)."""

    x = np.arange(np.size(val_x, 0))
    y = np.arange(np.size(val_x, 1))

    plt.streamplot(y, x, val_x, val_y)
    plt.axis('equal')

    plt.show()
    plt.clf()
