"""
This module implements utilities for generating initial
and target data for training the lbm layers.
"""

import math

import torch

from src.simconfig import SimulationConfig
from src.lbm_layer import LbmLayer
from src.lbm_layer import get_ref_lbm

def get_stationary_weights(config: SimulationConfig):
    """Returns uniform, stationary equilibrium weights."""
    shape = (config.grid_size_x, config.grid_size_y, 9)

    weights = torch.zeros(shape, dtype=torch.float)

    weights[:,:] = torch.tensor([
        4.0/9.0,
        1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
        1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
    ])

    return weights

def get_target(layer: LbmLayer, initial_data, config: SimulationConfig, num_steps: int = 1):
    """
    Returns initial data after evolving it a given number
    of steps with the reference solver.
    """

    ref_solver = get_ref_lbm(layer, config)
    ref_solver.weights = initial_data.clone()

    ref_solver.simulate(num_steps)

    return ref_solver.weights

def get_gaussian(config: SimulationConfig, width: float = 2.0, offset: (float, float) = (0.0, 0.0)):
    """
    Returns initial weights for lbm, density of which is a gaussian function and corresponding.
    """

    weights = get_stationary_weights(config)

    def gaussian(i: int, j: int) -> float:
        center_x = (1.0 + offset[0]) * config.grid_size_x/2.0
        center_y = (1.0 + offset[1]) * config.grid_size_y/2.0

        x = (i - center_x)/width
        y = (j - center_y)/width

        return 1.0 + math.exp(-(x*x + y*y))

    for i in range(config.grid_size_x):
        for j in range(config.grid_size_y):
            weights[i,j] *= gaussian(i,j)

    return weights

def get_gaussian_data(layer: LbmLayer, config: SimulationConfig, num_steps: int = 1):
    initial_data = get_gaussian(config)
    target = get_target(layer, initial_data, config, num_steps)

    return (initial_data, target)

def batch_training_data(inputs, targets):
    """Repackages lists of inputs and corresponding targest as stacked tensors."""

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return (inputs, targets)

def get_gaussian_batch(layer: LbmLayer, config: SimulationConfig, num_steps: int = 1):
    offset_arr = [(0.0, 0.0), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    width_arr = [3.0, 2.5, 2.0, 1.5, 1.0]

    inputs = []
    targets = []

    for offset, width in zip(offset_arr, width_arr):
        initial_data = get_gaussian(config, width, offset)

        inputs.append(initial_data)
        targets.append(get_target(layer, initial_data, config, num_steps))

    return batch_training_data(inputs, targets)

