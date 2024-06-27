"""
This module implements utilities for generating initial
and target data for training the lbm layers.
"""

import math

import torch

from src.simconfig import SimulationConfig
from src.lbm_layer import LbmLayer

from src.reference import LbmBGK

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

def macroscopic_from_weights(weights: torch.tensor):
    """
    Calculates macroscopic data (density, velocity)
    from the given distribution weights.
    """

    config = SimulationConfig(3,3,0.7)
    lbm = LbmBGK(config)

    return torch.stack([
        lbm.calc_densities(weights),
        lbm.calc_jx(weights),
        lbm.calc_jy(weights),
    ])

def weights_from_macroscopic(solution: torch.tensor):
    """
    Calculates weights from given macroscopic data (density, velocity),
    as an equilibrium distribution for those values.
    """

    config = SimulationConfig(3,3,0.7)
    lbm = LbmBGK(config)

    lbm.densities = solution[0,:,:]
    lbm.velocities_x = solution[1,:,:]
    lbm.velocities_y = solution[2,:,:]

    lbm.calculate_equilibrium()

    return lbm.eq_weights

def get_target(initial_data, config: SimulationConfig, num_steps: int = 1):
    """
    Returns initial data after evolving it a given number
    of steps with a reference solver.
    """

    ref_solver = LbmBGK(config)
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
    """
    Returns input and target for training lbm layers
    on the evolution of a gaussian packet of fluid.
    """

    initial_data = get_gaussian(config)
    target_w = get_target(initial_data, config, num_steps)

    target = macroscopic_from_weights(target_w)

    return (initial_data, target)

def batch_training_data(inputs, targets):
    """Repackages lists of inputs and corresponding targest as stacked tensors."""

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    return (inputs, targets)

def get_poiseuille_data(config: SimulationConfig):
    """
    Returns stationary homogeneous weigts as initial data
    and analytic Poiseuille flow solution as targets.
    Assumes that gravity is in the y-direction.
    """

    initial_data = get_stationary_weights(config)

    densties = torch.ones((config.grid_size_x, config.grid_size_y), dtype=torch.float)
    velocities_x = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float)

    def theoretical_velocity(i: float) -> float:
        nu = (lbm.tau - 0.5)/3.0
        #Channel half-width (-1 since boundaries are between nodes):
        a = (config.grid_size_y-1.0)/2.0
        x = i - a
        #v = g/(2 nu) (a^2 - x^2):
        return config.gravity[1] * (a*a - x*x)/(2.0*nu)

    profile = torch.tensor([
        theoretical_velocity(i) for i in range(0, config.grid_size_y)
    ]).reshape(1, config.grid_size_y)

    velocities_y = profile.repeat(config.grid_size_x, 1)

    solution = torch.stack([
        densties,
        velocities_x,
        velocities_y
    ])

    return initial_data, solution

def get_gaussian_batch(layer: LbmLayer, config: SimulationConfig, num_steps: int = 1):
    """
    Returns a batch of inputs/targets for training lbm layers
    on the evolution of a gaussian packet of fluid.
    """

    offset_arr = [(0.0, 0.0), (-0.5, 0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, -0.5)]
    width_arr = [3.0, 2.5, 2.0, 1.5, 1.0]

    inputs = []
    targets = []

    for offset, width in zip(offset_arr, width_arr):
        initial_data = get_gaussian(config, width, offset)

        target_w = get_target(initial_data, config, num_steps)
        target = macroscopic_from_weights(target_w)

        inputs.append(initial_data)
        targets.append(target)

    return batch_training_data(inputs, targets)

