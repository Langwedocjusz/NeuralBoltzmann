"""This module implements training of the lbm-scheme based NN layers."""

import math

import torch
from torch import nn
#from torch.utils.data import TensorDataset, DataLoader

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig

from src.lbm_layer import LbmLayer
from src.lbm_layer import get_lbm_layer
from src.lbm_layer import get_ref_lbm

from src import plotting

from src.lbm_print import print_model
from src.lbm_print import model_to_html

def train_generic(model, initial_data, target, config: LearningConfig):
    """
    Small utility that takes given model, initial data and target
    and performs training using the supplied parameters.
    """

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    for t in range(config.learning_epochs):
        prediction = model(initial_data)

        loss = criterion(prediction, target)

        if t % 100 == 0:
            print(f'Iteration: {t}, Loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def get_initial_data_gaussian(layer: LbmLayer, config: SimulationConfig):
    """
    Returns initial weights for lbm, density of which is a gaussian function.
    """

    ref_solver = get_ref_lbm(layer, config)
    ref_solver.init_stationary()

    def gaussian(i: int, j: int) -> float:
        width = 2.0

        x = (i-config.grid_size_x/2.0)/width
        y = (j-config.grid_size_y/2.0)/width

        return 1.0 + math.exp(-(x*x + y*y))

    for i in range(config.grid_size_x):
        for j in range(config.grid_size_y):
            ref_solver.weights[i,j] *= gaussian(i,j)

    return ref_solver.weights

def get_target(layer: LbmLayer, initial_data, config:SimulationConfig, num_steps: int = 1):
    """
    Returns initial data after evolving it a given number
    of steps with the reference solver.
    """

    ref_solver = get_ref_lbm(layer, config)
    ref_solver.weights = initial_data.clone()

    ref_solver.simulate(num_steps)

    return ref_solver.weights

def train_gaussian(layer: LbmLayer, lconf: LearningConfig, html: bool = False):
    """
    Trains an LBM layer using a setup that evolves
    a gaussian packet of higher density fluid.
    """

    grid_size: int = 15
    sim_steps: int = 3

    config = SimulationConfig(grid_size, grid_size, 0.5)

    initial_data = get_initial_data_gaussian(layer, config)
    target = get_target(layer, initial_data, config, sim_steps)

    plotting.show_heatmap(initial_data[:,:,0], "Initial data")
    plotting.show_heatmap(target[:,:,0], "Target")

    model = get_lbm_layer(layer, config, sim_steps)

    train_generic(model, initial_data, target, lconf)

    expected = model.get_expected_weights()

    if html:
        model_to_html(model, "test.html")
    else:
        print_model(model)


def train_poiseuille(layer: LbmLayer, lconf: LearningConfig, html: bool = False):
    """
    Trains an LBM layer using a minimal setup that results
    with a flow of parabolic (Poiseuille) profile.
    """

    sim_steps: int = 20

    size_x: int = 3
    size_y: int = 10
    g: float = 0.0001
    tau: float = 1.0

    config = SimulationConfig(size_x, size_y, tau, (0.0, g))

    ref = get_ref_lbm(layer, config)
    ref.init_stationary()
    ref.solid_mask[:,0] = True
    ref.solid_mask[:,size_y-1] = True

    initial_data = ref.weights.clone()

    ref.simulate(sim_steps)

    target = ref.weights.clone()

    target_v = ref.velocities_y[0,:]

    model = get_lbm_layer(layer, config, sim_steps)
    model.lbm.solid_mask[:,0] = True
    model.lbm.solid_mask[:,size_y-1] = True

    train_generic(model, initial_data, target, lconf)

    result = model(initial_data)

    tmp = get_ref_lbm(layer, config)
    tmp.new_weights = result.detach()
    tmp.update_macroscopic()

    result_v = tmp.velocities_y[0,:]

    functions = [target_v, result_v]
    names = ['target', 'result']
    plotting.show_functions_1d(functions, names, 'velocity profile')

    expected = model.get_expected_weights()

    if html:
        model_to_html(model, "test.html")
    else:
        print_model(model)
