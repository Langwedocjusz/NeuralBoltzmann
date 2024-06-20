"""This module implements training of the lbm-scheme based NN layers."""

import math

import torch
from torch import nn

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig

from src.lbm_layer import LbmLayer
from src.lbm_layer import get_lbm_layer
from src.lbm_layer import get_ref_lbm

from src.lbm_data import get_target
from src.lbm_data import get_gaussian_data
from src.lbm_data import get_gaussian_batch

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

def train_gaussian(layer: LbmLayer, lconf: LearningConfig, html: bool = False):
    """
    Trains an LBM layer using a setup that evolves
    a gaussian packet of higher density fluid.
    """

    sim_steps: int = 3

    config = SimulationConfig(
        grid_size_x = 15,
        grid_size_y = 15,
        tau = 0.7
    )

    initial_data, target = get_gaussian_data(layer, config, sim_steps)

    plotting.show_heatmap(initial_data[:,:,0], "Initial data")
    plotting.show_heatmap(target[:,:,0], "Target")

    model = get_lbm_layer(layer, config, sim_steps)

    train_generic(model, initial_data, target, lconf)

    if html:
        model_to_html(model, "results.html")
    else:
        print_model(model)


def train_poiseuille(layer: LbmLayer, lconf: LearningConfig, html: bool = False):
    """
    Trains an LBM layer using a minimal setup that results
    with a flow of parabolic (Poiseuille) profile.
    """

    sim_steps: int = 20

    config = SimulationConfig(
        grid_size_x = 3,
        grid_size_y = 10,
        tau = 0.7,
        gravity = (0.0, 0.0001)
    )

    ref = get_ref_lbm(layer, config)
    ref.init_stationary()
    ref.solid_mask[:,0] = True
    ref.solid_mask[:,config.grid_size_y-1] = True

    initial_data = ref.weights.clone()

    ref.simulate(sim_steps)

    target = ref.weights.clone()

    target_v = ref.velocities_y[0,:]

    model = get_lbm_layer(layer, config, sim_steps)
    model.lbm.solid_mask[:,0] = True
    model.lbm.solid_mask[:,config.grid_size_y-1] = True

    train_generic(model, initial_data, target, lconf)

    result = model(initial_data)

    tmp = get_ref_lbm(layer, config)
    tmp.new_weights = result.detach()
    tmp.update_macroscopic()

    result_v = tmp.velocities_y[0,:]

    functions = [target_v, result_v]
    names = ['target', 'result']
    plotting.show_functions_1d(functions, names, 'velocity profile')

    if html:
        model_to_html(model, "results.html")
    else:
        print_model(model)


def train_gaussian_batch(layer: LbmLayer, lconf: LearningConfig, html: bool = False):

    sim_steps: int = 3

    config = SimulationConfig(
        grid_size_x = 15,
        grid_size_y = 15,
        tau = 0.7
    )

    model = get_lbm_layer(layer, config, sim_steps)

    inputs, targets = get_gaussian_batch(layer, config, sim_steps)

    for i in range(inputs.shape[0]):
        plotting.show_heatmap(inputs[i,:,:,0], "Initial data")
        plotting.show_heatmap(targets[i,:,:,0], "Target")

    train_generic(model, inputs, targets, lconf)

    if html:
        model_to_html(model, "results.html")
    else:
        print_model(model)