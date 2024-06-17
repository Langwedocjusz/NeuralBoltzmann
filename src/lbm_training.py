"""This module implements training of the lbm-scheme based NN layers."""

import math

import torch
from torch import nn
#from torch.utils.data import TensorDataset, DataLoader

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig

from src.lbm_layer import LBMLayer
from src.torch_ref import LbmMomentH as RefLbm

from src import plotting

def train_generic(model, initial_data, target, config: LearningConfig):
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

def get_initial_data_gaussian(config: SimulationConfig):
    ref_solver = RefLbm(config)
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

def get_target(initial_data, config:SimulationConfig, num_steps: int = 1):
    ref_solver = RefLbm(config)
    ref_solver.weights = initial_data.clone()

    ref_solver.simulate(num_steps)

    return ref_solver.weights

def train(lconf: LearningConfig):
    grid_size = 15
    sim_steps = 3

    config = SimulationConfig(grid_size, grid_size, 0.5)

    initial_data = get_initial_data_gaussian(config)
    target = get_target(initial_data, config, sim_steps)

    plotting.ShowHeatmap(initial_data[:,:,0], "Initial data")
    plotting.ShowHeatmap(target[:,:,0], "Target")

    model = LBMLayer(config, sim_steps)

    train_generic(model, initial_data, target, lconf)

    print(model.weight.data)

def test_coherence():
    """Performs one step of simulation using reference solver and trainable model with parameters fixed at their target values"""
    config = SimulationConfig(3, 3, 0.5)

    ref_solver = RefLbm(config)
    ref_solver.init_stationary()
    ref_solver.weights[1,1] *= 2.0

    initial_data = ref_solver.weights.clone()

    print(initial_data)

    ref_solver.simulate(1)
    target = ref_solver.weights.clone()

    print(target)

    model = LBMLayer(config)
    model.weight.data = torch.ones(6)
    test_data = model(initial_data)

    print(test_data)

    print(test_data - target)