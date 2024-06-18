"""This module implements training of the lbm-scheme based NN layers."""

import math

import torch
from torch import nn
#from torch.utils.data import TensorDataset, DataLoader

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig

#from src.torch_ref import LbmMomentH as RefLbm
#from src.lbm_layer import LBMHermiteMinimalLayer as LbmLayer

from src.torch_ref import LbmMomentGS as RefLbm
from src.lbm_layer import LBMGramSchmidtLayer as LbmLayer

from src import plotting

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

def train_gaussian(lconf: LearningConfig):
    grid_size: int = 15
    sim_steps: int = 3

    config = SimulationConfig(grid_size, grid_size, 0.5)

    initial_data = get_initial_data_gaussian(config)
    target = get_target(initial_data, config, sim_steps)

    plotting.ShowHeatmap(initial_data[:,:,0], "Initial data")
    plotting.ShowHeatmap(target[:,:,0], "Target")

    model = LbmLayer(config, sim_steps)

    train_generic(model, initial_data, target, lconf)

    expected = model.get_expected_weights()

    print(f"Expected: {expected}")
    print(f"Obtained: {model.weight.data}")

def train_poiseuille(lconf: LearningConfig):
    sim_steps: int = 20

    size_x: int = 3
    size_y: int = 10
    g: float = 0.0001
    tau: float = 1.0

    config = SimulationConfig(size_x, size_y, tau, (0.0, g))

    ref = RefLbm(config)
    ref.init_stationary()
    ref.solid_mask[:,0] = True
    ref.solid_mask[:,size_y-1] = True

    initial_data = ref.weights.clone()

    ref.simulate(sim_steps)

    target = ref.weights.clone()

    target_v = ref.velocities_y[0,:]

    model = LbmLayer(config, sim_steps)
    model.lbm.solid_mask[:,0] = True
    model.lbm.solid_mask[:,size_y-1] = True

    train_generic(model, initial_data, target, lconf)

    result = model(initial_data)

    tmp = RefLbm(config)
    tmp.new_weights = result.detach()
    tmp.update_macroscopic()

    result_v = tmp.velocities_y[0,:]

    functions = [target_v, result_v]
    names = ['target', 'result']
    plotting.ShowFunctions1d(functions, names, 'velocity profile')

    expected = model.get_expected_weights()

    print(f"Expected: {expected}")
    print(f"Obtained: {model.weight.data}")