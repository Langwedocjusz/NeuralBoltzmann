"""This module implements training of the lbm-scheme based NN layers."""

from dataclasses import dataclass

import torch
from torch import nn
#from torch.utils.data import TensorDataset, DataLoader

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig

from src.lbm_layer import LBMLayer
from src.torch_ref import LbmMomentH as RefLbm

def train(lconf: LearningConfig):
    config = SimulationConfig(5, 5, 0.5)

    ref_solver = RefLbm(config)
    ref_solver.init_stationary()
    ref_solver.weights[2,2] *= 2.0

    initial_data = ref_solver.weights.clone()

    ref_solver.simulate(1)
    target = ref_solver.weights.clone()

    model = LBMLayer(config)
    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=lconf.learning_rate)

    for t in range(lconf.learning_epochs):
        prediction = model(initial_data)

        loss = criterion(prediction, target)

        if t % 100 == 0:
            print(f'Iteration: {t}, Loss: {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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