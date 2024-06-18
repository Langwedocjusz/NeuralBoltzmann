"""
This module implements training of simple neural networks
to recover numeric kernel of the diffusion equation.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from src import plotting
from src.learning_config import LearningConfig

from mock.classical_solver import evolve_to_convergence

# Simulation domain:
GRID_SIZE = 10
# Classical solver (target) parameters:
DT = 0.5
CLASSIC_EPSILON = 0.001
# Neural networks inner iteration count:
SIMULATION_STEPS = 60


class IteratedLinearNet(nn.Module):
    """Simple neural network that applies one linear (dense) layer multiple times."""

    def __init__(self, grid_size: int):
        super().__init__()

        self.lin = nn.Linear(grid_size, grid_size, bias=False)

    def forward(self, input):
        x = self.lin(input)

        for _ in range(SIMULATION_STEPS - 1):
            x = self.lin(x)

        return x


class CustomConvoLayer(nn.Module):
    """
    Custom layer that computes convolution of input with weights,
    while keeping boundary values fixed
    """

    def __init__(self, input_size: int):
        super().__init__()
        # Convolution weights (trainable)
        self.weight = nn.Parameter(torch.randn(3))
        # Input mask (constant)
        self.mask = torch.ones(
            input_size, dtype=torch.bool, requires_grad=False
        )
        self.mask[0] = False
        self.mask[-1] = False
        self.mask = self.mask.unsqueeze(0)

    def forward(self, input):
        res = input.clone()

        batched: bool = input.dim() == 3

        if batched:
            batch_size = input.shape[0]

            patches = input.unfold(2, 3, 1)
            patches = torch.matmul(patches, self.weight)

            m = self.mask.unsqueeze(0).repeat(batch_size, 1, 1)

            res[m] = patches.flatten()

        else:
            patches = input.unfold(1, 3, 1)
            patches = torch.matmul(patches, self.weight)

            res[self.mask] = patches

        return res


class CustomIteratedConvoNet(nn.Module):
    """Simple neural network that applies one CustomConvoLayer multiple times."""

    def __init__(self, input_size: int):
        super().__init__()

        self.conv = CustomConvoLayer(input_size)

    def forward(self, input):
        x = self.conv(input)

        for _ in range(SIMULATION_STEPS - 1):
            x = self.conv(x)

        return x


def point_impulses():
    """
    Returns a pair (inputs, targets).
    Inputs are functions which take value 1 at a particular point and 0 everywhere else.
    Targets are their converged time evolutions, obtained using finite difference
    solver for the diffusion equation.
    """

    inputs = []
    targets = []

    for i in range(GRID_SIZE):
        initial_data = torch.zeros(GRID_SIZE, dtype=torch.float)
        initial_data[i] = 1.0
        ref_result = evolve_to_convergence(initial_data, DT, CLASSIC_EPSILON)

        inputs.append(initial_data.unsqueeze(0))
        targets.append(ref_result.unsqueeze(0))

    return (inputs, targets)


def triangle_impulses():
    """
    Returns a pair (inputs, targets).
    Inputs are traingle functions with different widths and heights,
    centered around different points. Targets are their converged time evolutions,
    obtained using finite difference solver for the diffusion equation.
    """

    inputs = []
    targets = []

    widths = torch.linspace(
        1.0, GRID_SIZE/2, int(GRID_SIZE/2), dtype=torch.float
    )

    for center in range(GRID_SIZE):
        for width in widths:
            initial_data = torch.linspace(
                0.0, GRID_SIZE, GRID_SIZE, dtype=torch.float
            )

            initial_data.apply_(lambda x: max(
                0.0, 1.0 - abs(x - center)/width)
            )

            ref_result = evolve_to_convergence(
                initial_data, DT, CLASSIC_EPSILON)

            inputs.append(initial_data.unsqueeze(0))
            targets.append(ref_result.unsqueeze(0))

    return (inputs, targets)


def get_training_dataloader(generating_fn):
    """Repackages lists of inputs and corresponding targest as a torch Dataloader object"""

    inputs, targets = generating_fn()

    data_count = len(inputs)

    inputs = torch.stack(inputs)
    targets = torch.stack(targets)

    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=data_count)

    return dataloader


def train_linear(config: LearningConfig):
    """Example of training the IteratedLinearNet to solve the diffusion equation"""

    model = IteratedLinearNet(GRID_SIZE)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataloader = get_training_dataloader(triangle_impulses)

    for t in range(config.learning_epochs):
        for _batch, (initial_data, target) in enumerate(dataloader):
            prediction = model(initial_data)

            loss = criterion(prediction, target)

            if t % 100 == 0:
                print(f'Iteration: {t}, Loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_case = torch.linspace(0.0, 1.0, GRID_SIZE)
    test_case.apply_(lambda x: 0.5 + 0.5*math.sin(5.0*x))

    test_target = evolve_to_convergence(test_case, DT, CLASSIC_EPSILON).numpy()
    test_result = model(test_case).detach().numpy()

    plotting.show_functions_1d(
        [test_case, test_target, test_result],
        ["initial data", "ground truth", "result"],
        "final result"
    )

    print(model.lin.weight)
    print(model.lin.bias)


def train_custom(config: LearningConfig):
    """Example of training the CustomIteratedConvoNet to solve the diffusion equation"""

    model = CustomIteratedConvoNet(GRID_SIZE)
    model.conv.weight.data = 0.1*torch.randn(3)

    criterion = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    dataloader = get_training_dataloader(triangle_impulses)

    for t in range(config.learning_epochs):
        for _batch, (initial_data, target) in enumerate(dataloader):
            prediction = model(initial_data)

            loss = criterion(prediction, target)

            if t % 100 == 0:
                print(f'Iteration: {t}, Loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    test_case = torch.linspace(0.0, 1.0, GRID_SIZE)
    test_case.apply_(lambda x: 0.5 + 0.5*math.sin(5.0*x))

    test_target = evolve_to_convergence(test_case, DT, CLASSIC_EPSILON).numpy()

    test_result = model(test_case.unsqueeze(0)).detach()[0].numpy()

    plotting.show_functions_1d(
        [test_case.detach(), test_target, test_result],
        ["initial data", "ground truth", "result"],
        "final result"
    )

    print(model.conv.weight)
