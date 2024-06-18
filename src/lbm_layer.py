"""This module implements NN layers that perform lbm-scheme computations."""

import torch
from torch import nn

from src.simconfig import SimulationConfig

from src.lbm_spec import NoEqLbmHermite
from src.lbm_spec import NoEqLbmGramSchmidt


class LBMHermiteMinimalLayer(nn.Module):
    """
    This class implements neural network layer that performs
    a given number of simulation steps of the lbm scheme on the input data.
    Collision is done in Hermite moment space and the only trainable parameters
    are weights that directly multiply each equilibrium moment
    (so in principle they should all converge to 1 during training).
    """

    def __init__(self, config: SimulationConfig, iterations: int = 1):
        super().__init__()

        #Trainable parameters
        self.weight = nn.Parameter(torch.randn(6))

        #The rest of the lbm infrastracture
        self.lbm = NoEqLbmHermite(config)

        self.iterations = iterations

    def get_expected_weights(self):
        """Returns expected values of model parameters after training."""
        return torch.tensor([1,1,1,1,1,1], dtype=torch.float)

    def calculate_equilibrium(self):
        """Performs calculation of the equilibrium moments using autograd-enabled weights"""
        (gx, gy) = self.lbm.gravity

        ux = self.lbm.velocities_x + self.lbm.tau * gx * self.lbm.densities
        uy = self.lbm.velocities_y + self.lbm.tau * gy * self.lbm.densities

        rho = self.lbm.densities
        jx = self.lbm.densities * ux
        jy = self.lbm.densities * uy

        pxx = self.weight[0] * rho * ux * ux
        pyy = self.weight[1] * rho * uy * uy
        pxy = self.weight[2] * rho * ux * uy

        gmx = self.weight[3] * rho * ux * uy * uy
        gmy = self.weight[4] * rho * ux * ux * uy
        gm  = self.weight[5] * rho * ux * ux * uy * uy

        self.lbm.eq_weights = torch.stack(
            [rho, jx, jy, pxx, pyy, pxy, gmx, gmy, gm], dim=2
        )

    def forward(self, input):
        """Performs a set number of simulation steps using lbm scheme on input."""
        self.lbm.weights = input.clone()

        for _ in range(self.iterations):
            self.lbm.handle_boundaries()
            self.lbm.streaming()
            self.lbm.update_macroscopic()

            #Equilibrium is computed outside of the lbm:
            self.calculate_equilibrium()

            self.lbm.collision()

        return self.lbm.weights

class LBMGramSchmidtLayer(nn.Module):
    """
    This class implements neural network layer that performs
    a given number of simulation steps of the lbm scheme on the input data.
    Collision is done in Gram-Schmidt moment space and the only trainable parameters
    are weights, inserted in the equilibrium moments computation in a way that
    respects the symmetry of the problem.
    """

    def __init__(self, config: SimulationConfig, iterations: int = 1):
        super().__init__()

        #Trainable parameters
        self.weight = nn.Parameter(torch.randn(7))

        #The rest of the lbm infrastracture
        self.lbm = NoEqLbmGramSchmidt(config)

        self.iterations = iterations

    def get_expected_weights(self):
        """Returns expected values of model parameters after training."""
        return torch.tensor([-2, 3, 1, -3, -1, 1/3, 1/3], dtype=torch.float)

    def calculate_equilibrium(self):
        """Performs calculation of the equilibrium moments using autograd-enabled weights"""
        (gx, gy) = self.lbm.gravity

        ux = self.lbm.velocities_x + self.lbm.tau * gx * self.lbm.densities
        uy = self.lbm.velocities_y + self.lbm.tau * gy * self.lbm.densities

        rho = self.lbm.densities
        jx = self.lbm.densities * ux
        jy = self.lbm.densities * uy

        e   = self.weight[0]*rho + self.weight[1]*(jx*jx + jy*jy)
        eps = self.weight[2]*rho - self.weight[3]*(jx*jx + jy*jy)
        qx  = self.weight[4] * jx
        qy  = self.weight[4] * jy
        pxx = self.weight[5] * (jx*jx - jy*jy)
        pxy = self.weight[6] * jx*jy

        self.lbm.eq_weights = torch.stack(
            [rho, jx, jy, e, eps, qx, qy, pxx, pxy], dim=2
        )

    def forward(self, input):
        """Performs a set number of simulation steps using lbm scheme on input."""
        self.lbm.weights = input.clone()

        for _ in range(self.iterations):
            self.lbm.handle_boundaries()
            self.lbm.streaming()
            self.lbm.update_macroscopic()

            #Equilibrium is computed outside of the lbm:
            self.calculate_equilibrium()

            self.lbm.collision()

        return self.lbm.weights
