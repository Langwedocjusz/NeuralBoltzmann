"""This module implements NN layers that perform lbm-scheme computations."""

import torch
from torch import nn

from src.simconfig import SimulationConfig

from src.torch_ref import Lbm as BaseLbm

class NoEqLbmHermite(BaseLbm):
    """
    Specialization of the Lbm class, that implements transformation into
    hermite moments, but doesn't directly perform equilibrium computation.
    Meant to be used in a neural network layer.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        #Transformation to Hermite moments:

        rho = torch.tensor([ 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
        jx  = torch.tensor([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=torch.float)
        jy  = torch.tensor([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=torch.float)
        pxx = torch.tensor([-1, 2,-1, 2,-1, 2, 2, 2, 2], dtype=torch.float) / 3.0
        pyy = torch.tensor([-1,-1, 2,-1, 2, 2, 2, 2, 2], dtype=torch.float) / 3.0
        pxy = torch.tensor([ 0, 0, 0, 0, 0, 1,-1, 1,-1], dtype=torch.float)
        gmx = torch.tensor([ 0,-1, 0, 1, 0, 2,-2,-2, 2], dtype=torch.float) / 3.0
        gmy = torch.tensor([ 0, 0,-1, 0, 1, 2, 2,-2,-2], dtype=torch.float) / 3.0
        gm  = torch.tensor([ 1,-2,-2,-2,-2, 4, 4, 4, 4], dtype=torch.float) / 9.0

        self.weights_to_moments = torch.stack(
            [rho, jx, jy, pxx, pyy, pxy, gmx, gmy, gm]
        );

        self.moments_to_weights = torch.inverse(self.weights_to_moments)

    def calculate_equilibrium(self):
        """Does nothing by design."""
        pass

    def collision(self):
        """Performs collision in moment space at each node."""
        moments = torch.einsum("ab,ijb->ija", self.weights_to_moments, self.new_weights)

        new_moments = self.one_minus_tau_inverse * moments + self.tau_inverse * self.eq_weights

        self.weights = torch.einsum("ab,ijb->ija", self.moments_to_weights, new_moments)


class LBMLayer(nn.Module):
    """
    This class implements neural network layer that performs
    a given number of simulation steps of the lbm scheme on the input data.
    """

    def __init__(self, config: SimulationConfig, iterations: int = 1):
        super().__init__()

        #Trainable parameters
        self.weight = nn.Parameter(torch.randn(6))

        #The rest of the lbm infrastracture
        self.lbm = NoEqLbmHermite(config)

        self.iterations = iterations

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