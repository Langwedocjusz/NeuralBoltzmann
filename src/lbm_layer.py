"""This module implements NN layers that perform lbm-scheme computations."""

import math
from enum import Enum

import torch
from torch import nn

from src.simconfig import SimulationConfig

from src.lbm_spec import NoEqLbmHermite
from src.lbm_spec import NoEqLbmGramSchmidt
from src.lbm_spec import LbmEmpty

from src.torch_ref import LbmMomentH
from src.torch_ref import LbmMomentGS
from src.torch_ref import LbmBGK

class LbmLayer(Enum):
    """Enum representing supported types of lbm layers."""
    MINIMAL_HERMITE = 0
    GRAM_SCHMIDT = 1
    BGK = 2

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
        weight =  torch.tensor([1,1,1,1,1,1], dtype=torch.float)
        return [("weight", weight)]

    def get_current_weights(self):
        """Returns current values of model parameters."""
        return [("weight", self.weight.data)]

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
        weight = torch.tensor([-2, 3, 1, -3, -1, 1/3, 1/3], dtype=torch.float)
        return [("weight", weight)]

    def get_current_weights(self):
        """Returns current values of model parameters."""
        return [("weight", self.weight.data)]

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

class LbmBGKLayer(nn.Module):
    """
    This class implements neural network layer that performs
    a given number of simulation steps of the lbm scheme on the input data.
    Collision is done in the weights space as with the BGK operator.
    """

    def __init__(self, config: SimulationConfig, iterations: int = 1):
        super().__init__()

        #Trainable parameters
        self.M0 = nn.Parameter(torch.randn(9,9))

        #The rest of lbm infrastructure
        self.lbm = LbmEmpty(config)

        e = torch.stack((self.lbm.base_velocities_x, self.lbm.base_velocities_y))
        A = torch.matmul(e.t(), e)

        w = torch.tensor(
            [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
        )

        diag_w = torch.diag(w)

        self.D = math.sqrt(4.5/self.lbm.tau) * torch.matmul(A, torch.sqrt(diag_w))
        self.phi0 = (1.0/self.lbm.tau) * w
        self.Q = (-1.5/self.lbm.tau) * A

        self.iterations = iterations

        #Expected values of trainable parameters
        self.refM0 = M0 = self.lbm.one_minus_tau_inverse * torch.eye(9) + 3.0 * self.lbm.tau_inverse * torch.matmul(A, diag_w)

    def get_expected_weights(self):
        """Returns expected values of model parameters after training."""
        res = []

        res.append(("M0", self.refM0.data))

        return res

    def get_current_weights(self):
        """Returns current values of model parameters."""
        res = []

        res.append(("M0", self.M0.data))

        return res

    def collision(self):
        """Performs equilibrium computation and collision in one go."""

        M0 = torch.matmul(self.lbm.new_weights, self.M0)

        D = torch.square(torch.matmul(self.lbm.new_weights, self.D))
        D = D/self.lbm.densities.reshape(self.lbm.shape2d)

        phi0 = self.lbm.densities.reshape(self.lbm.shape2d) * self.phi0

        Q  = (torch.einsum("ijk,ijk->ij",
            (self.lbm.new_weights, torch.matmul(self.lbm.new_weights, self.Q))
            ))/self.lbm.densities

        self.lbm.weights = M0 + D + phi0 + Q.reshape(self.lbm.shape2d) * self.lbm.eq_factors

    def forward(self, input):
        """Performs a set number of simulation steps using lbm scheme on input."""
        self.lbm.weights = input.clone()

        for _ in range(self.iterations):
            self.lbm.handle_boundaries()
            self.lbm.streaming()
            self.lbm.update_macroscopic()

            #Collision is done outside of the lbm:
            self.collision()

        return self.lbm.weights


def get_lbm_layer(e : LbmLayer, config: SimulationConfig, iterations: int):
    """Factory function for creating an instance of a chosen lbm layer."""

    if e == LbmLayer.MINIMAL_HERMITE:
        return LBMHermiteMinimalLayer(config, iterations)
    elif e == LbmLayer.GRAM_SCHMIDT:
        return LBMGramSchmidtLayer(config, iterations)
    elif e == LbmLayer.BGK:
        return LbmBGKLayer(config, iterations)
    else:
        raise RuntimeError("Invalid LbmLayer type provided.")

def get_ref_lbm(e : LbmLayer, config: SimulationConfig):
    """Factory function for creating an instance of a chosen reference lbm."""

    if e == LbmLayer.MINIMAL_HERMITE:
        return LbmMomentH(config)
    elif e == LbmLayer.GRAM_SCHMIDT:
        return LbmMomentGS(config)
    elif e == LbmLayer.BGK:
        return LbmBGK(config)
    else:
        raise RuntimeError("Invalid LbmLayer type provided.")