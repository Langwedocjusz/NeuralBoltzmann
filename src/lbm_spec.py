"""
This module implements specializations of the abstract LBM class,
meant to be used inside neural network layers
"""

import torch

from src.simconfig import SimulationConfig
from src.torch_ref import Lbm

class NoEqLbmHermite(Lbm):
    """
    Specialization of the Lbm class, that implements transformation into
    Hermite moments, but doesn't directly perform equilibrium computation.
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


class NoEqLbmGramSchmidt(Lbm):
    """
    Specialization of the Lbm class, that implements transformation into
    Gram-Schmidt moments, but doesn't directly perform equilibrium computation.
    Meant to be used in a neural network layer.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

        #Transformation to Gram-Schmidt moments:

        rho = torch.tensor([ 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.float)
        jx  = torch.tensor([ 0, 1, 0,-1, 0, 1,-1,-1, 1], dtype=torch.float)
        jy  = torch.tensor([ 0, 0, 1, 0,-1, 1, 1,-1,-1], dtype=torch.float)
        e   = torch.tensor([-4,-1,-1,-1,-1, 2, 2, 2, 2], dtype=torch.float)
        eps = torch.tensor([ 4,-2,-2,-2,-2, 1, 1, 1, 1], dtype=torch.float)
        qx  = torch.tensor([ 0,-2, 0, 2, 0, 1,-1,-1, 1], dtype=torch.float)
        qy  = torch.tensor([ 0, 0,-2, 0, 2, 1, 1,-1,-1], dtype=torch.float)
        pxx = torch.tensor([ 0, 1,-1, 1,-1, 0, 0, 0, 0], dtype=torch.float)
        pxy = torch.tensor([ 0, 0, 0, 0, 0, 1,-1, 1,-1], dtype=torch.float)

        self.weights_to_moments = torch.stack(
            [rho, jx, jy, e, eps, qx, qy, pxx, pxy]
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