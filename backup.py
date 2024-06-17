from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

from src.simconfig import BC
from src.simconfig import SimulationConfig

from src.torch_ref import Lbm as BaseLbm
from src.torch_ref import LbmMomentGS as RefLbm

class LayerLbm(Lbm):
    """
    Specialization of the Lbm class, that doesn't directly perform
    equilibrium computation, to be used in a neural network layer.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__(config)

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
        pass

    def collision(self):
        """Performs collision in moment space at each node."""
        moments = torch.einsum("ab,ijb->ija", self.weights_to_moments, self.new_weights)

        new_moments = self.one_minus_tau_inverse * moments + self.tau_inverse * self.eq_weights

        self.weights = torch.einsum("ab,ijb->ija", self.moments_to_weights, new_moments)


class LBMLayer(nn.Module):
    """
    This class implements neural network layer that performs
    one simulation step of the lbm scheme on the input data.
    """

    def __init__(self, config: SimulationConfig):
        super().__init__()

        #Trainable parameters
        self.weight = nn.Parameter(torch.randn(6))

        #Usual LBM initialization (doesn't use autograd):

        self.tau: float = config.tau
        self.tau_inverse: float = 1.0/config.tau
        self.one_minus_tau_inverse: float = 1.0 - self.tau_inverse

        self.grid_size_x: int = config.grid_size_x
        self.grid_size_y: int = config.grid_size_y

        self.gravity: (float, float) = config.gravity

        #Lattice initialization
        self.shape   = (config.grid_size_x, config.grid_size_y, 9)
        self.shape2d = (config.grid_size_x, config.grid_size_y, 1)
        self.shape1d = (1, 1, 9)

        self.weights     = torch.zeros(self.shape, dtype=torch.float, requires_grad=False)
        self.new_weights = torch.zeros(self.shape, dtype=torch.float, requires_grad=False)
        self.eq_weights  = torch.zeros(self.shape, dtype=torch.float, requires_grad=False)

        self.eq_factors = torch.tensor([
            4.0/9.0,
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0,
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
        ], dtype=torch.float, requires_grad=False).reshape(self.shape1d)

        self.densities    = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float, requires_grad=False)
        self.velocities_x = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float, requires_grad=False)
        self.velocities_y = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float, requires_grad=False)

        self.boundary_conditions = config.boundary_conditions

        self.boundary_velocities = [
            torch.zeros(config.grid_size_y, dtype=torch.float, requires_grad=False),
            torch.zeros(config.grid_size_x, dtype=torch.float, requires_grad=False),
            torch.zeros(config.grid_size_y, dtype=torch.float, requires_grad=False),
            torch.zeros(config.grid_size_x, dtype=torch.float, requires_grad=False),
        ]

        #Weight ids convention:
        #  6 -- 2 -- 5
        #  |    |    |
        #  3 -- 0 -- 1
        #  |    |    |
        #  7 -- 4 -- 8

        #Left moving population at a given node was one step ago
        #at a node one unit to the right (since it's left moving)
        #so in the streaming step we need to index with opposite indices:

        #Doing this with numpy atm, since pure torch seems to be way more laborious here
        rows, cols, tubes = np.indices(self.shape)
        self.indices = torch.tensor(np.array((rows, cols, tubes)).T, requires_grad=False)

        #Left, right
        self.indices[1,:,:] = torch.roll(self.indices[1,:,:],  1, 0)
        self.indices[3,:,:] = torch.roll(self.indices[3,:,:], -1, 0)

        #Up, down
        self.indices[2,:,:] = torch.roll(self.indices[2,:,:],  1, 1)
        self.indices[4,:,:] = torch.roll(self.indices[4,:,:], -1, 1)

        #Upper corners
        self.indices[5,:,:] = torch.roll(self.indices[5,:,:],  1, 1)
        self.indices[5,:,:] = torch.roll(self.indices[5,:,:],  1, 0)

        self.indices[6,:,:] = torch.roll(self.indices[6,:,:],  1, 1)
        self.indices[6,:,:] = torch.roll(self.indices[6,:,:], -1, 0)

        #Lower corners
        self.indices[7,:,:] = torch.roll(self.indices[7,:,:], -1, 1)
        self.indices[7,:,:] = torch.roll(self.indices[7,:,:], -1, 0)

        self.indices[8,:,:] = torch.roll(self.indices[8,:,:], -1, 1)
        self.indices[8,:,:] = torch.roll(self.indices[8,:,:],  1, 0)

        #Base velocities
        diag = 1.0

        self.base_velocities_x = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, diag, -diag, -diag, diag], requires_grad=False)
        self.base_velocities_y = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, diag, diag, -diag, -diag], requires_grad=False)

        #Solid mask
        self.solid_mask = torch.zeros(self.shape2d, dtype=torch.bool)

        #Bounceback indices
        self.default_ids = [0,1,2,3,4,5,6,7,8]
        self.swapped_ids = [0,2,1,4,3,7,8,5,6]

        #Disable flow at non-periodic boundaries
        if self.boundary_conditions[0] == BC.VON_NEUMANN:
            for i in range(0, self.grid_size_y):
                for a in range(0,9):
                    self.indices[a,i,0] = torch.tensor([0,i,a])

        if self.boundary_conditions[1] == BC.VON_NEUMANN:
            for i in range(0, self.grid_size_x):
                for a in range(0,9):
                    self.indices[a,0,i] = torch.tensor([i,0,a])

        if self.boundary_conditions[0] == BC.VON_NEUMANN:
            lx = self.grid_size_x - 1

            for i in range(0, self.grid_size_y):
                for a in range(0,9):
                    self.indices[a,i,lx] = torch.tensor([lx,i,a])

        if self.boundary_conditions[3] == BC.VON_NEUMANN:
            ly = self.grid_size_y - 1

            for i in range(0, self.grid_size_x):
                for a in range(0,9):
                    self.indices[a,ly,i] = torch.tensor([i,ly,a])

        #Initialization of moment transformation matrices:
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

    def update_macroscopic(self):
        """(Re)Calculates densities and velocities from new weights."""

        self.densities    = torch.sum(self.new_weights, axis=2)

        self.velocities_x = torch.einsum("ijk,k->ij", (self.new_weights, self.base_velocities_x))
        self.velocities_x = self.velocities_x / self.densities

        self.velocities_y = torch.einsum("ijk,k->ij", (self.new_weights, self.base_velocities_y))
        self.velocities_y = self.velocities_y / self.densities

    def streaming(self):
        """Calculates new weights (after streaming step) from old ones."""
        stream_weights = self.weights[*self.indices.permute(3,2,1,0)]

        bounced_weights = torch.zeros(self.shape, dtype=torch.float)
        bounced_weights[:,:,self.default_ids] = self.weights[:,:,self.swapped_ids]

        self.new_weights = (~self.solid_mask) * stream_weights + self.solid_mask * bounced_weights

    def handle_boundary(self, edge_id: int):
        """
        Enforces Zou He flux boundary condition with perscribed
        von Neumann velocity at edge given by edge_id
        """
        horizontal = (edge_id % 2 == 0)

        lx = self.grid_size_x - 1
        ly = self.grid_size_y - 1

        pos_list = [0, 0, lx, ly]
        ingoing_ids_list  = [(4,7,8), (3,7,6), (2,5,6), (1,5,8)]
        outgoint_ids_list = [(2,5,6), (1,5,8), (4,7,8), (3,7,6)]

        v = self.boundary_velocities[edge_id]
        pos = pos_list[edge_id]
        in_ids = ingoing_ids_list[edge_id]
        out_ids = outgoint_ids_list[edge_id]

        mid_ids = (1, 3) if horizontal else (2,4)

        if horizontal:
            rho_a = self.weights[pos,:,0] + self.weights[pos,:,mid_ids[0]] + self.weights[pos,:,mid_ids[1]]
            rho_b = self.weights[pos,:,out_ids[0]] + self.weights[pos,:,out_ids[1]] + self.weights[pos,:,out_ids[2]]

            rho =  rho_a + 2.0 * rho_b
            rho = rho/(1.0 + v)

            ru = v * rho

            self.weights[pos,:,in_ids[0]] = self.weights[pos,:,out_ids[0]] - (2.0/3.0)*ru
            self.weights[pos,:,in_ids[1]] = self.weights[pos,:,out_ids[1]] - (1.0/6.0)*ru + 0.5*(self.weights[pos,:,mid_ids[0]] - self.weights[pos,:,mid_ids[0]])
            self.weights[pos,:,in_ids[2]] = self.weights[pos,:,out_ids[2]] - (1.0/6.0)*ru + 0.5*(self.weights[pos,:,mid_ids[1]] - self.weights[pos,:,mid_ids[1]])
        else:
            rho_a = self.weights[:,pos,0] + self.weights[:,pos,mid_ids[0]] + self.weights[:,pos,mid_ids[1]]
            rho_b = self.weights[:,pos,out_ids[0]] + self.weights[:,pos,out_ids[1]] + self.weights[:,pos,out_ids[2]]

            rho = rho_a + 2.0 * rho_b
            rho = rho/(1.0 + v)

            ru = v * rho

            self.weights[:,pos,in_ids[0]] = self.weights[:,pos,out_ids[0]] - (2.0/3.0)*ru
            self.weights[:,pos,in_ids[1]] = self.weights[:,pos,out_ids[1]] - (1.0/6.0)*ru + 0.5*(self.weights[:,pos,mid_ids[0]] - self.weights[:,pos,mid_ids[1]])
            self.weights[:,pos,in_ids[2]] = self.weights[:,pos,out_ids[2]] - (1.0/6.0)*ru + 0.5*(self.weights[:,pos,mid_ids[1]] - self.weights[:,pos,mid_ids[0]])

    def calculate_equilibrium(self):
        """Calculates equilibrium distributions of moments for all nodes."""
        (gx, gy) = self.gravity

        ux = self.velocities_x + self.tau * gx * self.densities
        uy = self.velocities_y + self.tau * gy * self.densities

        rho = self.densities
        jx = self.densities * ux
        jy = self.densities * uy

        pxx = self.weight[0] * rho * ux * ux
        pyy = self.weight[1] * rho * uy * uy
        pxy = self.weight[2] * rho * ux * uy

        gmx = self.weight[2] * rho * ux * uy * uy
        gmy = self.weight[3] * rho * ux * ux * uy
        gm  = self.weight[4] * rho * ux * ux * uy * uy

        self.eq_weights = torch.stack(
            [rho, jx, jy, pxx, pyy, pxy, gmx, gmy, gm], dim=2
        )

    def collision(self):
        """Performs collision in moment space at each node."""
        moments = torch.einsum("ab,ijb->ija", self.weights_to_moments, self.new_weights)

        new_moments = self.one_minus_tau_inverse * moments + self.tau_inverse * self.eq_weights

        self.weights = torch.einsum("ab,ijb->ija", self.moments_to_weights, new_moments)

    def handle_boundaries(self):
        """Enforces selected boundary conditions for all edges."""
        for i, bc in enumerate(self.boundary_conditions):
            if bc == BC.VON_NEUMANN:
                self.handle_boundary(i)

    def forward(self, input):
        """Performs one step of the lbm scheme on input."""
        self.weights = input.clone()

        self.handle_boundaries()
        self.streaming()
        self.update_macroscopic()
        self.calculate_equilibrium()
        self.collision()

        return self.weights

@dataclass(slots=True)
class LearningConfig:
    learning_epochs: int
    learning_rate: float
    min_loss: float

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