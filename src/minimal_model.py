import torch
import numpy as np

from dataclasses import dataclass
from enum import Enum

class BC(Enum):
    PERIODIC = 0
    VON_NEUMANN = 1

@dataclass
class SimulationConfig:
    grid_size_x: int
    grid_size_y: int
    tau: float
    gravity: (float, float) = (0.0, 0.0)
    boundary_conditions: (BC, BC, BC, BC) = (BC.PERIODIC, BC.PERIODIC, BC.PERIODIC, BC.PERIODIC)

class Lbm:
    def __init__(self, config: SimulationConfig):
        self.tau = config.tau
        self.tau_inverse = 1.0/config.tau
        self.one_minus_tau_inverse = 1.0 - self.tau_inverse

        self.grid_size_x = config.grid_size_x
        self.grid_size_y = config.grid_size_y

        self.gravity = config.gravity

        #Lattice initialization
        self.shape   = (config.grid_size_x, config.grid_size_y, 9)
        self.shape2d = (config.grid_size_x, config.grid_size_y, 1)
        self.shape1d = (1, 1, 9)

        self.weights     = torch.zeros(self.shape, dtype=torch.float)
        self.new_weights = torch.zeros(self.shape, dtype=torch.float)

        self.eq_factors = torch.tensor([
            4.0/9.0, 
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
        ]).reshape(self.shape1d)

        self.densities    = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float)
        self.velocities_x = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float)
        self.velocities_y = torch.zeros((config.grid_size_x, config.grid_size_y), dtype=torch.float)

        #Collision parameters (to be trained)
        self.M0 = torch.zeros((9,9), dtype=torch.float)
        self.D  = torch.zeros((9,9), dtype=torch.float)
        self.Q  = torch.zeros((9,9), dtype=torch.float)
        self.phi0 = torch.zeros((9), dtype=torch.float)

        #Boundaries
        self.boundary_conditions = config.boundary_conditions

        self.boundary_velocities = [
            torch.zeros(config.grid_size_y, dtype=torch.float),
            torch.zeros(config.grid_size_x, dtype=torch.float),
            torch.zeros(config.grid_size_y, dtype=torch.float),
            torch.zeros(config.grid_size_x, dtype=torch.float),
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
        self.indices = torch.tensor(np.array((rows, cols, tubes)).T)

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
        #diag = 0.5 * np.sqrt(2.0)
        diag = 1.0

        self.base_velocities_x = torch.tensor([0.0, 1.0, 0.0, -1.0, 0.0, diag, -diag, -diag, diag])
        self.base_velocities_y = torch.tensor([0.0, 0.0, 1.0, 0.0, -1.0, diag, diag, -diag, -diag])

        #Solid mask
        self.solid_mask = torch.zeros(self.shape2d, dtype=torch.bool)

        #Bounceback indices
        self.default_ids = [0,1,2,3,4,5,6,7,8]
        self.swapped_ids = [0,2,1,4,3,7,8,5,6]

        #Disable flow at non-periodic boundaries 
        if (self.boundary_conditions[0] == BC.VON_NEUMANN):
            for i in range(0, self.grid_size_y):
                for a in range(0,9):
                    self.indices[a,i,0] = [0,i,a]

        if (self.boundary_conditions[1] == BC.VON_NEUMANN):
            for i in range(0, self.grid_size_x):
                for a in range(0,9):
                    self.indices[a,0,i] = [i,0,a]

        if (self.boundary_conditions[0] == BC.VON_NEUMANN):
            lx = self.grid_size_x - 1

            for i in range(0, self.grid_size_y):
                for a in range(0,9):
                    self.indices[a,i,lx] = [lx,i,a]

        if (self.boundary_conditions[3] == BC.VON_NEUMANN):
            ly = self.grid_size_y - 1

            for i in range(0, self.grid_size_x):
                for a in range(0,9):
                    self.indices[a,ly,i] = [i,ly,a]


    def InitStationary(self):
        self.weights[:,:] = torch.tensor(
            [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
        )

    def InitExactCollision(self):
        e = torch.stack((self.base_velocities_x, self.base_velocities_y))
        A = torch.matmul(e.t(), e)

        w = torch.tensor(
            [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]
        )

        diag_w = torch.diag(w)

        self.M0 = self.one_minus_tau_inverse * torch.eye(9) + 3.0 * self.tau_inverse * torch.matmul(A, diag_w)
        self.D = np.sqrt(4.5*self.tau_inverse) * torch.matmul(A, torch.sqrt(diag_w))
        self.phi0 = (1.0/self.tau) * w
        self.Q = (-1.5 * self.tau_inverse) * A

    def HandleBoundary(self, edge_id: int):
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
            rho = self.weights[pos,:,0] + self.weights[pos,:,mid_ids[0]] + self.weights[pos,:,mid_ids[0]]
            + 2.0 * (self.weights[pos,:,out_ids[0]] + self.weights[pos,:,out_ids[1]] + self.weights[pos,:,out_ids[2]])

            rho = rho/(1.0 + v)
            ru = v * rho

            self.weights[pos,:,in_ids[0]] = self.weights[pos,:,out_ids[0]] - (2.0/3.0)*ru
            self.weights[pos,:,in_ids[1]] = self.weights[pos,:,out_ids[1]] - (1.0/6.0)*ru + 0.5*(self.weights[pos,:,mid_ids[0]] - self.weights[pos,:,mid_ids[0]])
            self.weights[pos,:,in_ids[2]] = self.weights[pos,:,out_ids[2]] - (1.0/6.0)*ru + 0.5*(self.weights[pos,:,mid_ids[1]] - self.weights[pos,:,mid_ids[1]])
        else:
            rho = self.weights[:,pos,0] + self.weights[:,pos,mid_ids[0]] + self.weights[:,pos,mid_ids[1]]
            + 2.0 * (self.weights[:,pos,out_ids[0]] + self.weights[:,pos,out_ids[1]] + self.weights[:,pos,out_ids[2]])

            rho = rho/(1.0 + v)
            ru = v * rho

            self.weights[:,pos,in_ids[0]] = self.weights[:,pos,out_ids[0]] - (2.0/3.0)*ru
            self.weights[:,pos,in_ids[1]] = self.weights[:,pos,out_ids[1]] - (1.0/6.0)*ru + 0.5*(self.weights[:,pos,mid_ids[0]] - self.weights[:,pos,mid_ids[1]])
            self.weights[:,pos,in_ids[2]] = self.weights[:,pos,out_ids[2]] - (1.0/6.0)*ru + 0.5*(self.weights[:,pos,mid_ids[1]] - self.weights[:,pos,mid_ids[0]])

    def HandleBoundaries(self):
        for i in range(0, len(self.boundary_conditions)):
            if self.boundary_conditions[i] == BC.VON_NEUMANN:
                self.HandleBoundary(i)

    def Streaming(self):
        stream_weights = self.weights[*self.indices.permute(3,2,1,0)]

        bounced_weights = torch.zeros(self.shape, dtype=torch.float)
        bounced_weights[:,:,self.default_ids] = self.weights[:,:,self.swapped_ids]

        self.new_weights = (~self.solid_mask) * stream_weights + self.solid_mask * bounced_weights

    def UpdateMacroscopic(self):
        self.densities    = torch.sum(self.new_weights, axis=2)
        self.velocities_x = torch.einsum("ijk,k->ij", (self.new_weights, self.base_velocities_x))
        self.velocities_x = self.velocities_x / self.densities
        self.velocities_y = torch.einsum("ijk,k->ij", (self.new_weights, self.base_velocities_y))
        self.velocities_y = self.velocities_y / self.densities

    def Collision(self):
        M0 = torch.matmul(self.new_weights, self.M0)

        D = torch.square(torch.matmul(self.new_weights, self.D))
        D = D/self.densities.reshape(self.shape2d)

        phi0 = self.densities.reshape(self.shape2d) * self.phi0

        Q  = (torch.einsum("ijk,ijk->ij", (self.new_weights, torch.matmul(self.new_weights, self.Q))))/self.densities
        
        self.weights = M0 + D + phi0 + Q.reshape(self.shape2d) * self.eq_factors

    def Simulate(self, num_steps: int):
        for i in range (0, num_steps):
            self.HandleBoundaries()
            self.Streaming()
            self.UpdateMacroscopic()
            self.Collision()