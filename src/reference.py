import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    grid_size_x: int
    grid_size_y: int
    tau: float

class Lbm:
    def __init__(self, config: SimulationConfig):
        self.tau_inverse = 1.0/config.tau
        self.one_minus_tau_inverse = 1.0 - self.tau_inverse

        #Lattice initialization
        self.shape   = (config.grid_size_x, config.grid_size_x, 9)
        self.shape2d = (config.grid_size_x, config.grid_size_y, 1)
        self.shape1d = (1, 1, 9)

        self.weights     = np.full(self.shape, 0.0)
        self.new_weights = np.full(self.shape, 0.0)
        self.eq_weights  = np.full(self.shape, 0.0)

        self.eq_factors = np.array([
            4.0/9.0, 
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
        ]).reshape(self.shape1d)

        self.densities    = np.full((config.grid_size_x, config.grid_size_y), 0.0)
        self.velocities_x = np.full((config.grid_size_x, config.grid_size_y), 0.0)
        self.velocities_y = np.full((config.grid_size_x, config.grid_size_y), 0.0)

        #Weight ids convention:
        #  6 -- 2 -- 5
        #  |    |    |
        #  3 -- 0 -- 1
        #  |    |    |
        #  7 -- 4 -- 8

        #Left moving population at a given node was one step ago
        #at a node one unit to the right (since it's left moving)
        #so in the streaming step we need to index with opposite indices:

        rows, cols, tubes = np.indices(self.shape)

        self.indices = np.array((rows, cols, tubes)).T

        #Left, right
        self.indices[1,:,:] = np.roll(self.indices[1,:,:],  1, axis=0)
        self.indices[3,:,:] = np.roll(self.indices[3,:,:], -1, axis=0)

        #Up, down
        self.indices[2,:,:] = np.roll(self.indices[2,:,:],  1, axis=1)
        self.indices[4,:,:] = np.roll(self.indices[4,:,:], -1, axis=1)

        #Upper corners
        self.indices[5,:,:] = np.roll(self.indices[5,:,:],  1, axis=1)
        self.indices[5,:,:] = np.roll(self.indices[5,:,:],  1, axis=0)

        self.indices[6,:,:] = np.roll(self.indices[6,:,:],  1, axis=1)
        self.indices[6,:,:] = np.roll(self.indices[6,:,:], -1, axis=0)

        #Lower corners
        self.indices[7,:,:] = np.roll(self.indices[7,:,:], -1, axis=1)
        self.indices[7,:,:] = np.roll(self.indices[7,:,:], -1, axis=0)

        self.indices[8,:,:] = np.roll(self.indices[8,:,:], -1, axis=1)
        self.indices[8,:,:] = np.roll(self.indices[8,:,:],  1, axis=0)

        #Base velocities
        diag = 0.5 * np.sqrt(2.0)

        self.base_velocities_x = np.array([0.0, 1.0, 0.0, -1.0, 0.0, diag, -diag, -diag, diag])
        self.base_velocities_y = np.array([0.0, 0.0, 1.0, 0.0, -1.0, diag, diag, -diag, -diag])

    def UpdateMacroscopic(self):
        self.densities    = np.sum(self.weights, axis=2)
        self.velocities_x = np.dot(self.weights, self.base_velocities_x)
        self.velocities_y = np.dot(self.weights, self.base_velocities_y)

    def Streaming(self):
        self.new_weights = self.weights[*self.indices.T]

    def CalculateEquilibrium(self):
        #Second order approximation to Maxwell distribution:
        # f^eq_i = w_i * rho * (1.0 + 3.0 e_i.u + 4.5 * (e_i.u)^2 - 1.5 u.u)

        eDotUx = self.base_velocities_x.reshape(self.shape1d) * self.velocities_x.reshape(self.shape2d)
        eDotUy = self.base_velocities_y.reshape(self.shape1d) * self.velocities_y.reshape(self.shape2d)

        eDotU = eDotUx + eDotUy
        eDotU2 = np.square(eDotU)

        u2 = np.square(self.velocities_x) + np.square(self.velocities_y)

        f_tmp = 1.0 + 3.0 * eDotU + 4.5 * eDotU2 - 1.5 * u2.reshape(self.shape2d)
        f_tmp = self.densities.reshape(self.shape2d) * f_tmp

        self.eq_weights = f_tmp * self.eq_factors

    def Collision(self):
        #BGK collision operator:
        #f - > f + 1/tau * (f_eq - f)
        self.weights = self.one_minus_tau_inverse * self.new_weights + self.tau_inverse * self.eq_weights