import numpy as np
from dataclasses import dataclass

@dataclass
class SimulationConfig:
    grid_size_x: int
    grid_size_y: int
    tau: float
    gravity: (float, float) = (0.0, 0.0)

class Lbm:
    def __init__(self, config: SimulationConfig):
        self.tau = config.tau
        self.tau_inverse = 1.0/config.tau
        self.one_minus_tau_inverse = 1.0 - self.tau_inverse

        self.gravity = config.gravity

        #Lattice initialization
        self.shape   = (config.grid_size_x, config.grid_size_y, 9)
        self.shape2d = (config.grid_size_x, config.grid_size_y, 1)
        self.shape1d = (1, 1, 9)

        self.weights     = np.zeros(self.shape, dtype=float)
        self.new_weights = np.zeros(self.shape, dtype=float)
        self.eq_weights  = np.zeros(self.shape, dtype=float)

        self.eq_factors = np.array([
            4.0/9.0, 
            1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
            1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0
        ]).reshape(self.shape1d)

        self.densities    = np.zeros((config.grid_size_x, config.grid_size_y), dtype=float)
        self.velocities_x = np.zeros((config.grid_size_x, config.grid_size_y), dtype=float)
        self.velocities_y = np.zeros((config.grid_size_x, config.grid_size_y), dtype=float)

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
        #diag = 0.5 * np.sqrt(2.0)
        diag = 1.0

        self.base_velocities_x = np.array([0.0, 1.0, 0.0, -1.0, 0.0, diag, -diag, -diag, diag])
        self.base_velocities_y = np.array([0.0, 0.0, 1.0, 0.0, -1.0, diag, diag, -diag, -diag])

        #Solid mask
        self.solid_mask = np.zeros(self.shape2d, dtype=bool)

        #Bounceback indexes
        self.default_ids = [0,1,2,3,4,5,6,7,8]
        self.swapped_ids = [0,2,1,4,3,7,8,5,6]

    def InitStationary(self):
        self.weights[:,:] = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]

    def UpdateMacroscopic(self):
        self.densities    = np.sum(self.new_weights, axis=2)
        self.velocities_x = np.dot(self.new_weights, self.base_velocities_x)
        self.velocities_x = self.velocities_x / self.densities
        self.velocities_y = np.dot(self.new_weights, self.base_velocities_y)
        self.velocities_y = self.velocities_y / self.densities

    def Streaming(self):
        stream_weights = self.weights[*self.indices.T]

        bounced_weights = np.zeros(self.shape, dtype=float)
        bounced_weights[:,:,self.default_ids] = self.weights[:,:,self.swapped_ids]

        self.new_weights = (1.0 - self.solid_mask) * stream_weights + self.solid_mask * bounced_weights

    def CalculateEquilibrium(self):
        #Second order approximation to Maxwell distribution:
        # f^eq_i = w_i * rho * (1.0 + 3.0 e_i.u + 4.5 * (e_i.u)^2 - 1.5 u.u)

        (gx, gy) = self.gravity
        velocities_eq_x = self.velocities_x + self.tau * gx * self.densities
        velocities_eq_y = self.velocities_y + self.tau * gy * self.densities

        eDotUx = self.base_velocities_x.reshape(self.shape1d) * velocities_eq_x.reshape(self.shape2d)
        eDotUy = self.base_velocities_y.reshape(self.shape1d) * velocities_eq_y.reshape(self.shape2d)

        eDotU = eDotUx + eDotUy
        eDotU2 = np.square(eDotU)

        u2 = np.square(velocities_eq_x) + np.square(velocities_eq_y)

        f_tmp = 1.0 + 3.0 * eDotU + 4.5 * eDotU2 - 1.5 * u2.reshape(self.shape2d)
        f_tmp = self.densities.reshape(self.shape2d) * f_tmp

        self.eq_weights = f_tmp * self.eq_factors

    def Collision(self):
        #BGK collision operator:
        #f - > f + 1/tau * (f_eq - f) = (1 - 1/tau) * f + f_eq/tau
        self.weights = self.one_minus_tau_inverse * self.new_weights + self.tau_inverse * self.eq_weights

    def Simulate(self, num_steps: int):
        for i in range (0, num_steps):
            self.Streaming()
            self.UpdateMacroscopic()
            self.CalculateEquilibrium()
            self.Collision()