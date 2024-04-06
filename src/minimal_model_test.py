import src.minimal_model as solver
import src.plotting as plotting

import torch
import numpy as np

def TestSimulation():
    size = 200

    config = solver.SimulationConfig(size, size, 0.5)
    lbm = solver.Lbm(config)
    lbm.InitStationary()
    lbm.InitExactCollision()

    circle_rad = 15.0
    a = 0.33

    circle_pos_x = [a*size, (1.0-a)*size, a*size, (1.0-a)*size]
    circle_pos_y = [a*size, a*size, (1.0-a)*size, (1.0-a)*size]

    for i in range(0, size):
        for j in range(0, size):
            x = (i-size/2.0)/10.0
            y = (j-size/2.0)/10.0
     
            rho = 1.0 + np.exp(-(x*x + y*y))
    
            lbm.densities[i,j] = rho
            lbm.weights[i,j] *= rho

            for k in range(0,4):
                dx = circle_pos_x[k] - i
                dy = circle_pos_y[k] - j

                d2 = dx*dx + dy*dy

                if d2 < circle_rad * circle_rad:
                    lbm.solid_mask[i,j] = True

    for i in range(0, 15):
        lbm.Simulate(10)
        #plotting.ShowHeatmap(lbm.densities, 'density', 0.0, 2.0)
        plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')
        #plotting.SaveHeatmap(lbm.densities, 'densities', str(i), 0.0, 2.0)
        #plotting.SaveVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity', str(i))