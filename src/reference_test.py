import src.plotting as plotting
import src.reference as ref

import numpy as np

def TestMacroscopic():
    config = ref.SimulationConfig(3, 3, 0.05)
    lbm = ref.Lbm(config)

    lbm.weights[0,0,0] = 0.33
    lbm.weights[1,0,1] = 0.33
    lbm.weights[2,0,2] = 0.33
    lbm.weights[0,1,3] = 0.66
    lbm.weights[1,1,4] = 0.66
    lbm.weights[2,1,5] = 0.66
    lbm.weights[0,2,6] = 1.0
    lbm.weights[1,2,7] = 1.0
    lbm.weights[2,2,8] = 1.0

    lbm.UpdateMacroscopic()

    plotting.BasicHeatmap(lbm.densities, 'densities')
    plotting.VectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')

def TestStreaming():
    config = ref.SimulationConfig(3, 3, 0.05)
    lbm = ref.Lbm(config)

    for test_id in range(0,9):
        lbm.weights     = np.full(lbm.shape, 0.0)
        lbm.new_weights = np.full(lbm.shape, 0.0)
        
        lbm.weights[1, 1, test_id] = 1.0

        plotting.BasicHeatmap(lbm.weights[:,:,test_id], str(test_id))
        lbm.Streaming()
        plotting.BasicHeatmap(lbm.new_weights[:,:,test_id], str(test_id))

def TestEquilibrium():
    config = ref.SimulationConfig(3, 3, 0.05)
    lbm = ref.Lbm(config)

    lbm.weights = np.full(lbm.shape, 0.0)

    lbm.weights[0,0] = [0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]

    lbm.weights[1,0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    lbm.UpdateMacroscopic()
    lbm.CalculateEquilibrium()

    #plotting.BasicHeatmap(densities, 'densities')
    #plotting.VectorField(velocities_x, velocities_y, 'velocity')

    print(lbm.eq_weights)