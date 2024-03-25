import src.plotting as plotting
import src.reference as ref

import numpy as np

def TestMacroscopic():
    config = ref.SimulationConfig(3, 3, 0.5)
    lbm = ref.Lbm(config)

    lbm.new_weights[0,0,0] = 0.33
    lbm.new_weights[1,0,1] = 0.33
    lbm.new_weights[2,0,2] = 0.33
    lbm.new_weights[0,1,3] = 0.66
    lbm.new_weights[1,1,4] = 0.66
    lbm.new_weights[2,1,5] = 0.66
    lbm.new_weights[0,2,6] = 1.0
    lbm.new_weights[1,2,7] = 1.0
    lbm.new_weights[2,2,8] = 1.0

    lbm.UpdateMacroscopic()

    plotting.ShowHeatmap(lbm.densities, 'densities')
    plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')

def TestStreaming():
    config = ref.SimulationConfig(3, 3, 0.5)
    lbm = ref.Lbm(config)

    for test_id in range(0,9):
        lbm.weights     = np.full(lbm.shape, 0.0)
        lbm.new_weights = np.full(lbm.shape, 0.0)
        
        lbm.weights[1, 1, test_id] = 1.0

        plotting.ShowHeatmap(lbm.weights[:,:,test_id], str(test_id))
        lbm.Streaming()
        plotting.ShowHeatmap(lbm.new_weights[:,:,test_id], str(test_id))
        lbm.weights = lbm.new_weights
        lbm.Streaming()
        plotting.ShowHeatmap(lbm.new_weights[:,:,test_id], str(test_id))

def TestEquilibrium():
    config = ref.SimulationConfig(3, 3, 0.5)
    lbm = ref.Lbm(config)

    lbm.densities[:,:] = 0.1
    lbm.new_weights = np.full(lbm.shape, 0.01)

    lbm.densities[0,0] = 1.0
    lbm.densities[1,0] = 1.0
    lbm.new_weights[0,0] = [0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]
    lbm.new_weights[1,0] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    lbm.UpdateMacroscopic()

    lbm.CalculateEquilibrium()

    print(lbm.eq_weights)

def TestSimulation():
    size = 200

    config = ref.SimulationConfig(size, size, 0.5)

    lbm = ref.Lbm(config)

    lbm.weights[:,:] = [4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0]

    for i in range(0, size):
        for j in range(0, size):
            x = (i-size/2.0)/10.0
            y = (j-size/2.0)/10.0
     
            rho = 1.0 + np.exp(-(x*x + y*y))
    
            lbm.densities[i,j] = rho
            lbm.weights[i,j] *= rho

    for i in range(0, 10):
        lbm.Simulate(10)
        #plotting.SaveHeatmap(lbm.densities, 'densities', str(i), 0.0, 2.0)
        plotting.ShowHeatmap(lbm.densities, 'densities', 0.0, 2.0)
