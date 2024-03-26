import src.plotting as plotting
import src.reference as ref

import numpy as np
import time

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
    lbm.InitStationary()

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
        

def TestSimulationPoiseuille():
    size_x = 40
    size_y = 20
    g = 0.01

    config = ref.SimulationConfig(size_x, size_y, 1.0, (0.0, g))
    lbm = ref.Lbm(config)
    lbm.InitStationary()

    lbm.solid_mask[:,0]        = True
    lbm.solid_mask[:,size_y-1] = True

    lbm.Simulate(1000)

    numeric = lbm.velocities_y[0,:]
    
    #theoretical:
    #v = 1/(2 nu) (a^2 - x^2)
    
    a = size_y/2.0
    nu = 0.33*(lbm.tau - 0.5)
    
    theoretical = np.zeros(size_y)
    
    for i in range(0, size_y):
        x = i - a
        theoretical[i] = g * (a*a - x*x)/(2.0*nu)
    
    functions = [numeric, theoretical]
    names = ['numeric', 'theoretical']
    
    plotting.ShowFunctions1d(functions, names, 'velocity profile')

def Curl(vel_x, vel_y):
    range_x = np.arange(0, len(vel_x[:,0]))
    range_y = np.arange(0, len(vel_x[0,:]))

    dv_xx = vel_x[range_x,:] - vel_x[np.roll(range_x, 1),:]
    dv_xy = vel_x[:,range_y] - vel_x[:,np.roll(range_y, 1)]

    dv_yx = vel_y[range_x,:] - vel_y[np.roll(range_x, 1),:]
    dv_yy = vel_y[:,range_y] - vel_y[:,np.roll(range_y, 1)]

    return dv_xx * dv_yy - dv_xy * dv_yx

def SimulateCylinder():
    size_x = 100
    size_y = 400
    rad = 5.0
    g = 0.001

    config = ref.SimulationConfig(size_x, size_y, 0.6, (g, 0.0))
    lbm = ref.Lbm(config)
    lbm.InitStationary()
    lbm.weights *= 1.0

    center_x = 0.5*size_x
    center_y = 0.25*size_y

    for i in range(0, size_x):
        for j in range(0, size_y):
            dx = i - center_x
            dy = j - center_y

            if dx*dx + dy*dy < rad**2:
                lbm.solid_mask[i,j] = True
    
    t0 = time.time()
    lbm.Simulate(500)
    t1 = time.time()

    print("Execution took ", t1-t0, " [s]")

    #for i in range(0,10):
        #lbm.Simulate(100)
        #curl = Curl(lbm.velocities_x, lbm.velocities_y)
        #plotting.ShowHeatmap(curl, 'curl', 0.0, 0.001)
        #plotting.ShowHeatmap(lbm.densities, 'density')
        #plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')


    #plotting.ShowHeatmap(lbm.densities, 'density')
    #plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')