import torch
import math

import src.torch_ref as ref
import src.plotting as plotting

def TestMacroscopic():
    """
    Test of the UpdateMacroscopic method of the lbm class.
    Method is called with new_weights values set by hand.
    The resulting denisties and velocities are then plotted.
    """

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
    """
    Test of the Streaming method of the lbm class.
    Method is called for single-node blobs of fluid moving in all of the 9 principal directions of D29Q lattice.
    Streaming is then called twice and the movement (with periodic boundary conditions) is plotted.
    """

    config = ref.SimulationConfig(3, 3, 0.5)
    lbm = ref.Lbm(config)

    for test_id in range(0,9):
        lbm.weights     = torch.zeros(lbm.shape, dtype=torch.float)
        lbm.new_weights = torch.zeros(lbm.shape, dtype=torch.float)
        
        lbm.weights[1, 1, test_id] = 1.0

        plotting.ShowHeatmap(lbm.weights[:,:,test_id], str(test_id))
        lbm.Streaming()
        plotting.ShowHeatmap(lbm.new_weights[:,:,test_id], str(test_id))
        lbm.weights = lbm.new_weights
        lbm.Streaming()
        plotting.ShowHeatmap(lbm.new_weights[:,:,test_id], str(test_id))

def TestEquilibrium():
    """
    Test of the CalculateEquilibrium method of the lbm class.
    The method is called on a lattice with hand picked densities and velocities.
    The resulting equilibrium weights are then printed.
    """

    config = ref.SimulationConfig(3, 3, 0.5)
    lbm = ref.Lbm(config)

    lbm.densities[:,:] = 0.1
    lbm.new_weights = torch.full(lbm.shape, 0.01)

    lbm.densities[0,0] = 1.0
    lbm.densities[1,0] = 1.0
    lbm.new_weights[0,0] = torch.tensor([0.0, 0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0])
    lbm.new_weights[1,0] = torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    lbm.UpdateMacroscopic()

    lbm.CalculateEquilibrium()

    print(lbm.eq_weights)

def TestBoundary():
    """ 
    Test of boundary conditions handling in the lbm class.
    Left wall is assigned nonzero von neumann velocity.
    Fluid is initialized with a constant distribution and simulation is carried out.
    Velocity fields at subsequent simulation steps are then plotted.
    """
    
    size = 20

    config = ref.SimulationConfig(size, size, 0.5)
    config.boundary_conditions = (ref.BC.PERIODIC, ref.BC.VON_NEUMANN, ref.BC.PERIODIC, ref.BC.VON_NEUMANN)
    #config.boundary_conditions = (ref.BC.VON_NEUMANN, ref.BC.PERIODIC, ref.BC.VON_NEUMANN, ref.BC.PERIODIC)

    lbm = ref.Lbm(config)
    lbm.InitStationary()

    v = 0.1

    a = - 4.0 * v / (size*size)

    for i in range(0, size):
        vel = a * (i-size/2.0)**2 + v

        lbm.boundary_velocities[1][i] = v
        #lbm.boundary_velocities[3][i] = v
        #lbm.boundary_velocities[0][i] = v
        #lbm.boundary_velocities[2][i] = v

    for i in range(0, 20):
        lbm.Simulate(1)
        plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')
        #plotting.ShowHeatmap(lbm.velocities_x, 'velocity x')

def TestSimulation():
    """
    Test of fluid simulation using the lbm class.
    Fluid density is initialized as a gaussian distribution.
    It then flows around 4 symmetrically placed cylinders.
    """

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
     
            rho = 1.0 + math.exp(-(x*x + y*y))
    
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
    """
    Test of fluid simulation using the lbm class.
    Lattice is initialized as a vertical pipe with periodic boundary conditions
    on top and bottom and with nonzero gravity.
    Simulation is then carried out and resulting velocity profile is plotted
    and compared to the theoretical profile of a Poiseuille flow.
    """

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
    
    theoretical = torch.zeros(size_y)
    
    for i in range(0, size_y):
        x = i - a
        theoretical[i] = g * (a*a - x*x)/(2.0*nu)
    
    functions = [numeric, theoretical]
    names = ['numeric', 'theoretical']
    
    plotting.ShowFunctions1d(functions, names, 'velocity profile')

def SimulateCylinder():
    """
    Test of fluid simulation using the lbm class.
    Fluid density is initialized as a constant distribution.
    Left and right walls are assigned von neumann (inflow/outflow) velocities,
    while top and bottom are kept as periodic.
    A cyllinder is placed inside the lattice and flow simulation is carried out.
    As the flow develops, resulting flow lines are plotted.
    """

    size_x = 150
    size_y = 300
    rad = 12.0
    v = 0.04
    Re = 1.0

    nu = v*rad/Re
    tau = 3.0*nu+0.5

    config = ref.SimulationConfig(size_x, size_y, tau)
    config.boundary_conditions = (ref.BC.PERIODIC, ref.BC.VON_NEUMANN, ref.BC.PERIODIC, ref.BC.VON_NEUMANN)

    lbm = ref.Lbm(config)
    lbm.InitStationary()
    lbm.weights *= 1.0

    for i in range(0, size_x):
        a = - 4.0 * v / (size_x*size_x)
        vel = a * (i-size_x/2.0)**2 + v

        lbm.boundary_velocities[1][i] = v#el
        lbm.boundary_velocities[3][i] = -v#el

    center_x = 0.5*size_x
    center_y = (1.0/6.0)*size_y

    for i in range(0, size_x):
        for j in range(0, size_y):
            dx = i - center_x
            dy = j - center_y

            if dx*dx + dy*dy < rad**2:
                lbm.solid_mask[i,j] = True

    for i in range(0,50):
        lbm.Simulate(25)
        
        plotting.ShowFlowLines(lbm.velocities_x, lbm.velocities_y)
        #u2 = torch.square(lbm.velocities_x) + torch.square(lbm.velocities_y)
        #plotting.ShowHeatmap(u2, 'velocity magnitude', 0.0, 0.1)
        #plotting.ShowHeatmap(lbm.densities, 'density')
        #plotting.ShowVectorField(lbm.velocities_x, lbm.velocities_y, 'velocity')