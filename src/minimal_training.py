import src.minimal_model as solver

import torch
import numpy as np

def TestTraining():
    training_steps = 200

    #Simulation config
    size = 200
    num_steps = 50
    tau = 0.5

    def InitialDistribution(densities: torch.tensor, weights: torch.tensor):
        for i in range(0, size):
            for j in range(0, size):
                x = (i-size/2.0)/10.0
                y = (j-size/2.0)/10.0
    
                rho = 1.0 + np.exp(-(x*x + y*y))

                densities[i,j] = rho
                weights[i,j] *= rho

    #Ground truth data via 'exact' solver:
    config = solver.SimulationConfig(size, size, tau)
    
    exact_lbm = solver.Lbm(config)
    exact_lbm.InitStationary()
    exact_lbm.InitExactCollision()

    InitialDistribution(exact_lbm.densities, exact_lbm.weights)
    exact_lbm.Simulate(num_steps)

    exact_weights = exact_lbm.weights

    #Lbm to be trained
    lbm = solver.Lbm(config)

    #Initial collision operators for training
    lbm.M0   = torch.randn((9,9), dtype=torch.float)
    lbm.D    = torch.randn((9,9), dtype=torch.float)
    lbm.Q    = torch.randn((9,9), dtype=torch.float)
    lbm.phi0 = torch.randn((9),   dtype=torch.float)

    #lbm.InitAutoGrad()
    lbm.M0.requires_grad_()
    lbm.D.requires_grad_()
    lbm.Q.requires_grad_()
    lbm.phi0.requires_grad_()

    #Training loop
    learning_rate = 1e-3

    for t in range(training_steps):
        lbm.InitStationary()
        InitialDistribution(lbm.densities, lbm.weights)
        lbm.Simulate(num_steps)

        loss = (exact_weights - lbm.weights).pow(2).sum()

        if t % 100 == 99:
            print(t, loss.item())

        loss.backward()

        #lbm.GradientDescent(learning_rate)
        with torch.no_grad():
            lbm.M0   -= learning_rate * lbm.M0.grad
            lbm.D    -= learning_rate * lbm.D.grad
            lbm.phi0 -= learning_rate * lbm.phi0.grad
            lbm.Q    -= learning_rate * lbm.Q.grad
            
            lbm.M0.grad   = None
            lbm.D.grad    = None
            lbm.phi0.grad = None
            lbm.Q.grad    = None