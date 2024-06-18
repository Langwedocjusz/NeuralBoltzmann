"""This module implements tools for numerically solving the diffusion equation."""

import torch

#from src import plotting

def get_diffusion_operator(grid_size: int, dt: float):
    """
    Returns a matrix of shape (grid_size, grid_size) that
    implements time evolution of the diffusion equation with
    fixed values at the boundaries.
    """

    #Matrix that will implement finite differences
    A = torch.zeros((grid_size, grid_size), dtype=torch.float)

    #Central difference second derivative coefficients
    stencil = torch.tensor([1.0, -2.0, 1.0])

    #Skip first and last row, since boundaries are fixed
    for i in range(1, grid_size-1):
        #Everywhere else place the stencil at the diagonal
        for j in range(0, grid_size):
            s_id = i-j + 1
            if 0 <= s_id <= 2:
                A[i][j] = stencil[s_id]

    # (1 + dt A)v = v + d_xx v = v + d_t v
    return torch.eye(grid_size) + dt*A

class Solver:
    """
    Finite difference solver for the 1+1 dimensional
    diffusion equation, with fixed value boundary conditions.
    """
    def __init__(self, grid_size: int, dt: float):
        self.solution = torch.zeros(grid_size, dtype=torch.float)
        self.evolution_operator = get_diffusion_operator(grid_size, dt)

    def set_initial_data(self, init: torch.tensor):
        """Clones the provided tensor data into Solver's solution."""

        if self.solution.shape != init.shape:
            raise RuntimeError("Provided initial data doesn't match prevoiusly defined grid size")

        self.solution = torch.clone(init)

    def evolve(self, num_steps: int):
        """Applies the evolution operator specified number of times."""

        for _ in range(0, num_steps):
            self.solution = torch.einsum("ij,j->i", self.evolution_operator, self.solution)


def evolve_num_steps(initial_data: torch.tensor, dt: float, num_steps: int):
    """
    Evolves the initial data for a given number of steps,
    using the finite difference diffusion equation solver.
    """
    grid_size = initial_data.size()[0]

    solver = Solver(grid_size, dt)
    solver.set_initial_data(initial_data)
    solver.evolve(num_steps)

    return solver.solution


def evolve_to_convergence(initial_data: torch.tensor, dt: float, epsilon: float):
    """
    Evolves the initial data using the finite difference
    diffusion equation solver until difference between two consecutive
    iterations has supremum norm smaller than epsilon.
    """

    grid_size = initial_data.size()[0]

    solver = Solver(grid_size, dt)
    solver.set_initial_data(initial_data)

    sup_norm: float = 1e9

    num_iter = 0

    while sup_norm > epsilon:
        old_solution = solver.solution
        solver.evolve(1)
        sup_norm = torch.amax(torch.abs(solver.solution - old_solution))
        #plotting.show_function_1d(solver.solution)

        num_iter += 1

    #print(f'Classical solver: convergence took {num_iter} steps')

    return solver.solution
