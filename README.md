# NeuralBoltzmann

Numerical optimization of collision operators for the Lattice Boltzmann scheme. Project realized under the supervision of Michał Dzikowski.

## Lattice Boltzmann

Core of this project is an implementation of the Lattice Boltzmann D2Q9 scheme for simulating flows, written in PyTorch. All functionality apart from equilibrium computation and collision is implemented by the abstract base class Lbm, found in the `src/reference.py` module. In the same module there are also specializations that implement reference collision operators, like the classical BGK. Those reference solvers have been tested against some scenarios, defined in the `src/reference_test.py` module, like:

### Flow around a cylinder

| Re = 4 | Re = 150 |
| ------------ | ------------- |
| ![alt text](https://github.com/Langwedocjusz/NeuralBoltzmann/blob/main/img/cyl2.gif?raw=true) | ![alt text](https://github.com/Langwedocjusz/NeuralBoltzmann/blob/main/img/cyl.gif?raw=true) | 

### Poiseuille flow

| Development of the flow | Agreement with theoretical value |
| ------------ | ------------- |
| ![alt text](https://github.com/Langwedocjusz/NeuralBoltzmann/blob/main/img/poise_dev.gif?raw=true) | ![alt text](https://github.com/Langwedocjusz/NeuralBoltzmann/blob/main/img/vel_profile.png?raw=true) | 

## LBM Layers

The module `src/lbm_layer.py` implements layers that perform LBM simulation with custom collision operators. They utilize base lbm functionality, so only the collision part needs to be specified. They also subclass `torch.nn.Module` and as such can be easily used with PyTorch's optimizers as seen in `src/lbm_training.py`.

