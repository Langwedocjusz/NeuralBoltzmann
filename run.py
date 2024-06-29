import torch

from src import reference_test as ref_test

from src.simconfig import SimulationConfig
from src.learning_config import LearningConfig
from src.lbm_layer import LbmLayer

from src.lbm_training import train_gaussian
from src.lbm_training import train_poiseuille
from src.lbm_training import train_gaussian_batch

from src.lbm_data import weights_from_macroscopic

from src import plotting

def test_ref():
    ref_test.test_macroscopic()
    ref_test.test_streaming()
    ref_test.test_equilibrium()
    ref_test.test_boundary()

    ref_test.test_simulation()
    ref_test.test_simulation_poiseuille()
    ref_test.simulate_cylinder()

def main():
    layer = LbmLayer.MINIMAL_HERMITE

    config = LearningConfig(10000, 1e-3, 1e-3)
    train_gaussian(layer, config, True)

    #config = LearningConfig(2500, 1e-3, 1e-3)
    #train_poiseuille(layer, config, True)

    #train_gaussian_batch(layer, config, True)


if __name__ == "__main__":
    main()
