import src.reference_test as ref_test
import src.torch_ref_test as torch_test

from src.learning_config import LearningConfig

from mock.optimizer import train_linear
from mock.optimizer import train_custom

from src.lbm_training import train
from src.lbm_training import test_coherence

def main():
    #config = LearningConfig(2000, 1e-3, 1e-3)

    #train_linear(config)
    #train_custom(config)

    # =====================================

    #ref_test.test_macroscopic()
    #ref_test.test_streaming()
    #ref_test.test_equilibrium()
    #ref_test.test_boundary()

    #ref_test.test_simulation()
    #ref_test.test_simulation_poiseuille()
    #ref_test.simulate_cylinder()

    # =====================================

    #torch_test.test_macroscopic()
    #torch_test.test_streaming()
    #torch_test.test_equilibrium()
    #torch_test.test_boundary()

    #torch_test.test_simulation()
    #torch_test.test_simulation_poiseuille()
    #torch_test.simulate_cylinder()

    # =====================================

    #test_coherence()

    config = LearningConfig(5000, 1e-3, 1e-3)
    train(config)


if __name__ == "__main__":
    main()
