import src.reference_test as ref_test
import src.torch_ref_test as torch_test
import src.minimal_model_test as minimal_test
import src.minimal_training as minimal_training

from mock.optimizer import LearningConfig
from mock.optimizer import train_linear
from mock.optimizer import train_custom


def main():
    config = LearningConfig(2000, 1e-3, 1e-3)

    #train_linear(config)
    train_custom(config)

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

    #minimal_test.TestSimulation()
    #minimal_training.TestTraining()


if __name__ == "__main__":
    main()
