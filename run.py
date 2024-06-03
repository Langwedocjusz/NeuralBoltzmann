import src.reference_test as ref_test
import src.torch_ref_test as torch_test
import src.minimal_model_test as minimal_test
import src.minimal_training as minimal_training

from mock.optimizer import LearningConfig
from mock.optimizer import train_linear
from mock.optimizer import train_custom

def main():
    config = LearningConfig(1000, 1e-3, 1e-3)

    #train_linear(config)
    train_custom(config)

    #=====================================

    #ref_test.TestMacroscopic()
    #ref_test.TestStreaming()
    #ref_test.TestEquilibrium()
    #ref_test.TestBoundary()

    #ref_test.TestSimulation()
    #ref_test.TestSimulationPoiseuille()
    #ref_test.SimulateCylinder()

    #=====================================

    #torch_test.TestMacroscopic()
    #torch_test.TestStreaming()
    #torch_test.TestEquilibrium()
    #torch_test.TestBoundary()

    #torch_test.TestSimulation()
    #torch_test.TestSimulationPoiseuille()
    #torch_test.SimulateCylinder()

    #=====================================

    #minimal_test.TestSimulation()
    #minimal_training.TestTraining()

if __name__ == "__main__":
    main()

