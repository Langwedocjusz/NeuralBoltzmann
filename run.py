import torch
import numpy as np

import src.reference_test as ref_test
import src.torch_ref_test as torch_test
import src.minimal_model_test as minimal_test

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

#torch_test.TestSimulation()
#torch_test.TestSimulationPoiseuille()

#=====================================

minimal_test.TestSimulation()

