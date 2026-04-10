import os
import numpy as np
import matplotlib.pyplot as plt

from utils import *

from training import RLPP, SupervisedLearning
from draw_figures import showSimuResults

# Ensure the results directory exists
if not os.path.exists('./Python_Ver/results'):
    os.makedirs('./Python_Ver/results')

# Data synthesis
print("Start data synthesis")
fig, axes = simuData()

# Model training
DataName = "./Python_Ver/data/simulations.mat"
data = data_setup(DataName)  # Load data
opt = opt_Setup_simu(data)  # Set options

np.random.seed(0)  # Set random seed to 0

# PreProcess: get mPFC ensemble for every time bin
inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
    data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
)

# Loop over RL and Supervised Learning
for sch in ['RL', 'Sup']:
    for testFold in [0]:
        opt["testFold"] = testFold
        trainFolds = list(range(opt["foldNum"]))
        trainFolds.remove(testFold)
        opt["trainTrials"] = opt["folds"][trainFolds].flatten()
        opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
        opt["testTrials"] = opt["folds"][testFold]
        opt["NumberOfTestTrials"] = len(opt["testTrials"])

        print(f'----{opt["DataIndex"]} {sch} fold {opt["testFold"]} Train start----')
        # s = np.random.get_state()
        if sch == 'RL':
            RLPP(data, opt, inputEnsemble, M1_truth, Actions, Trials)
        else:
            SupervisedLearning(data, opt, inputEnsemble, M1_truth, Actions, Trials)

# Show simulation data
plt.show()
# Show results
showSimuResults()
