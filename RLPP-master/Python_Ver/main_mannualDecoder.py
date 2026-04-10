import os
import numpy as np
from utils import data_setup
from utils.opt_Setup_real import opt_Setup_real
from utils.PreProcess import PreProcess
from training import RLPP
from scipy.io import loadmat

# Ensure the results directory exists
if not os.path.exists('./Python_Ver/results'):
    os.makedirs('./Python_Ver/results')

# Model training
DataName = "./Python_Ver/data/rat02.mat"

data = data_setup(DataName)  # Load data
data["M1num_pre"] = 4

data["modelName"] = "decodingModel_manual"
data["his"] = 50
data["DataIndex"] = data["DataIndex"] + "Manual"
print(data["DataIndex"])

opt = opt_Setup_real(data)  # Set options

resultsDir = f'./Python_Ver/results/{opt["DataIndex"]}/'
if not os.path.exists(resultsDir):
    os.makedirs(resultsDir)

np.random.seed(0)  # Set random seed to default

# PreProcess: get mPFC ensemble for every time bin
inputEnsemble, M1Truth, Actions, Trials, opt = PreProcess(
    data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
)

sch = "RL"

for testFold in range(opt["foldNum"]):  # 5-fold cross-validation
    opt["testFold"] = testFold
    trainFolds = list(range(opt["foldNum"]))
    trainFolds.remove(testFold)
    opt["trainTrials"] = np.concatenate([opt["folds"][i] for i in trainFolds]).flatten()
    opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
    opt["testTrials"] = opt["folds"][testFold]
    opt["NumberOfTestTrials"] = len(opt["testTrials"])

    print(f'----{opt["DataIndex"]} {sch} fold {opt["testFold"]} Train start----')
    
    for i in range(opt["ReTrainTimes"]):
        RLPP(data, opt, inputEnsemble, M1Truth, Actions, Trials)

    # Evaluate results
    paths = [f for f in os.listdir(resultsDir) if f.startswith(f'{sch}_{opt["testFold"]}_') and f.endswith('.mat')]
    SucRate = []
    testSuc = []
    
    for path in paths:
        file_path = os.path.join(resultsDir, path)
        data_dict = loadmat(file_path)
        SucRate.append(data_dict["MaxReward"].item())
        testSuc.append(data_dict["testSucRate"].item())
    
    best_idx = np.argmax(SucRate)
    best_path = paths[best_idx]
    source = os.path.join(resultsDir, best_path)
    destination = f'./Python_Ver/results/MATdata_{data["DataIndex"]}_{sch}_{opt["testFold"]}.mat'
    
    os.replace(source, destination)
    print(f'{destination} testSuc: {testSuc[best_idx]}')
