import os
import numpy as np
from scipy.io import loadmat
from utils import *
from training import *

# Ensure the results directory exists
if not os.path.exists('./Python_Ver/results'):
    os.makedirs('./Python_Ver/results')

# List of dataset names
DataNameList = ["./Python_Ver/data/rat01.mat", "Python_Ver/data/rat02.mat"]

for DataName in DataNameList:
    data = data_setup(DataName)  # Load data
    opt = opt_Setup_real(data)  # Set options
    
    results_dir = f'./Python_Ver/results/{opt["DataIndex"]}/'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    np.random.seed(0)  # Set random seed to default
    
    # PreProcess: get mPFC ensemble for every time bin
    inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
        data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
    )
    for sch in ["RL", "Sup"]:
        for testFold in range(opt["foldNum"]):  # 5-fold cross-validation
            opt["testFold"] = testFold
            trainFolds = list(range(opt["foldNum"]))
            trainFolds.remove(testFold)
            opt["trainTrials"] = np.concatenate([opt["folds"][i] for i in trainFolds])
            opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
            opt["testTrials"] = opt["folds"][testFold] 
            opt["NumberOfTestTrials"] = len(opt["testTrials"])
            
            print(f'----{opt["DataIndex"]} {sch} fold {opt["testFold"]} Train start----')
            
            for _ in range(opt["ReTrainTimes"]):
                if sch == 'RL':
                    RLPP(data, opt, inputEnsemble, M1_truth, Actions, Trials)
                else:   
                    SupervisedLearning(data, opt, inputEnsemble, M1_truth, Actions, Trials)
            
            # Evaluate results
            paths = [f for f in os.listdir(results_dir) if f.startswith(f'{sch}_{opt["testFold"]}_') and f.endswith('.mat')]
            SucRate = []
            testSuc = []
            
            for path in paths:
                file_path = os.path.join(results_dir, path)
                data_dict = loadmat(file_path)
                SucRate.append(data_dict["MaxReward"].item())
                testSuc.append(data_dict["testSucRate"].item())
            
            best_idx = np.argmax(SucRate)
            best_path = paths[best_idx]
            source = os.path.join(results_dir, best_path)
            destination = f'./Python_Ver/results/{data["DataIndex"]}_{sch}_{opt["testFold"]}.mat'
            
            os.replace(source, destination)
            print(f'{destination} testSuc: {testSuc[best_idx]}')
