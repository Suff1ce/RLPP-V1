import numpy as np
import scipy.io as sio
from .getTestRasters import getTestRasters

def getModulationDepth(bestFold, lR, hR, lRp, hRp, lRsp, hRsp):

    DataNameList = ['rat01', 'rat02', 'rat03', 'rat04', 'rat05', 'rat06']

    # Initialize modDep as a 12 x 0 array (empty initially)
    modDep = np.zeros((12, 0))
    
    # Iterate over the DataNameList
    for dataIdx in range(6):
        # Load the RL and Sup result files
        RL_data = sio.loadmat(f'./Python_Ver/results/{DataNameList[dataIdx]}_RL_{bestFold[dataIdx]}.mat')
        testActions = RL_data['testActions']
        testM1_truth = RL_data['testM1_truth']
        spkOutPredictTest = RL_data['spkOutPredictTest']
        
        SL_data = sio.loadmat(f'./Python_Ver/results/{DataNameList[dataIdx]}_Sup_{bestFold[dataIdx]}.mat')
        spkOutPredictTest = SL_data['spkOutPredictTest']
        
        getTestRasters()

        base = modDep.shape[1]  # Get the current size of the second dimension of modDep

        # Iterate through the rows of lR, hR, lRp, hRp, lRsp, and hRsp
        for n in range(lR.shape[0]):
            nIdx = n + base
            # Process real data (lR and hR)
            temp_1 = np.sum(lR[n, :, :], axis=1) / lR.shape[2]
            temp_2 = np.sum(hR[n, :, :], axis=1) / hR.shape[2]
            modDep[0, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_1[51:101]))  # low-rest
            modDep[1, nIdx] = -(np.mean(temp_2[0:50]) - np.mean(temp_2[51:101]))  # high-rest
            modDep[2, nIdx] = -(np.mean(temp_1[51:101]) - np.mean(temp_2[51:101]))  # high-low
            modDep[3, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_2[0:50]))  # high rest - low rest

            # Process RLPP data (lRp and hRp)
            temp_1 = np.sum(lRp[n, :, :], axis=1) / lRp.shape[2]
            temp_2 = np.sum(hRp[n, :, :], axis=1) / hRp.shape[2]
            modDep[4, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_1[51:101]))  # low-rest
            modDep[5, nIdx] = -(np.mean(temp_2[0:50]) - np.mean(temp_2[51:101]))  # high-rest
            modDep[6, nIdx] = -(np.mean(temp_1[51:101]) - np.mean(temp_2[51:101]))  # high-low
            modDep[7, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_2[0:50]))  # high rest - low rest

            # Process SLPP data (lRsp and hRsp)
            temp_1 = np.sum(lRsp[n, :, :], axis=1) / lRsp.shape[2]
            temp_2 = np.sum(hRsp[n, :, :], axis=1) / hRsp.shape[2]
            modDep[8, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_1[51:101]))  # low-rest
            modDep[9, nIdx] = -(np.mean(temp_2[0:50]) - np.mean(temp_2[51:101]))  # high-rest
            modDep[10, nIdx] = -(np.mean(temp_1[51:101]) - np.mean(temp_2[51:101]))  # high-low
            modDep[11, nIdx] = -(np.mean(temp_1[0:50]) - np.mean(temp_2[0:50]))  # high rest - low rest

    # Save the results to a .mat file
    sio.savemat('./Python_Ver/trained_results/modDep.mat', {'modDep': modDep})
