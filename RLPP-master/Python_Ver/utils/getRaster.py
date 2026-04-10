import numpy as np

def getRaster(allActions, spikes):
    # Initialize variables
    flag = 0
    idx = 0
    lowRaster = None
    
    # Process lowRaster
    for i in range(1, len(allActions)):
        if allActions[i - 1] == 0 and allActions[i] == 1:
            start = i
        if allActions[i - 1] == 2 and allActions[i] == 0:
            stop = i - 1
            flag = 1
        if flag == 1:
            temp = spikes[:, start:stop+1]
            temp = np.hstack([temp[:, allActions[start:stop+1] > 0], spikes[:, stop:stop + 20]])
            if lowRaster is None:
                lowRaster = temp[:, None, :101]
            else:
                lowRaster = np.concatenate((lowRaster, temp[:, None, :101]), axis=1)
            idx += 1
            flag = 0
    
    # Re-initialize variables for highRaster
    flag = 0
    idx = 0
    highRaster = None
    
    # Process highRaster
    for i in range(1, len(allActions)):
        if allActions[i - 1] == 0 and allActions[i] == 1:
            start = i
        if allActions[i - 1] == 3 and allActions[i] == 0:
            stop = i - 1
            flag = 1
        if flag == 1:
            temp = spikes[:, start:stop+1]
            temp = np.hstack([temp[:, allActions[start:stop+1] > 0], spikes[:, stop:stop + 20]])
            if highRaster is None:
                highRaster = temp[:, None, :101]
            else:
                highRaster = np.concatenate((highRaster, temp[:, None, :101]), axis=1)
            idx += 1
            flag = 0
    
    return lowRaster, highRaster
