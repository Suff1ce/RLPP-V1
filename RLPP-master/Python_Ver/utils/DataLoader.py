import numpy as np

def DataLoader(inputEnsemble, M1_truth, Actions, Trials, opt):
    # Select trial indexes based on mode
    if opt["Mode"] == 'train':
        if opt["DataLoaderCursor"] == 1:  # shuffle when all trials have been trained once
            opt["trainTrials"] = np.random.permutation(opt["trainTrials"])
        start = opt["DataLoaderCursor"]
        stop = min(opt["NumberOfTrainTrials"], opt["DataLoaderCursor"] + opt["batchSize"] - 1)
        opt["DataLoaderCursor"] = (stop % opt["NumberOfTrainTrials"]) + 1  # move the cursor forward
        trialIndexes = opt["trainTrials"][start-1:stop]
    elif opt["Mode"] == 'test':
        if opt["testTrials"].ndim == 1:
            trialIndexes = opt["testTrials"]
        elif opt["testTrials"].ndim == 2 and opt["testTrials"].shape[0] == 1:
            trialIndexes = opt["testTrials"].flatten()
        else:
            print(opt["testTrials"].shape)
            raise ValueError("testTrials should be a 1D array")
    elif opt["Mode"] == 'all':
        trialIndexes = np.arange(1, opt["NumberOfTrainTrials"] + 1)
    
    # Get time indexes for each selected trial
    time_indexes = np.concatenate([np.where(Trials == t)[0] for t in trialIndexes])
    # Expand time indexes with discount length offsets and take unique values
    offsets = np.arange(-opt["discountLength"], 1)
    time_indexes = np.unique(time_indexes[:, None] + offsets).astype(int)
    
    batchInput    = inputEnsemble[:, time_indexes]
    batchM1_truth = M1_truth[:, time_indexes].astype(float)
    batchActions  = Actions[time_indexes]
    
    return batchInput, batchM1_truth, batchActions, opt
