import numpy as np

def PreProcess(mPFC, M1, segment, trialNo, decay_parameter, opt):
    # Get spike time train for each input channel (convert to 1-indexed)
    mPFC_Time = []
    for ch in range(opt["mPFCchannels"]):
        spikes = np.nonzero(mPFC[ch, :])[0] + 1
        mPFC_Time.append(spikes)
    
    # Get start position of dataset
    # Find the time of n+1 spike (n is the number of relevant spikes) of each input channel, then use the max as start point
    start_candidates = [spikes[ opt["RelevantSpikes"] ] + 1 for spikes in mPFC_Time]
    Start = int(max(start_candidates))
    
    # Get output data
    M1_truth = M1[np.array(opt["M1index"]) - 1, Start - 1:]
    Actions  = segment[Start - 1:]
    Trials   = trialNo[Start - 1:]
    
    # Get each input channel's ensemble of three sets
    Input_hist = History(mPFC_Time, M1.shape[1], opt)       # get ensembles
    spikeEnsemble = []                          # initial each set input
    for ch in range(opt["mPFCchannels"]):       # get input for each channel & each set
        spikeEnsemble.append(Input_hist[ch][:, Start - 1:]) 
    inputEnsemble = np.vstack([np.exp(-ensemble / decay_parameter) for ensemble in spikeEnsemble])  # delete cell2mat if need cell format variable
    
    # Get train / test trial indexes
    Trials = Trials.astype(float)
    Trials[Trials <= 1] = np.nan  # sometimes the first few trials do not have enough history for prediction
    AllTrialIndexes = np.unique(Trials[~np.isnan(Trials)])
    opt["NumberOfAllTrials"] = len(AllTrialIndexes)
    AllTrialIndexes = np.random.permutation(AllTrialIndexes)
    
    foldTrialNum = int(np.floor(opt["NumberOfAllTrials"] / opt["foldNum"]))
    fold_start = np.arange(0,  opt["foldNum"] * foldTrialNum , foldTrialNum)
    fold_stop = np.concatenate((fold_start[1:] - 1, [ opt["foldNum"] * foldTrialNum - 1 ]))
    opt["folds"] = []
    opt["FoldTrialNumber"] = np.zeros(opt["foldNum"], dtype=int)
    for i in range(opt["foldNum"]):
        fold = AllTrialIndexes[fold_start[i] : fold_stop[i]]
        opt["folds"].append(fold)
        opt["FoldTrialNumber"][i] = len(fold)
    opt["folds"] = np.array(opt["folds"])   
    return inputEnsemble, M1_truth, Actions, Trials, opt


def History(xt, Ny, opt):
    '''
    % Function History
    % get ensemble of spike time
    %    Input: xt - spike time train {channelNumber, 1}(1, time of m th spike)
    %           Ny - total length of time bins of output signal
    %           opt - opt.mPFCchannels: input channel number
    %                 opt.RelevantSpikes: number of history length (number of past spikes suppossed to be relevant to current spike)
    %    Output: x1 - input history ensemble {channelNumber, 1}(opt.RelevantSpikes+1-i th nearest spike from kth time bin)
    %    Algorithm: Given m th spike time t_m and t_m+1 the spike time m+1,
    %               from timebins t_m to (t_m+1)-1, the past relevant spikes are
    %               same.
    %               Thus, first take past relevant spike time to fill in the
    %               ensemble t_m to (t_m+1)-1, then use current time deduct the
    %               past spike time, we get the ensemble
    %    Notes: The code simply use time bins t_m to t_m+1, because the t_m+1
    %           will be rewrite in next iteration. As a result, although the
    %           code is not same as the algorithm, but have same results
    '''
    mPFCchannels = opt["mPFCchannels"]
    RelevantSpikes = int(opt["RelevantSpikes"])
    Ensemble = [len(x) - RelevantSpikes for x in xt]    # Ensemble is (number of spikes - relevant spike number) for each input channel
    x1 = [np.full((RelevantSpikes, Ny), np.inf) for _ in range(mPFCchannels)] # Create a history ensemble for each input channel. Initial the x1
    
    for j in range(mPFCchannels):   # For each channel
        spikes = xt[j]  
        # For time bin 1 to time bin xt{jndex,1}(opt.RelevantSpikes), no enough past spikes are avilible, so do some special process
        for index in range(1, RelevantSpikes):
            t_start = int(spikes[index - 1])
            t_end   = int(spikes[index])
            num_cols = t_end - t_start + 1
            rep_val = np.tile(spikes[:index].reshape((index, 1)), (1, num_cols))
            x1[j][0:index, t_start - 1:t_end] = rep_val
        # Deal with the ensemble part. all have enough relevant spike history
        for index in range(0, Ensemble[j] ):    # Traverse all ensemble indexes, same way as above, instead of a slide window
            t_start = int(spikes[index + RelevantSpikes - 1])
            t_end   = int(spikes[index + RelevantSpikes])
            num_cols = t_end - t_start + 1
            rep_val = np.tile(spikes[index :index + RelevantSpikes ].reshape((RelevantSpikes, 1)), (1, num_cols))
            x1[j][:, t_start - 1:t_end] = rep_val
        # Fill in the rest time bins, since index=Ensemble and there is no more xt{jndex,1}(index+1)
        t_start = int(spikes[-1])
        num_cols = Ny - t_start + 1
        rep_val = np.tile(spikes[-RelevantSpikes:].reshape((RelevantSpikes, 1)), (1, num_cols))
        x1[j][:, t_start - 1:Ny] = rep_val
        # current time index - past spike time index to get duration
        time_matrix = np.tile(np.arange(1, Ny + 1), (RelevantSpikes, 1))
        x1[j] = time_matrix - x1[j]
    
    return x1
