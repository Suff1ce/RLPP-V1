import numpy as np
from scipy.io import loadmat
from utils.find_behavior_index import find_behavior_index

def spikeTime2Train(filename):
    # Discretize the original time series from Plexon to the spike trains and behavior labels
    # INPUT:
    #   filename    - path to the original data
    # OUTPUT:
    #   M1          - M1 spike trains. Matrix of TimeLength * M1num
    #   mPFC        - mPFC spike trains. Matrix of TimeLength * NumOfNeurons
    #   M1num       - Number of M1 neurons
    #   mPFCnum     - Number of mPFC neurons
    #   M1channelName   - Char matrix of M1num * 5
    #   mPFCchannelName - Char matrix of M1num * 5
    #   segment     - Label rest(1), high-press(2), and low-press(3) behaviors in successful trials
    #   trialNo     - Label the number index of each segment. The index add 1 per trial (1 rest + 1 press)
    #   actions     - Mark all press-release movements by 1.

    # Load file
    S = loadmat(filename)
    timebins = 0.01  # 10ms time bins

    # Get number of M1 and mPFC neurons and the spike length
    spikeLength = float('inf')
    M1num = 0
    mPFCnum = 0
    for channelNo in range(1, 33):
        for sub in ['a', 'b', 'c', 'd']:  # Maximum of 4 units per channel
            channelName = f'WB{channelNo:02d}{sub}'
            if channelName not in S:
                continue  # no such unit
            if channelNo <= 16:
                M1num += 1
            else:
                mPFCnum += 1
            spikeLength = min(spikeLength, np.ceil(np.max(S[channelName]) / timebins).astype(int))

    # Initialize the spike trains
    M1 = np.zeros((spikeLength, M1num))
    mPFC = np.zeros((spikeLength, mPFCnum))

    # From spike time to spike train
    M1i = 0
    mPFCi = 0  # cursor for units
    M1channelName = np.chararray(M1num, itemsize=5)
    mPFCchannelName = np.chararray(mPFCnum, itemsize=5)
    
    for channelNo in range(1, 33):
        for sub in ['a', 'b', 'c', 'd']:
            channelName = f'WB{channelNo:02d}{sub}'
            if channelName not in S:
                continue
            # Get spike train
            spikeTrain = np.histogram(np.floor(S[channelName] * 1 / timebins), bins=np.arange(0, spikeLength + 1))[0]
            if channelNo <= 16:  # first 16 are M1
                M1[:, M1i] = spikeTrain[:spikeLength]
                M1channelName[M1i] = channelName
                M1i += 1
            else:  # latter 16 are mPFC
                mPFC[:, mPFCi] = spikeTrain[:spikeLength]
                mPFCchannelName[mPFCi] = channelName
                mPFCi += 1

    M1 = (M1 > 0).astype(float)  # convert to binary
    mPFC = (mPFC > 0).astype(float)

    # Event time to event train
    rest_time, press_low_time, press_high_time, press_release = find_behavior_index(
        S['EVT01'], S['EVT02'], S['EVT03'], S['EVT04'], S['EVT05'], S['EVT06'], S['EVT07'], S['EVT08'])

    # Assign values to segment
    segment = np.zeros(spikeLength)
    for r in rest_time:
        segment[int(np.floor(r[0] / timebins)):int(np.floor(r[1] / timebins))] = 1
    for p in press_low_time:
        segment[int(np.floor(p[0] / timebins)):int(np.floor(p[1] / timebins))] = 2
    for p in press_high_time:
        segment[int(np.floor(p[0] / timebins)):int(np.floor(p[1] / timebins))] = 3
    segment = segment[:spikeLength]  # the event recording may be longer than the spike recording

    # Number the trials
    trialNo = np.nan * np.ones(spikeLength)
    trials = 0
    for i in range(1, spikeLength):
        if segment[i] > 0:
            if segment[i] == 1 and segment[i - 1] == 0:  # rising edge trigger
                trials += 1
            trialNo[i] = trials
    trialNo = trialNo[:spikeLength]
    trialNo[trialNo == 0] = np.nan  # the very first trial may have no start part been recorded

    # Assign values to actions
    actions = np.zeros(spikeLength)
    for r in press_release:
        actions[int(np.floor(r[0] / timebins)):int(np.floor(r[1] / timebins))] = 1
    actions = actions[:spikeLength]

    return M1, mPFC, M1num, mPFCnum, M1channelName, mPFCchannelName, segment, trialNo, actions
