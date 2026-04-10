import numpy as np
import os
from scipy.io import loadmat

from preprocess import *
from utils.spikeTime2Train import spikeTime2Train
from scipy.io import savemat

# Define paths
DATA_INDEX = 'rat01'
DATA_INDEX_SHORT = '01'
ORIGIN_DATA = f'./Python_Ver/raw_data/{DATA_INDEX}.mat'
TARGET_DATA = f'./Python_Ver/data/{DATA_INDEX}.mat'

# Load or preprocess data
if not os.path.exists(TARGET_DATA):
    M1, mPFC, M1num, mPFCnum, M1channelName, mPFCchannelName, segment, trialNo, actions = spikeTime2Train(ORIGIN_DATA)
else:
    data = loadmat(TARGET_DATA)
    M1, mPFC, M1num, mPFCnum = data['M1'], data['mPFC'], data['M1num'].item(), data['mPFCnum']
    segment, trialNo, actions = data['segment'], data['trialNo'], data['actions'][0]

# Rank neurons based on mutual information
_,_,_,M1order = sortM1neurons(M1num, M1, actions)

# Prepare data for decoder training
his = 60  # 600ms history for M1 ensemble
temp = segment[0,(his + 1):] if segment.ndim == 2 else segment[his + 1:]
y = temp[temp > 0]
y_onehot = np.equal(y[:, None], np.arange(1, 4)).astype(int)  # Convert to one-hot encoding

x_full = np.zeros(((his + 1) * M1num, np.sum(temp > 0)))
for i in range(his + 1):
    M1ensemble_temp = M1[(his + 1) - i: -i if i > 0 else None, :].T
    x_full[i * M1num: (i + 1) * M1num, :] = M1ensemble_temp[:, temp > 0]

# Search for best hidden layer size
searchHiddenSize(x_full.T, y_onehot)

# Select optimal hidden size & top-N neurons for decoding
hiddenLayerSize = 2 ** 3
searchM1number(x_full, y_onehot, M1num, M1order, his, hiddenLayerSize)  

# TODO: automatically save the best decoder for RLPP training

# Save preprocessed data
M1num_pre = 2
modelName = f'decodingModel_{DATA_INDEX_SHORT}'

savemat(TARGET_DATA, {
    "DataIndex": DATA_INDEX,
    "his": his,
    "M1": M1,
    "M1num": M1num,
    "M1order": M1order,
    "M1num_pre": M1num_pre,
    "mPFC": mPFC,
    "mPFCnum": mPFCnum,
    "segment": segment,
    "trialNo": trialNo,
    "modelName": modelName,
    "M1channelName": M1channelName,
    "mPFCchannelName": mPFCchannelName,
    "actions": actions
    })