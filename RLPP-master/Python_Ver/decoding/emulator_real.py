import numpy as np
# from decoding import *
from .decodingModel_01 import decodingModel_01
from .decodingModel_02 import decodingModel_02
from .decodingModel_manual import decodingModel_manual

def emulator_real(spikes, motorExpect, indexes, his, modelName):
    """
    Emulates motor performance based on M1 spike data.

    Parameters:
    spikes : ndarray
        M1 spike activity (neuronal firing).
    motorExpect : ndarray
        Expected motor output.
    indexes : ndarray
        Index order for M1 neurons.
    his : int
        Number of historical time steps to consider.
    modelName : str
        Name of the model function (should be callable in the current scope).

    Returns:
    success : ndarray
        Array indicating success (1 if match, NaN if motorExpect is 0).
    rate : float
        Success rate (proportion of correct predictions).
    motorPerform : ndarray
        Predicted motor performance.
    ensemble : ndarray
        Constructed spike ensemble input for the model.
    """

    # Rearrange M1 order
    sortedIndices = np.argsort(indexes)
    spikes = spikes[sortedIndices, :]

    # Get M1 spike ensemble
    M1num, timeLength = spikes.shape
    ensemble = np.zeros(((his + 1) * M1num, timeLength))

    for i in range(his + 1):
        ensemble[i * M1num:(i + 1) * M1num, :] = np.hstack( [np.zeros((M1num, i)), spikes[:, :-i] if i > 0 else spikes])

    # Feedforward to get behavior
    if modelName == 'decodingModel_manual':
        motorModel = decodingModel_manual
    elif modelName == 'decodingModel_01':
        motorModel = decodingModel_01
    elif modelName == 'decodingModel_02':
        motorModel = decodingModel_02
    else:
        raise ValueError(f'Unknown model name: {modelName}')
    y = motorModel(ensemble)
    motorPerform = np.argmax(y, axis=0) + 1  

    # Compute reward
    success = (motorPerform == motorExpect).astype(float)
    success[motorExpect == 0] = np.nan
    rate = np.nansum(success) / np.sum(~np.isnan(success))

    return success, rate, motorPerform, ensemble
