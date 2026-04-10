import numpy as np
from .decodingModel_simulation import decodingModel_simulation

def emulator_simu(spikes, motorExpect, indexes, his, modelName):
    """
    Simulates the motor response based on neural spikes.

    Parameters:
    spikes : ndarray
        The spike train data.
    motorExpect : ndarray
        The expected motor output.
    indexes : ndarray
        Index order for rearranging M1 neurons.
    his : any
        History variable used by the model.
    modelName : str
        Name of the model function (should be callable in the current scope).

    Returns:
    success : ndarray
        Binary success indicators (1 for success, 0 for failure, NaN where undefined).
    rate : float
        Success rate, excluding NaN values.
    motorPerform : ndarray
        The predicted motor performance.
    """
    # Rearrange M1 order
    sortedIndices = np.argsort(indexes)
    spikes = spikes[sortedIndices, :]

    # Feedforward to get behavior
    if modelName == 'decodingModel_simulation':        # Retrieve function by name
        motorModel = decodingModel_simulation
    else:
        raise ValueError(f'Unknown model name: {modelName}')
    motorPerform = motorModel(spikes, his)

    # Get reward
    success = (motorPerform == motorExpect).astype(float)
    success[motorExpect == 0] = np.nan  # Set undefined cases to NaN
    
    validSuccess = success[~np.isnan(success)]  # Exclude NaN values
    rate = np.sum(validSuccess) / len(validSuccess) if len(validSuccess) > 0 else np.nan
    
    return success, rate, motorPerform
