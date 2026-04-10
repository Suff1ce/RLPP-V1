import numpy as np
from scipy.ndimage import uniform_filter1d

def decodingModel_simulation(spk, his):
    """
    Decode motor actions based on the moving-average firing rate of two M1 neurons.
    
    Parameters:
        spk : numpy.ndarray
            2D array of spike data with shape (2, T) for two M1 neurons.
        his : int
            The history window (number of time bins) used for the moving average.
    
    Returns:
        motor : numpy.ndarray
            1D array of motor actions (length T) determined as:
                1 -> rest (both neurons firing below 0.25)
                2 -> press low (M1_2 firing >= 0.25 and M1_1 firing <= 0.25)
                3 -> press high (both neurons firing high; M1_2 >= 0.25 and M1_1 > 0.25)
    """
    # Compute the moving average firing rate for each M1 neuron using a trailing window
    spk = spk.astype(float)
    M1_1_mean = uniform_filter1d(spk[0, :], size=his, mode='nearest')
    M1_2_mean = uniform_filter1d(spk[1, :], size=his, mode='nearest')
    
    # Determine motor action:
    motor = ((M1_2_mean < 0.25) & (M1_1_mean < 0.25)).astype(int) * 1 + \
            ((M1_2_mean >= 0.25) & (M1_1_mean <= 0.25)).astype(int) * 2 + \
            ((M1_2_mean >= 0.25) & (M1_1_mean > 0.25)).astype(int) * 3
    
    return motor
