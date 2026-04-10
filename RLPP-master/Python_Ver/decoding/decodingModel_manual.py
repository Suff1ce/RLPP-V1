import numpy as np

def decodingModel_manual(M1ensemble):
    """
    Manually designed tuning function to decode movements based on M1 neuron firing rates.

    Parameters:
        M1ensemble : numpy.ndarray
            2D array of shape (4N, T) representing M1 neuron activity, where neurons are grouped in sets of 4.

    Returns:
        decoded_movements : numpy.ndarray
            2D array of shape (3, T) containing:
                Row 1 -> Constant baseline of 0.3
                Row 2 -> Difference between Neuron 1 and Neuron 3 firing rates
                Row 3 -> Difference between Neuron 2 and Neuron 4 firing rates
    """
    # Compute the firing rate of each M1 neuron group (averaging every 4th row)
    M1_fr = np.array([np.mean(M1ensemble[i::4, :], axis=0) for i in range(4)])

    # Construct decoded movements output
    decoded_movements = np.vstack([
        0.3 * np.ones(M1_fr.shape[1]),  # Baseline value
        M1_fr[0, :] - M1_fr[2, :],      # Difference between Neuron 1 and Neuron 3
        M1_fr[1, :] - M1_fr[3, :]       # Difference between Neuron 2 and Neuron 4
    ])

    return decoded_movements
