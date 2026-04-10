import numpy as np

def CrossEntropyError(action, p_output, num_of_samples):
    """
    Computes the cross-entropy error.

    Parameters:
    action : ndarray
        The ground truth (actual actions taken).
    p_output : ndarray
        The predicted probabilities.
    num_of_samples : int
        The number of samples.

    Returns:
    float
        The computed cross-entropy error.
    """
    return -np.sum(action * np.log(p_output)) / num_of_samples
