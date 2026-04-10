import numpy as np

def X_PART(x, i, n, h):
    """
    Take corresponding ensemble with given M1 indexes.
    
    Parameters:
    x : numpy.ndarray
        Full ensemble (2D array).
    i : numpy.ndarray
        Required M1 indexes (1D array).
    n : int
        Total M1 numbers.
    h : int
        Additional parameter for reshaping the indexes.
    
    Returns:
    numpy.ndarray
        Corresponding ensemble.
    """
    
    # Ensure 'i' is a column vector (transpose if necessary)
    if i.ndim == 1:
        i = i[:, np.newaxis]
    
    # Reshape the indices and sort them, then select corresponding rows
    indices = np.sort((i + n * np.arange(0, h + 1)).reshape(-1, 1))
    ensemble = x[indices.flatten(), :]
    
    return ensemble
