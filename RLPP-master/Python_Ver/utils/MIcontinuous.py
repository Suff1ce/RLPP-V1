import numpy as np
from scipy.stats import gaussian_kde

def MIcontinuous(signal1, signal2):
    """
    Calculate the mutual information between two signals.

    Parameters:
    signal1, signal2 : 1D arrays
        The time series data vectors.

    Returns:
    MI : float
        The mutual information between signal1 and signal2.
    """
    # Define edges for histogram bins
    edges1 = np.linspace(np.min(signal1), np.max(signal1), 21)
    edges2 = np.linspace(np.min(signal2), np.max(signal2), 21)

    # Compute joint histogram (probability density)
    p_xy, _, _ = np.histogram2d(signal1, signal2, bins=[edges1, edges2], density=True)
    
    # Compute marginal histograms (probability densities)
    p_x, _ = np.histogram(signal1, bins=edges1, density=True)
    p_y, _ = np.histogram(signal2, bins=edges2, density=True)
    
    # Calculate mutual information using the formula
    p_x = p_x[:, None]  # Make p_x a column vector (broadcasting)
    p_y = p_y[None, :]  # Make p_y a row vector (broadcasting)

    MI = np.nansum(p_xy * np.log2(p_xy / (p_x * p_y)))

    return MI
