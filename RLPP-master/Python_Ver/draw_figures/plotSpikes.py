import numpy as np

def plotSpikes(ax, index, spikes, pos, color):
    """
    Plots spike raster using vertical lines.

    Parameters:
    index : array-like
        X-axis indices for plotting.
    spikes : array-like
        Binary spike data (1 for spike, 0 for no spike).
    pos : float
        Vertical position multiplier for spikes.
    color : str or tuple
        Color of the spike markers.
    """
    x = np.where(spikes, pos, np.nan)  # Replace 0s with NaN to avoid plotting them
    ax.plot(index, x, "|", color=color, markersize=6)
