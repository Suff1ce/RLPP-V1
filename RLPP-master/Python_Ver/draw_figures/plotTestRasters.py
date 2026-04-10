import numpy as np
import matplotlib.pyplot as plt


def plotOneModel(ax, raster, base, color):
    """
    Plots spike raster for a single model.

    Parameters:
    raster : ndarray
        Spike raster data for trials.
    base : float
        Base y-position for the plot.
    color : tuple
        Color of the plot.
    """
    for rasterIdx in range(raster.shape[0]):
        x = np.where(raster[rasterIdx, :], base + rasterIdx * 0.001, np.nan)
        ax.plot(x, "|", color=color, markersize=0.5)


def plotTestRasters(ax, R, Rp, Rsp, neuronIdx, color):
    """
    Plots raster test data for different models.

    Parameters:
    R : ndarray
        Raster data for the real test set.
    Rp : ndarray
        Raster data for the RL predicted test set.
    Rsp : ndarray
        Raster data for the supervised predicted test set.
    neuronIdx : int
        Index of the neuron to plot.
    color : list of tuples
        Colors for different model plots.
    """
    plotTrialNo = min(R.shape[1], 20)

    plotOneModel(ax, R[neuronIdx, :plotTrialNo, :], 1.03, color[0])
    plotOneModel(ax, Rp[neuronIdx, :plotTrialNo, :], 1.00, color[1])
    plotOneModel(ax, Rsp[neuronIdx, :plotTrialNo, :], 0.97, color[2])

    ax.set_ylim([0.97, 1.05])
    plt.box()
    ax.tick_params(axis='y', colors='white', length=0)
    ax.tick_params(axis='x', colors='white', length=0)
