import numpy as np
from scipy.ndimage import gaussian_filter1d
from .getRaster import getRaster

# Define the function to get plot modulation
def getPlotModulation(x):
    low_part = np.mean(x[:, :, :51], axis=1)
    high_part = np.mean(x[:, :, 51:], axis=1)
    low_smooth = gaussian_filter1d(low_part, sigma=5, axis=1, mode='nearest')
    high_smooth = gaussian_filter1d(high_part, sigma=5, axis=1, mode='nearest')
    return np.concatenate((low_smooth, high_smooth), axis=1)

def getTestRasters(testActions, testM1_truth, RL, SL):
    # Get rasters for low and high actions
    lR, hR = getRaster(testActions, testM1_truth)
    plot_lR, plot_hR = getPlotModulation(lR), getPlotModulation(hR)

    # Get rasters for RL prediction
    lRp, hRp = getRaster(testActions, RL['spkOutPredictTest'])
    plot_lRp, plot_hRp = getPlotModulation(lRp), getPlotModulation(hRp)

    # Get rasters for SL prediction
    lRsp, hRsp = getRaster(testActions, SL['spkOutPredictTest'])
    plot_lRsp, plot_hRsp = getPlotModulation(lRsp), getPlotModulation(hRsp)

    return lR, hR, lRp, hRp, lRsp, hRsp, plot_lR, plot_hR, plot_lRp, plot_hRp, plot_lRsp, plot_hRsp