import numpy as np

def get_yl(plot_lR, plot_hR, plot_lRp, plot_hRp, plot_lRsp, plot_hRsp, neuron_idx):
    """
    Compute the ymax, ymin, and h values based on the given neuron index.

    Parameters:
    plot_lR, plot_hR, plot_lRp, plot_hRp, plot_lRsp, plot_hRsp (np.array): 2D arrays with neuronal data.
    neuron_idx (int): Index of the neuron to extract data from.

    Returns:
    tuple: (ymax, ymin, h) where ymax is the ceiling-rounded maximum,
           ymin is the floor-rounded minimum, and h is the same as ymax.
    """
    # Stack the selected neuron data and find min/max values
    data = np.vstack([plot_lR[neuron_idx, :], plot_hR[neuron_idx, :], 
                      plot_lRp[neuron_idx, :], plot_hRp[neuron_idx, :], 
                      plot_lRsp[neuron_idx, :], plot_hRsp[neuron_idx, :]])
    
    ymax = np.ceil(np.max(data) * 10) / 10
    ymin = np.floor(np.min(data) * 10) / 10
    h = ymax
    
    return ymax, ymin, h
