import numpy as np
import matplotlib.pyplot as plt

def selectOneAction(actions, a):
    """
    Pick out one action and process it for visualization.

    Parameters:
    actions : array-like
        The sequence of actions.
    a : int
        The action to be selected.

    Returns:
    x : numpy array
        Processed action sequence for plotting.
    """
    x = (actions == a).astype(float)  # Pick out the specific action

    # Convert 010 into 111 to make it visible in the plot
    single_indexes = np.where((x[:-2] == 0) & (x[1:-1] == 1) & (x[2:] == 0))[0]
    for idx in single_indexes:
        x[idx:idx+3] = [1, 1, 1]

    # Convert 0 values to NaN to avoid plotting them
    x[x == 0] = np.nan
    return x

def plotActions(ax, index, actions, color, xRange):
    """
    Plot action sequences with different colors.

    Parameters:
    index : array-like
        The x-axis indices.
    actions : array-like
        The sequence of actions.
    color : array-like
        A list of colors for different actions.
    xRange : tuple
        The x-axis range for the plot.
    """
    
    for i in range(1, 4):  # Actions 1, 2, and 3
        plot_actions = selectOneAction(actions, i)
        ax.plot(index, plot_actions, color=color[i-1], linewidth=15, solid_capstyle='butt')

    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(xRange)
    plt.box(on=False)
    ax.tick_params(axis='y', colors='white', length=0)
    ax.tick_params(axis='x', colors='white', length=0)
