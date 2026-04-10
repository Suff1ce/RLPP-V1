import numpy as np

def plotModulation(ax, plot_R, plot_Rp, plot_Rsp, modelColor, actionsColor, ymin, ymax, h):
    """
    Plots modulation curves with shaded action areas.

    Parameters:
    plot_R : array-like
        First set of modulation data.
    plot_Rp : array-like
        Second set of modulation data.
    plot_Rsp : array-like
        Third set of modulation data.
    modelColor : array-like
        List of colors for different curves.
    actionsColor : array-like
        Colors for action shading.
    ymin : float
        Minimum Y-axis limit.
    ymax : float
        Maximum Y-axis limit.
    h : float
        Height reference for shaded action areas.
    """
    x1 = np.arange(-0.5, 0.01, 0.01)  # -0.5 to 0 step 0.01 (51 points)
    x2 = np.arange(0.01, 0.51, 0.01)  # 0.01 to 0.5 step 0.01 (50 points)
    x1_area = np.arange(-0.5, 0.01, 0.01)  # -0.5 to 0 (51 points)
    x2_area = np.arange(0, 0.51, 0.01)     # 0 to 0.5 (51 points)

    # Fill areas
    ax.fill_between(x1_area, h, color=actionsColor[0], alpha=0.3, edgecolor='none')
    ax.fill_between(x2_area, h, color=actionsColor[1], alpha=0.3, edgecolor='none')

    # Plot lines with styles
    l1 = ax.plot(x1, plot_R[:51], color=modelColor[0], linewidth=2, linestyle=':')
    ax.plot(x2, plot_R[51:], color=modelColor[0], linewidth=2, linestyle=':')

    l3 = ax.plot(x1, plot_Rsp[:51], color=modelColor[2], linewidth=2, linestyle='-.')
    ax.plot(x2, plot_Rsp[51:], color=modelColor[2], linewidth=2, linestyle='-.')

    l2 = ax.plot(x1, plot_Rp[:51], color=modelColor[1], linewidth=1.5)
    ax.plot(x2, plot_Rp[51:], color=modelColor[1], linewidth=1.5)

    ax.set_ylim(ymin, ymax)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return l1[0], l2[0], l3[0]

