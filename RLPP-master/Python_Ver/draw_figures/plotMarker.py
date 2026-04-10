import matplotlib.pyplot as plt

def plotMarker(x, y):
    """
    Plot a marker with a horizontal line.

    Parameters:
    x : float
        X-coordinate of the marker.
    y : float
        Y-coordinate of the marker.
    """
    plt.plot([x - 0.08, x + 0.08], [y, y], 'k-', linewidth=1.5)  # Horizontal line
    plt.plot(x, y, '^', color='k', markerfacecolor='k', markersize=8)  # Triangle marker
