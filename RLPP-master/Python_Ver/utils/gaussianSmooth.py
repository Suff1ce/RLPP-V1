from scipy.ndimage import gaussian_filter1d

# Gaussian smoothing function
def gaussianSmooth(signal, sigma):
    return gaussian_filter1d(signal.astype(float), sigma)