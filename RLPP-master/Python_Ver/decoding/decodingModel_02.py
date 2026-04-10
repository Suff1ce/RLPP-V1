import numpy as np
from scipy.io import loadmat
import os

def decodingModel_02(X):
    # Format Input Arguments
    isCellX = isinstance(X, list)
    if not isCellX:
        X = [X]
    
    # Dimensions
    TS = len(X)  # timesteps
    Q = X[0].shape[1] if X else 0  # samples/series
    
    # Allocate Outputs
    Y = [None] * TS
    
    # Load required parameters from .mat files
    x1_step1, b1, IW1_1, b2, LW2_1 = load_params()
    
    # Time loop
    for ts in range(TS):
        # Input 1
        Xp1 = mapminmax_apply(X[ts], x1_step1)
        
        # Layer 1
        a1 = tansig_apply(np.tile(b1[:, None], (1, Q)) + IW1_1 @ Xp1)
        
        # Layer 2
        a2 = softmax_apply(np.tile(b2[:, None], (1, Q)) + LW2_1 @ a1)
        
        # Output 1
        Y[ts] = a2
    
    # Final Delay States
    Xf = []
    Af = [[], []]
    
    # Format Output Arguments
    if not isCellX:
        Y = np.hstack(Y)
    
    return Y

def load_params():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    params = loadmat(os.path.join(dir_path, 'decodingModel_02_para.mat'))
    x1_step1 = params["x1_step1"]
    new_x1_step1 = {
        "xoffset": x1_step1["xoffset"].item().flatten().astype(float),
        "gain": x1_step1["gain"].item().flatten().astype(float),
        "ymin": x1_step1["ymin"].item().item()
    }
    b1 = params["b1"].flatten().astype(float)
    IW1_1 = params["IW1_1"].astype(float)
    b2 = params["b2"].flatten().astype(float)
    LW2_1 = params["LW2_1"].astype(float)
    return new_x1_step1, b1, IW1_1, b2, LW2_1

def mapminmax_apply(x, settings):
    y = (x - settings["xoffset"][:, None]) * settings["gain"][:, None] + settings["ymin"]
    return y

def softmax_apply(n):
    nmax = np.max(n, axis=0, keepdims=True)
    n = n - nmax
    numerator = np.exp(n)
    denominator = np.sum(numerator, axis=0, keepdims=True)
    denominator[denominator == 0] = 1
    return numerator / denominator

def tansig_apply(n):
    return 2 / (1 + np.exp(-2 * n)) - 1
