import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
from scipy.io import savemat

def simuData():
        
    # Set random seed for reproducibility
    np.random.seed(0)

    # Pre-define storage lists for variable-length data
    rate_input_list = []  # one mPFC neuron
    rate_output_list = []  # three M1 neurons
    segment_list = []
    trialNo_list = []

    mPFCnum = 1
    M1num = 3
    M1order = [1, 2, 3]
    M1num_pre = 2  # only predict the first two

    trialType = 1

    for trialIdx in range(1, 201):  # simulate 200 trials
        startTime = np.random.randint(40, 51)  # start cue time
        responseTime = np.random.randint(50, 61)
        pressTime = np.random.randint(80, 101)

        trialType = 2 if trialType == 1 else 1

        # mPFC firing changes 200 ms ahead of press movement
        mPFC_part = np.concatenate([
            np.full(startTime + 50 + responseTime - 20, 0.05),
            np.full(20 + pressTime, 0.3 + 0.4 * (trialType == 2))
        ])

        # M1 firing patterns
        M1_part = np.vstack([
            np.concatenate([np.full(startTime + 50 + responseTime, 0.1),
                            np.full(pressTime, 0.1 + 0.7 * (trialType == 2))]),
            np.concatenate([np.full(startTime + 50 + responseTime, 0.05),
                            np.full(pressTime, 0.7)]),
            np.full(startTime + 50 + responseTime + pressTime, 0.5)
        ])

        segment_part = np.concatenate([
            np.zeros(startTime), np.ones(50), np.zeros(responseTime),
            np.full(pressTime, 1 + trialType)
        ])

        trialNo_part = np.concatenate([
            np.zeros(startTime), np.full(50, trialIdx), np.zeros(responseTime),
            np.full(pressTime, trialIdx)
        ])

        # Append new data
        rate_input_list.append(mPFC_part)
        rate_output_list.append(M1_part)
        segment_list.append(segment_part)
        trialNo_list.append(trialNo_part)

    # Convert lists to NumPy arrays
    rate_input = np.concatenate(rate_input_list)
    rate_output = np.hstack(rate_output_list)  # Stack horizontally for (3, N) shape
    segment = np.concatenate(segment_list)
    trialNo = np.concatenate(trialNo_list)

    # Apply Gaussian smoothing
    def gaussian_smooth(data, sigma=10):
        return gaussian_filter1d(data, sigma=sigma, axis=-1)

    rate_input = gaussian_smooth(rate_input)
    mPFC = rate_input > np.random.rand(*rate_input.shape)
    rate_output = gaussian_smooth(rate_output)
    M1 = rate_output > np.random.rand(*rate_output.shape)

    modelName = 'decodingModel_simulation'
    his = 16

    # Plotting
    plotIndex = np.arange(1000)
    timeIndex = plotIndex / 100

    fig, axes = plt.subplots(4, 1, figsize=(8, 10))
    fig.patch.set_facecolor('w')

    axes[0].plot(timeIndex, rate_input[plotIndex], 'k', linewidth=1.5)
    axes[0].set_ylabel('mPFC')
    axes[0].set_title("Close the window to see training results.")

    axes[1].plot(timeIndex, rate_output[0, plotIndex], 'k', linewidth=1.5)
    axes[1].set_ylim([0, 0.8])
    axes[1].set_ylabel('M1 1')

    axes[2].plot(timeIndex, rate_output[1, plotIndex], 'k', linewidth=1.5)
    axes[2].set_ylim([0, 0.8])
    axes[2].set_ylabel('M1 2')

    axes[3].plot(timeIndex, segment[plotIndex], 'k', linewidth=1.5)
    axes[3].set_ylabel('movements')

    # Save data to .mat file
    os.makedirs('data', exist_ok=True)  # Ensure the directory exists
    savemat('Python_Ver/data/simulations.mat', {
        'mPFC': mPFC,
        'mPFCnum': mPFCnum,
        'M1': M1.T,
        'M1num': M1num,
        'M1order': M1order,
        'M1num_pre': M1num_pre,
        'segment': segment,
        'trialNo': trialNo,
        'his': his,
        'modelName': modelName,
        'DataIndex': 'Simulations'
    })

    print("Data saved to simulations.mat")

    return fig, axes
    
