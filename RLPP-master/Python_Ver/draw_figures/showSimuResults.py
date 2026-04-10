import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from utils import gaussianSmooth

# Function to plot spikes (plots nonzero values)
def plot_spikes(ax, x, signal, pos, color):
    spikes = np.where((~np.isnan(signal)) & (signal != 0))[0]
    ax.scatter(x[spikes], signal[spikes]*pos, s=30, color=color, marker='|')

def showSimuResults():
    legendPos = [0.38, 0.95, 0.53, 0.04]
    plotIndex = np.arange(1200, 2400)  
    timeIndex = plotIndex / 100.0

    # ---------------------------
    # RL results
    # ---------------------------
    data_rl = loadmat(r'Python_Ver/results/Simulations_RL_0.mat')
    testM1Truth     = data_rl['testM1Truth']
    pOutputTest      = data_rl['pOutputTest']
    spkOutPredictTest= data_rl['spkOutPredictTest']
    testActions      = data_rl['testActions'].squeeze()
    motorPerformTest = data_rl['motorPerformTest'].squeeze()
    testSucRate      = float(data_rl['testSucRate'].squeeze())

    fig_rl = plt.figure("RLPP results", facecolor='w', figsize=(8, 6))
    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(timeIndex, gaussianSmooth(testM1Truth[0, plotIndex], 10), 'r', linewidth=1.5)
    ax1.plot(timeIndex, pOutputTest[0, plotIndex], color=np.array([54, 56, 131]) / 255, linewidth=1.5)
    plot_spikes(ax1, timeIndex, testM1Truth[0, plotIndex], 1.4, 'r')
    plot_spikes(ax1, timeIndex, spkOutPredictTest[0, plotIndex], 1.2, np.array([54, 56, 131]) / 255)
    ax1.set_ylabel('M1 1')
    ax1.legend(["Simulated M1 firing", "RL-generated M1 firing"], loc='upper left', 
            bbox_to_anchor=(legendPos[0], legendPos[1]), ncol=2)

    ax2 = plt.subplot(3, 1, 2, sharex=ax1)
    ax2.plot(timeIndex, gaussianSmooth(testM1Truth[1, plotIndex], 10), 'r', linewidth=1.5)
    ax2.plot(timeIndex, pOutputTest[1, plotIndex], color=np.array([54, 56, 131]) / 255, linewidth=1.5)
    plot_spikes(ax2, timeIndex, testM1Truth[1, plotIndex], 1.4, 'r')
    plot_spikes(ax2, timeIndex, spkOutPredictTest[1, plotIndex], 1.2, np.array([54, 56, 131]) / 255)
    ax2.set_ylabel('M1 2')

    ax3 = plt.subplot(3, 1, 3, sharex=ax1)
    for act, col in zip([3, 2, 1], ['r', 'y', 'b']):
        temp = testActions[plotIndex].astype(float).copy()
        temp[temp != act] = np.nan
        temp[temp == act] = 1.5
        ax3.plot(timeIndex, temp, col, linewidth=8, solid_capstyle='butt')
    ax3.text(timeIndex[0] + 0.49, 1.7, 'Correct movements')
    for act, col in zip([3, 2, 1], ['r', 'y', 'b']):
        temp = motorPerformTest[plotIndex].astype(float).copy()
        temp[testActions[plotIndex] == 0] = np.nan
        temp[temp != act] = np.nan
        temp[temp == act] = 0.5
        ax3.plot(timeIndex, temp, col, linewidth=8, solid_capstyle='butt')
    ax3.text(timeIndex[0] + 0.49, 0.7, f'Predicted movements from RL outputs (Time-bin success rate: {testSucRate:.2f})')
    ax3.set_ylim([0, 2])
    ax3.legend(["Press High", "Press Low", "rest"], loc='upper left', 
            bbox_to_anchor=(0.43, 0.31), ncol=3)
    ax3.set_xlabel('Time (sec)')
    ax3.set_ylabel('Movements')
    plt.tight_layout()

    # ---------------------------
    # SL results
    # ---------------------------
    data_sl = loadmat(r'Python_Ver/results/Simulations_Sup_0.mat')
    testM1Truth     = data_sl['testM1Truth']
    pOutputTest      = data_sl['pOutputTest']
    spkOutPredictTest= data_sl['spkOutPredictTest']
    testActions      = data_sl['testActions'].squeeze()
    motorPerformTest = data_sl['motorPerformTest'].squeeze()
    testSucRate      = float(data_sl['testSucRate'].squeeze())

    fig_sl = plt.figure("Supervised learning results", facecolor='w', figsize=(8, 6))
    ax1_sl = plt.subplot(3, 1, 1)
    ax1_sl.plot(timeIndex, gaussianSmooth(testM1Truth[0, plotIndex], 10), 'r', linewidth=1.5)
    ax1_sl.plot(timeIndex, pOutputTest[0, plotIndex], color=np.array([103, 146, 70]) / 255, linewidth=1.5)
    plot_spikes(ax1_sl, timeIndex, testM1Truth[0, plotIndex], 1.4, 'r')
    plot_spikes(ax1_sl, timeIndex, spkOutPredictTest[0, plotIndex], 1.2, np.array([103, 146, 70]) / 255)
    ax1_sl.set_ylabel('M1 1')
    ax1_sl.legend(["Simulated M1 firing", "SL-predicted M1 firing"], loc='upper left', 
                bbox_to_anchor=(legendPos[0], legendPos[1]), ncol=2)

    ax2_sl = plt.subplot(3, 1, 2, sharex=ax1_sl)
    ax2_sl.plot(timeIndex, gaussianSmooth(testM1Truth[1, plotIndex], 10), 'r', linewidth=1.5)
    ax2_sl.plot(timeIndex, pOutputTest[1, plotIndex], color=np.array([103, 146, 70]) / 255, linewidth=1.5)
    plot_spikes(ax2_sl, timeIndex, testM1Truth[1, plotIndex], 1.4, 'r')
    plot_spikes(ax2_sl, timeIndex, spkOutPredictTest[1, plotIndex], 1.2, np.array([103, 146, 70]) / 255)
    ax2_sl.set_ylabel('M1 2')

    ax3_sl = plt.subplot(3, 1, 3, sharex=ax1_sl)
    for act, col in zip([3, 2, 1], ['r', 'y', 'b']):
        temp = testActions[plotIndex].astype(float).copy()
        temp[temp != act] = np.nan
        temp[temp == act] = 1.5
        ax3_sl.plot(timeIndex, temp, col, linewidth=8, solid_capstyle='butt')
    ax3_sl.text(timeIndex[0] + 0.49, 1.7, 'Correct movements')
    for act, col in zip([3, 2, 1], ['r', 'y', 'b']):
        temp = motorPerformTest[plotIndex].astype(float).copy()
        temp[testActions[plotIndex] == 0] = np.nan
        temp[temp != act] = np.nan
        temp[temp == act] = 0.5
        ax3_sl.plot(timeIndex, temp, col, linewidth=8, solid_capstyle='butt')
    ax3_sl.text(timeIndex[0] + 0.49, 0.7, f'Predicted movements from SL outputs  (Time-bin success rate: {testSucRate:.2f})')
    ax3_sl.set_ylim([0, 2])
    ax3_sl.legend(["Press High", "Press Low", "rest"], loc='upper left', 
                bbox_to_anchor=(0.43, 0.31), ncol=3)
    ax3_sl.set_xlabel('Time (sec)')
    ax3_sl.set_ylabel('Movements')
    plt.tight_layout()
    plt.show()
