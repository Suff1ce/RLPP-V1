# %% show time domain of rat01, test set
import numpy as np
import scipy.io as sio
import os
import random
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import wilcoxon

from draw_figures import *
from utils import *
from model import *
from decoding import *

# Load MATLAB .mat file
folds =  sio.loadmat("trained_results/rat01_RL_results.mat")["folds"][0]

# %% define color
modelColor = np.array([[250, 0, 0], [54, 56, 131], [103, 146, 70]]) / 255
actionsColor = np.array([[178, 178, 178], [ 255, 165, 0], [ 128, 0, 128]]) / 255

# %% load data
DataName = "./Python_Ver/data/rat01.mat"
# DataName = "data/rat02.mat"
data = data_setup(DataName)  # Load data
opt = opt_Setup_real(data)

random.seed(0)  # Equivalent to rng('default') in MATLAB

inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
    data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
)

opt["testFold"] = 5
trainFolds = list(range(1, opt["foldNum"] + 1))
trainFolds.remove(opt["testFold"])  # Exclude test fold

opt["trainTrials"] = np.concatenate([folds[i] for i in trainFolds])
opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
opt["testTrials"] = folds[opt["testFold"] - 1]
opt["NumberOfTestTrials"] = len(opt["testTrials"])

opt["Mode"] = "test"
testInput, testM1_truth, testActions, opt = DataLoader(
    inputEnsemble, M1_truth, Actions, Trials, opt
)
TestSamples = testInput.shape[1]
inputUnitTest = np.vstack([testInput, np.ones((1, TestSamples))])

# %% Get RL outputs
mat_data = sio.loadmat("./Python_Ver/trained_results/rat01_RL_results.mat")

L1WeightBestReward = mat_data["L1WeightBestReward"]
L2WeightBestReward = mat_data["L2WeightBestReward"]
MaxRewardEpisode = mat_data["MaxRewardEpisode"]

pOutputTest, _, spkOutPredictTest = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, TestSamples,
    MaxRewardEpisode, opt["prioriM"], opt["prioriN"]
)

success, testSucRate, motor_perform_test, _ = emulator_real(
    spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"]
)

RL = {
    "motor_perform_test": motor_perform_test,
    "pOutputTest": pOutputTest,
    "spkOutPredictTest": spkOutPredictTest,
    "testSucRate": testSucRate,
    "success": success
}

# %% Get SL outputs
mat_data = sio.loadmat("./Python_Ver/trained_results/rat01_Sup_results.mat")
L1Weight = mat_data["L1Weight"]
L2Weight = mat_data["L2Weight"]

pOutputTest, _, spkOutPredictTest = applynets(
    inputUnitTest, L1Weight, L2Weight, TestSamples
)

success, testSucRate, motor_perform_test, _ = emulator_real(
    spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"]
)

SL = {
    "motor_perform_test": motor_perform_test,
    "pOutputTest": pOutputTest,
    "spkOutPredictTest": spkOutPredictTest,
    "testSucRate": testSucRate,
    "success": success
}

# %% Calculate Success rate
print(f'RL time-bin success rate of Rat 01: {RL["testSucRate"]}')
print(f'RL trial success rate of Rat 01: {calculateTrialSucRate(RL["success"], testActions)}\n')
print(f'SL time-bin success rate of Rat 01: {SL["testSucRate"]}')
print(f'SL trial success rate of Rat 01: {calculateTrialSucRate(SL["success"], testActions)}')

# %% Draw plots
fig = plt.figure(figsize=(15 / 2.54, 17 / 2.54))  # Convert cm to inches
fig.patch.set_facecolor('white')
timeIndex = np.arange(0.01, 20.01, 0.01)
timeRange = [0.01, 20]
dataIndex = np.arange(2000)



'''
Neural data
''' 
# %% Neuron 1
# Spikes
fig.text(0, 0.85, 'a(i)', fontsize=8, fontweight='bold')

ax1 = fig.add_axes([0.1, 0.78, 0.87, 0.09])
plotSpikes(ax1, timeIndex, testM1_truth[0, dataIndex], 1.1, modelColor[0])
plotSpikes(ax1, timeIndex, RL["spkOutPredictTest"][0, dataIndex], 1, modelColor[1])
plotSpikes(ax1, timeIndex, SL["spkOutPredictTest"][0, dataIndex], 0.9, modelColor[2])
ax1.set_yticks([])
ax1.set_xticks([])
ax1.set_ylim([0.8, 1.2])
ax1.set_xlim(timeRange)
ax1.set_title("M1 Neuron 1", fontsize=10, fontweight="bold")

ax1.set_frame_on(False)

# Firing probabilities
fig.text(0, 0.785, 'a(ii)', fontsize=8, fontweight='bold')

ax2 = fig.add_axes([0.1, 0.67, 0.87, 0.10])
ax2.plot(timeIndex, gaussian_filter1d(testM1_truth[0, dataIndex], sigma=10), color=modelColor[0])
ax2.plot(timeIndex, RL["pOutputTest"][0, dataIndex], color=modelColor[1])
ax2.plot(timeIndex, SL["pOutputTest"][0, dataIndex], color=modelColor[2])
ax2.set_xlabel("Time (s)", fontsize=8)
ax2.set_ylabel(r"${\rm Firing ~Probability}$", fontsize=6, rotation=90)
ax2.set_xlim(timeRange)
ax2.legend(["Recordings", "RLPP", "SLPP"], loc="upper right", fontsize=8, frameon=False)

# %% Neuron 2
# Spikes
fig.text(0, 0.52, 'a(iii)', fontsize=8, fontweight='bold')

ax3 = fig.add_axes([0.1, 0.45, 0.87, 0.09])
plotSpikes(ax3, timeIndex, testM1_truth[1, dataIndex], 1.1, modelColor[0])
plotSpikes(ax3, timeIndex, RL["spkOutPredictTest"][1, dataIndex], 1, modelColor[1])
plotSpikes(ax3, timeIndex, SL["spkOutPredictTest"][1, dataIndex], 0.9, modelColor[2])
ax3.set_yticks([])
ax3.set_xticks([])
ax3.set_ylim([0.8, 1.2])
ax3.set_xlim(timeRange)
ax3.set_title("M1 Neuron 2", fontsize=10, fontweight="bold")

ax3.set_frame_on(False)

# Firing probabilities
fig.text(0, 0.455, 'a(iv)', fontsize=8, fontweight='bold')

ax4 = fig.add_axes([0.1, 0.34, 0.87, 0.10])
ax4.plot(timeIndex, gaussian_filter1d(testM1_truth[1, dataIndex], sigma=10), color=modelColor[0])
ax4.plot(timeIndex, RL["pOutputTest"][1, dataIndex], color=modelColor[1])
ax4.plot(timeIndex, SL["pOutputTest"][1, dataIndex], color=modelColor[2])
ax4.set_xlabel("Time (s)", fontsize=8)
ax4.set_ylabel(r"${\rm Firing ~Probability}$", fontsize=6, rotation=90)
ax4.set_xlim(timeRange)





'''
Movements
'''
# %% Real Movements
fig.text(0.01, 0.21, 'b', fontsize=8, fontweight='bold')


ax5 = fig.add_axes([0.1, 0.16, 0.87, 0.03])
plotActions(ax5, timeIndex, testActions[dataIndex], actionsColor, timeRange)
legend = ax5.legend(["Rest", "Press Low", "Press High"], loc="upper center", fontsize=8, ncol=3, frameon=False, bbox_to_anchor=(0.8, 2.4))
lines = legend.get_lines()
for line in legend.get_lines():
    line.set_linewidth(10)
ax5.text(0.2, 1, "Recordings", fontsize=8, color=modelColor[0], 
         verticalalignment="center", horizontalalignment="center")

# %% RLPP movements
ax6 = fig.add_axes([0.1, 0.12, 0.87, 0.03])
motor_perform = RL["motor_perform_test"].copy()
motor_perform[testActions == 0] = 0
plotActions(ax6, timeIndex, motor_perform[dataIndex], actionsColor, timeRange)
ax6.text(0.2, 1, "RLPP", fontsize=8, color=modelColor[1], 
         verticalalignment="center", horizontalalignment="center")

# %% SLPP movements
ax7 = fig.add_axes([0.1, 0.08, 0.87, 0.03])
motor_perform = SL["motor_perform_test"].copy()
motor_perform[testActions == 0] = 0
plotActions(ax7, timeIndex, motor_perform[dataIndex], actionsColor, timeRange)
ax7.text(0.2, 1, "SLPP", fontsize=8, color=modelColor[2], 
         verticalalignment="center", horizontalalignment="center")

# %% X-axis
ax8 = fig.add_axes([0.1, 0.06, 0.87, 0.03])
ax8.plot(timeIndex, np.zeros(len(timeIndex)), 'k')
ax8.set_yticklabels([])
ax8.set_yticks([])
ax8.set_ylim([0, 1])
ax8.set_xticks(np.arange(0, 21, 2))
ax8.set_xlim(timeRange)
ax8.set_xlabel("Time (s)", fontsize=8, color='k')
ax8.set_ylabel(r"${\rm Movements}$", fontsize=6.5, rotation=90, color='k')
plt.box(on=False)




'''
calculate success rate
'''
# %% Calculate successRate
success_rate_file = "./Python_Ver/trained_results/successRate.mat"
if not os.path.exists(success_rate_file):
    DataNameList = ['rat01', 'rat02', 'rat03', 'rat04', 'rat05', 'rat06']

    testSucRateRL = np.zeros((6, 5))
    testSucRateSL = np.zeros((6, 5))
    testTrialSucRateRL = np.zeros((6, 5))
    testTrialSucRateSL = np.zeros((6, 5))

    for dataIdx in range(6):
        for i in range(5):
            # RL results
            mat_data = sio.loadmat(f'./Python_Ver/results/{DataNameList[i]}_RL_{i+1}.mat')
            testSucRateRL[dataIdx, i] = mat_data["testSucRate"].item()
            
            _, _, spkOutPredictTest = applynets_priori(
                inputUnitTest, L1WeightBestReward, L2WeightBestReward, 
                TestSamples, MaxRewardEpisode, opt["prioriM"], opt["prioriN"]
            )
            
            success, _, _, _ = emulator_real(spkOutPredictTest, testActions, 
                                          opt["M1index"], 60, 
                                          f'decodingModel_{DataNameList[i][3:5]}')
            
            testTrialSucRateRL[dataIdx, i] = calculateTrialSucRate(success, testActions)

            # SL results
            mat_data = sio.loadmat(f'./Python_Ver/results/{DataNameList[i]}_Sup_{i+1}.mat')
            testSucRateSL[dataIdx, i] = mat_data["testSucRate"].item()
            
            pOutputTest, _, spkOutPredictTest = applynets(
                inputUnitTest, L1Weight, L2Weight, TestSamples
            )
            
            success, _, _, _ = emulator_real(spkOutPredictTest, testActions, 
                                          opt["M1index"], 60, 
                                          f'decodingModel_{DataNameList[i][3:5]}')
            
            testTrialSucRateSL[dataIdx, i] = calculateTrialSucRate(success, testActions)

    testSucRate_mean = np.column_stack((testSucRateRL.mean(axis=1), testSucRateSL.mean(axis=1)))
    testSucRate_pos = np.column_stack((testSucRateRL.max(axis=1) - testSucRateRL.mean(axis=1),
                                       testSucRateSL.max(axis=1) - testSucRateSL.mean(axis=1)))
    testSucRate_neg = -np.column_stack((testSucRateRL.min(axis=1) - testSucRateRL.mean(axis=1),
                                        testSucRateSL.min(axis=1) - testSucRateSL.mean(axis=1)))
    testTrialSucRate_mean = np.column_stack((testTrialSucRateRL.mean(axis=1), testTrialSucRateSL.mean(axis=1)))
    testTrialSucRate_pos = np.column_stack((testTrialSucRateRL.max(axis=1) - testTrialSucRateRL.mean(axis=1),
                                            testTrialSucRateSL.max(axis=1) - testTrialSucRateSL.mean(axis=1)))
    testTrialSucRate_neg = -np.column_stack((testTrialSucRateRL.min(axis=1) - testTrialSucRateRL.mean(axis=1),
                                             testTrialSucRateSL.min(axis=1) - testTrialSucRateSL.mean(axis=1)))

    sio.savemat(success_rate_file, {
        "testSucRateRL": testSucRateRL,
        "testSucRateSL": testSucRateSL,
        "testTrialSucRateRL": testTrialSucRateRL,
        "testTrialSucRateSL": testTrialSucRateSL,
        "testSucRate_mean": testSucRate_mean,
        "testSucRate_pos": testSucRate_pos,
        "testSucRate_neg": testSucRate_neg,
        "testTrialSucRate_mean": testTrialSucRate_mean,
        "testTrialSucRate_pos": testTrialSucRate_pos,
        "testTrialSucRate_neg": testTrialSucRate_neg
    })
else:
    successRate = sio.loadmat(success_rate_file)
    testSucRateRL = successRate["testSucRateRL"]
    testSucRateSL = successRate["testSucRateSL"]
    testTrialSucRateRL = successRate["testTrialSucRateRL"]
    testTrialSucRateSL = successRate["testTrialSucRateSL"]
    testSucRate_mean = successRate["testSucRate_mean"]
    testSucRate_pos = successRate["testSucRate_pos"]
    testSucRate_neg = successRate["testSucRate_neg"]
    testTrialSucRate_mean = successRate["testTrialSucRate_mean"]
    testTrialSucRate_pos = successRate["testTrialSucRate_pos"]
    testTrialSucRate_neg = successRate["testTrialSucRate_neg"]




'''
plot success rate
'''
# Define figure
fig, axes = plt.subplots(1, 2, figsize=(13, 5)) 
fig.patch.set_facecolor('w')

# First subplot
ax1 = axes[0]
bars_rl = ax1.bar(np.arange(6) - 0.12, testSucRate_mean[:, 0], width=0.2, edgecolor='none', color=modelColor[1])
bars_sl = ax1.bar(np.arange(6) + 0.12, testSucRate_mean[:, 1], width=0.2, edgecolor='none', color=modelColor[2])

RLcoordinates = [bar.get_x() + bar.get_width() / 2 for bar in bars_rl]
SLcoordinates = [bar.get_x() + bar.get_width() / 2 for bar in bars_sl]

ax1.errorbar(RLcoordinates, testSucRate_mean[:, 0], yerr=[testSucRate_neg[:, 0], testSucRate_pos[:, 0]], fmt='k', capsize=5, linestyle='none')
ax1.errorbar(SLcoordinates, testSucRate_mean[:, 1], yerr=[testSucRate_neg[:, 1], testSucRate_pos[:, 1]], fmt='k', capsize=5, linestyle='none')

ax1.set_xticks(np.arange(6))
ax1.set_xticklabels([r'${\rm Rat ~01}$', r'${\rm Rat ~02}$', r'${\rm Rat ~03}$', r'${\rm Rat ~04}$', r'${\rm Rat ~05}$', r'${\rm Rat ~06}$'], rotation=30)
ax1.tick_params(labelsize=12)
ax1.set_ylabel('Time-bin Success Rate', fontsize=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

for i in range(6):
    stat, temp = wilcoxon(testSucRateRL[i, :], testSucRateSL[i, :], alternative='greater')
    if temp >= 0.05:
        continue
    ax1.plot([RLcoordinates[i] - 0.04, SLcoordinates[i] + 0.04],
             0.03 + max(testSucRate_mean[i, :] + testSucRate_pos[i, :]) * np.array([1, 1]), 'k', linewidth=2)
    ax1.plot(i, 0.07 + max(testSucRate_mean[i, :] + testSucRate_pos[i, :]), 'k*')

ax1.text(0.09, 0.9, 'a', transform=fig.transFigure, fontsize=12, fontweight='bold')

# Second subplot
ax2 = axes[1]
bars_rl = ax2.bar(np.arange(6) - 0.12, testTrialSucRate_mean[:, 0], width=0.2, edgecolor='none', color=modelColor[1])
bars_sl = ax2.bar(np.arange(6) + 0.12, testTrialSucRate_mean[:, 1], width=0.2, edgecolor='none', color=modelColor[2])

RLcoordinates = [bar.get_x() + bar.get_width() / 2 for bar in bars_rl]
SLcoordinates = [bar.get_x() + bar.get_width() / 2 for bar in bars_sl]

ax2.errorbar(RLcoordinates, testTrialSucRate_mean[:, 0], yerr=[testTrialSucRate_neg[:, 0], testTrialSucRate_pos[:, 0]], fmt='k', capsize=5, linestyle='none')
ax2.errorbar(SLcoordinates, testTrialSucRate_mean[:, 1], yerr=[testTrialSucRate_neg[:, 1], testTrialSucRate_pos[:, 1]], fmt='k', capsize=5, linestyle='none')

ax2.set_xticks(np.arange(6))
ax2.set_xticklabels([r'${\rm Rat ~01}$', r'${\rm Rat ~02}$', r'${\rm Rat ~03}$', r'${\rm Rat ~04}$', r'${\rm Rat ~05}$', r'${\rm Rat ~06}$'], rotation=30)
ax2.tick_params(labelsize=12)
ax2.set_ylabel(r'${\rm Trial ~Success ~Rate}$', fontsize=12)
legend = ax2.legend(['RLPP', 'SLPP'], loc='upper right', fontsize=10, ncol=2, frameon=True, bbox_to_anchor=(1.0, 1.1))
frame = legend.get_frame()
frame.set_facecolor('None')
frame.set_edgecolor('k')
ax2.set_ylim([0, 1])
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i in range(6):
    stat, temp = wilcoxon(testTrialSucRateRL[i, :], testTrialSucRateSL[i, :], alternative='greater')
    if temp >= 0.05:
        continue
    ax2.plot([RLcoordinates[i] - 0.04, SLcoordinates[i] + 0.04],
             0.03 + max(testTrialSucRate_mean[i, :] + testTrialSucRate_pos[i, :]) * np.array([1, 1]), 'k', linewidth=2)
    ax2.plot(i, 0.07 + max(testTrialSucRate_mean[i, :] + testTrialSucRate_pos[i, :]), 'k*')

ax2.text(0.52, 0.9, 'b', transform=fig.transFigure, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()