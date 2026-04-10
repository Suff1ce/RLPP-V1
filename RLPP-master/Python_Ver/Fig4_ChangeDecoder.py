import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from utils import *
from model import applynets_priori
from decoding import emulator_real
from draw_figures import *

# %% show time domain of rat02 under manually designed decoder

# Load data
mat_contents = sio.loadmat("trained_results/rat02_RL_results.mat")
folds = mat_contents["folds"][0]


# Define action colors
modelColor = np.array([[250, 0, 0], [54, 56, 131], [103, 146, 70]]) / 255
actionsColor = np.array([[178, 178, 178], [ 255, 165, 0], [128, 0, 128]]) /  255

# %% Load data
DataName = "data/rat02.mat"
data = data_setup(DataName)
data["M1num_pre"] = 4
data["modelName"] = "decodingModel_manual"
data["his"] = 50
data["DataIndex"] = [data["DataIndex"]]
data["DataIndex"].append("Manual")

opt = opt_Setup_real(data)
np.random.seed(0)  # Equivalent to MATLAB's rng('default')
inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
    data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
)
opt["testFold"] = 2 - 1
trainFolds = list(range(opt["foldNum"]))
del trainFolds[opt["testFold"]]
opt["trainTrials"] = np.concatenate([folds[i] for i in trainFolds])
opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
opt["testTrials"] = folds[opt["testFold"]]
opt["NumberOfTestTrials"] = len(opt["testTrials"])

opt["Mode"] = "test"
testInput, testM1Truth, testActions, opt = DataLoader(
    inputEnsemble, M1_truth, Actions, Trials, opt
)
TestSamples = testInput.shape[1]
inputUnitTest = np.vstack([testInput, np.ones((1, TestSamples))])

# %% Get RL outputs
rl_mat_contents = sio.loadmat("trained_results/rat02manual_RL_results.mat")
L1WeightBestReward = rl_mat_contents["L1WeightBestReward"]
L2WeightBestReward = rl_mat_contents["L2WeightBestReward"]
MaxRewardEpisode = rl_mat_contents["MaxRewardEpisode"]

pOutputTest, _, spkOutPredictTest = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, TestSamples, MaxRewardEpisode,
    opt["prioriM"], opt["prioriN"]
)

success, testSucRate, motor_perform_test, _ = emulator_real(
    spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"]
)


'''
Show RL outputs
'''
# %% Show results
data_index = np.arange(1001, 2501)
time_index = data_index / 100
time_range = [10.01, 25]

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_axes([0.13, 0.9, 0.5, 0.02])
plotActions(ax1, time_index, testActions[data_index], actionsColor, time_range)
ax1.set_title("Rat 02", fontsize=14)
ax1.text(time_range[0] * 0.98, 1, 'Real', verticalalignment='center', horizontalalignment='right')
ax1.text(time_range[0] - 2, 1.7, 'a', fontweight='bold', fontsize=12)

ax2 = fig.add_axes([0.13, 0.85, 0.5, 0.02])
motor_perform = motor_perform_test.copy()
motor_perform[testActions == 0] = 0
plotActions(ax2, time_index, motor_perform[data_index], actionsColor, time_range)
ax2.text(time_range[0] * 0.98, 1, 'Decoded', verticalalignment='center', horizontalalignment='right')

I = np.argsort(opt["M1index"])
pOutputTest = pOutputTest[I, :]
spikePredict = spkOutPredictTest[I, :]

ax3 = fig.add_axes([0.13, 0.61, 0.5, 0.18])
ax3.plot(time_index, gaussianSmooth(pOutputTest[0, data_index], 5), color=[0.8500, 0.3250, 0.0980])
ax3.plot(time_index, gaussianSmooth(pOutputTest[2, data_index], 5), color='blue')
ax3.legend(["Artificial Neuron 1", "Artificial  Neuron 2"], loc='upper right', ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(1.2, 1.2))
plotSpikes(ax3, time_index, spikePredict[0, data_index], 1.05, [0.8500, 0.3250, 0.0980])
plotSpikes(ax3, time_index, spikePredict[2, data_index], 0.92, "blue")
ax3.text(time_range[0] - 2, 1.2, 'b', fontweight='bold', fontsize=12)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

ax4 = fig.add_axes([0.13, 0.36, 0.5, 0.18])
ax4.plot(time_index, gaussianSmooth(pOutputTest[1, data_index], 5), color='magenta')
ax4.plot(time_index, gaussianSmooth(pOutputTest[3, data_index], 5), color=[0.4660, 0.6740, 0.1880])
ax4.legend(["Artificial Neuron 3", "Artificial Neuron 4"], loc='upper right', ncol=2, fontsize=10, frameon=False, bbox_to_anchor=(1.2, 1.2))
plotSpikes(ax4, time_index, spikePredict[1, data_index], 1.03, "magenta")
plotSpikes(ax4, time_index, spikePredict[3, data_index], 0.9, [0.4660, 0.6740, 0.1880])
ax4.text(time_range[0] - 2, 1.2, 'c', fontweight='bold', fontsize=12)
ax4.spines['right'].set_visible(False)
ax4.spines['top'].set_visible(False)

ax5 = fig.add_axes([0.13, 0.08, 0.5, 0.18])
ax5.plot(time_index, gaussianSmooth(testM1Truth[0, data_index], 10), color='k')
ax5.plot(time_index, gaussianSmooth(pOutputTest[0, data_index], 5), color=[0.8500, 0.3250, 0.0980])
ax5.legend(["M1 Neuron 1 recording", "Artificial Neuron 1"], loc='upper right', fontsize=10, frameon=False, bbox_to_anchor=(1.0, 1.4))
plotSpikes(ax5, time_index, testM1Truth[0, data_index], 1.05, 'k')
plotSpikes(ax5, time_index, spikePredict[0, data_index], 0.92, [0.8500, 0.3250, 0.0980])
ax5.set_xlabel("Time (sec)", fontsize=10)
ax5.set_ylabel("Firing probability", fontsize=10)
ax5.text(time_range[0] - 2, 1.2, 'd', fontweight='bold', fontsize=12)
ax5.spines['right'].set_visible(False)
ax5.spines['top'].set_visible(False)



'''
Similar modulation under different decoders
'''
def getPlotModulation(x):
    part1 = np.sum(x[:, :, :51], axis=1) / x.shape[1]
    part2 = np.sum(x[:, :, 51:], axis=1) / x.shape[1]
    return np.hstack((
        gaussian_filter1d(part1, sigma=5, axis=1),
        gaussian_filter1d(part2, sigma=5, axis=1)
    ))

# Create figure with MATLAB-like styling
fig = plt.figure(figsize=(8,12))  # Convert cm to inches
fig.patch.set_facecolor('white')

input_unit_test = np.vstack([testInput, np.ones((1, testInput.shape[1]))])

# Generate ground truth plots
lR, hR = getRaster(testActions, testM1Truth)
plot_lR = getPlotModulation(lR)
plot_hR = getPlotModulation(hR)

# Load first decoder results

rat02_RL_results = sio.loadmat("trained_results/rat02_RL_results.mat")
L1WeightBestReward = rat02_RL_results["L1WeightBestReward"]
L2WeightBestReward = rat02_RL_results["L2WeightBestReward"]
MaxRewardEpisode = rat02_RL_results["MaxRewardEpisode"]

_, _, self_decoder = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, TestSamples, MaxRewardEpisode,
    opt["prioriM"], opt["prioriN"]
)


lRp1, hRp1 = getRaster(testActions, self_decoder)
plot_lRp1 = getPlotModulation(lRp1)
plot_hRp1 = getPlotModulation(hRp1)

# Load second decoder results  

rat02decoder01_RL_results = sio.loadmat("trained_results/rat02decoder01_RL_results.mat")
L1WeightBestReward = rat02decoder01_RL_results["L1WeightBestReward"]
L2WeightBestReward = rat02decoder01_RL_results["L2WeightBestReward"]
MaxRewardEpisode = rat02decoder01_RL_results["MaxRewardEpisode"]

_, _, decoder01 = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, TestSamples, MaxRewardEpisode,
    opt["prioriM"], opt["prioriN"]
)

lRp2, hRp2 = getRaster(testActions, decoder01)
plot_lRp2 = getPlotModulation(lRp2)
plot_hRp2 = getPlotModulation(hRp2)

# Plotting parameters
neuron_idx = 1 
modelColor = ['k', 'b', 'c']

# Raster plots
ax1 = fig.add_axes([0.07, 0.81, 0.12, 0.05])
plotTestRasters(ax1, lR, lRp1, lRp2, neuron_idx, modelColor)
ax2 = fig.add_axes([0.20, 0.81, 0.12, 0.05])
plotTestRasters(ax2, hR, hRp1, hRp2, neuron_idx, modelColor)

# Modulation plots
ymax, ymin, h = get_yl(plot_lR, plot_hR, plot_lRp1, plot_hRp1, 
                       plot_lRp2, plot_hRp2, neuron_idx)

ax3 = fig.add_axes([0.07, 0.7, 0.12, 0.1])
l1, l2, l3 = plotModulation(
    ax3, 
    plot_lR[neuron_idx], plot_lRp1[neuron_idx], plot_lRp2[neuron_idx],
    modelColor, actionsColor[[0,1]], ymin, ymax, h
)

# Custom y-axis labels
for y in np.arange(ymin, ymax+0.3, 0.3):
    ax3.text(-0.54, y, f"{y:.1f}", ha='right', va='center', 
            fontsize=10, usetex=True)

ax3.text(-0.82, (ymax+ymin)/2-0.07, 'firing probability',
        rotation=90, va='center', ha='center', 
        fontsize=10, usetex=True)

# Condition labels
ax3.text(-0.25, ymin-(ymax-ymin)/9, 'Rest', 
        ha='center', color=[0.5,0.5,0.5], fontsize=10, usetex=True)
ax3.text(0.25, ymin-(ymax-ymin)/9, 'Press',
        ha='center', color=actionsColor[1], fontsize=10, usetex=True)
ax3.text(0, ymin-(ymax-ymin)/3, '(Low Trials)',
        ha='center', color='k', fontsize=10, usetex=True)

# Scale bar
ax3.plot([-0.2, 0], [0.5, 0.5], color='k', lw=1.5)
ax3.text(-0.4, 0.6, '200 ms', fontsize=10, usetex=True)

# Right modulation plot
ax4 = fig.add_axes([0.20, 0.7, 0.12, 0.1])
plotModulation(
    ax4,
    plot_hR[neuron_idx], plot_hRp1[neuron_idx], plot_hRp2[neuron_idx],
    modelColor, actionsColor[[0,2]], ymin, ymax, h
)

ax4.text(-0.25, ymin-(ymax-ymin)/9, 'Rest', 
        ha='center', color=[0.5,0.5,0.5], fontsize=10, usetex=True)
ax4.text(0.25, ymin-(ymax-ymin)/9, 'Press',
        ha='center', color=actionsColor[2], fontsize=10, usetex=True)
ax4.text(0, ymin-(ymax-ymin)/3, '(High Trials)',
        ha='center', color='k', fontsize=10, usetex=True)

# Legend
leg = fig.legend(
    [l1, l2, l3],
    [f'Rat 02 Neuron {neuron_idx+1} Recordings',  # Original MATLAB neuronIdx=2
     'RLPP with decoder 02',
     'RLPP with decoder 01'],
    loc='upper left',
    bbox_to_anchor=(0.04, 0.95),
    frameon=False,
    fontsize=10,
    handlelength=1.5,
    columnspacing=1.5
)




'''
Behavioral Performance under different decoders
'''


# Configure Matplotlib settings
plt.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'axes.spines.right': False,
    'axes.spines.top': False
})

# Load data
data = sio.loadmat('trained_results/successRate_DifferentDecoders.mat')
testSucRate = data['testSucRate']
testTrialSucRate = data['testTrialSucRate']

# Define colors
model_color = np.array([[0.7, 0.7, 0.7],  # Placeholder color 1
                        [0, 0.4470, 0.7410]])  # Blue color
group_color = [200/255, 220/255, 255/255]

# First subplot: Time-bin Success Rate
ax1 = fig.add_axes([0.1, 0.4, 0.6, 0.2])  # [left, bottom, width, height]

# Calculate statistics
means = np.mean(testSucRate, axis=2)
mins = np.min(testSucRate, axis=2)
maxs = np.max(testSucRate, axis=2)

# Plot parameters
n_groups, n_bars = means.shape
x = np.arange(n_groups)
width = 0.8 / n_bars

# Create bars
bars = []
for i in range(n_bars):
    bars.append(ax1.bar(x + i*width, means[:,i], width, 
                        color='white', edgecolor='k', linewidth=0.5))

# Color bars
for i in range(n_bars):
    for j in range(n_groups):
        if i == j:
            bars[i][j].set_facecolor(model_color[1])
        elif i != 6:  # 7th bar (index 6) remains white
            bars[i][j].set_facecolor(group_color)

# Add error bars
for i in range(n_bars):
    x_pos = x + i*width + width/2
    y = means[:,i]
    yerr = [y - mins[:,i], maxs[:,i] - y]
    ax1.errorbar(x_pos, y, yerr=yerr, fmt='none', 
                c='k', linewidth=0.8, capsize=3)

# Formatting
ax1.plot([-0.45, 5.45], [1/3, 1/3], 'k--', linewidth=1.5)
ax1.set_xticks(x + width*3)
ax1.set_xticklabels([r'$\rm{Rat~01}$', r'$\rm{Rat~02}$', r'$\rm{Rat~03}$',
                     r'$\rm{Rat~04}$', r'$\rm{Rat~05}$', r'$\rm{Rat~06}$'])
ax1.set_ylabel('Time-bin Success Rate', fontsize=12, labelpad=15)
ax1.text(5.5, 1/3, 'One Step\nChance\nRate', 
         fontsize=9, va='center', linespacing=1.5)

# Second subplot: Trial Success Rate
ax2 = fig.add_axes([0.1, 0.1, 0.6, 0.2])  # [left, bottom, width, height]

# Calculate statistics
means_trial = np.mean(testTrialSucRate, axis=2)
mins_trial = np.min(testTrialSucRate, axis=2)
maxs_trial = np.max(testTrialSucRate, axis=2)

# Create bars
bars_trial = []
for i in range(n_bars):
    bars_trial.append(ax2.bar(x + i*width, means_trial[:,i], width, 
                             color='white', edgecolor='k', linewidth=0.5))

# Color bars
for i in range(n_bars):
    for j in range(n_groups):
        if i == j:
            bars_trial[i][j].set_facecolor(model_color[1])
        elif i != 6:
            bars_trial[i][j].set_facecolor(group_color)

# Add error bars
for i in range(n_bars):
    x_pos = x + i*width + width/2
    y = means_trial[:,i]
    yerr = [y - mins_trial[:,i], maxs_trial[:,i] - y]
    ax2.errorbar(x_pos, y, yerr=yerr, fmt='none', 
                c='k', linewidth=0.8, capsize=3)

# Formatting
ax2.plot([-0.45, 5.45], [0.02, 0.02], 'k--', linewidth=1.5)
ax2.set_xticks(x + width*3)
ax2.set_xticklabels([r'$\rm{Rat~01}$', r'$\rm{Rat~02}$', r'$\rm{Rat~03}$',
                    r'$\rm{Rat~04}$', r'$\rm{Rat~05}$', r'$\rm{Rat~06}$'])
ax2.set_ylabel('Trial Success Rate', fontsize=12, labelpad=15)
ax2.text(5.4, 0.15, 'Multi-Step\nChance\nRate', 
        fontsize=9, va='center', linespacing=1.5)

plt.show()