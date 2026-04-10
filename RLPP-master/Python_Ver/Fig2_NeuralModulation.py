import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
from scipy.io import loadmat, savemat
import os

from utils import *
from draw_figures import plotTestRasters, plotModulation
from model import applynets, applynets_priori


# Set style parameters
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 9
})

# Define colors
modelColor = np.array([[250, 0, 0], [54, 56, 131], [103, 146, 70]]) / 255
actionsColor = np.array([[178, 178, 178], [ 255, 165, 0], [128, 0, 128]]) /  255

# Load data
folds = loadmat("trained_results/rat01_RL_results.mat")['folds'][0]
DataName = "./Python_Ver/data/rat01.mat"
data = data_setup(DataName)
opt = opt_Setup_real(data)
np.random.seed(42)  # For reproducibility

# Assuming PreProcess and DataLoader return appropriate values
inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
    data['mPFC'], data['M1'], data['segment'], data['trialNo'], 
    opt['decay_parameter'], opt)  # Access dictionary values with []
opt['testFold'] = 4         # test fold index = 5
trainFolds = [i for i in range(opt['foldNum']) if i != opt['testFold']]
opt['trainTrials'] = np.concatenate([folds[i] for i in trainFolds])  # Convert to 0-based
opt['NumberOfTrainTrials'] = len(opt['trainTrials'])
opt['testTrials'] = folds[opt['testFold']]
opt['NumberOfTestTrials'] = len(opt['testTrials'])

opt['Mode'] = 'test'
testInput, testM1_truth, testActions, opt = DataLoader(
    inputEnsemble, M1_truth, Actions, Trials, opt)
TestSamples = testInput.shape[1]
inputUnitTest = np.concatenate([testInput, np.ones((1, TestSamples))], axis=0)

# Get RL outputs
rl_data = loadmat("./Python_Ver/trained_results/rat01_RL_results.mat")
L1WeightBestReward = rl_data['L1WeightBestReward']
L2WeightBestReward = rl_data['L2WeightBestReward']
MaxRewardEpisode = rl_data['MaxRewardEpisode'][0,0]

_, _, spkOutPredictTest = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, 
    TestSamples, MaxRewardEpisode, opt['prioriM'], opt['prioriN'])
RL = {'spkOutPredictTest': spkOutPredictTest}

# Get SL outputs
sl_data = loadmat("./Python_Ver/trained_results/rat01_Sup_results.mat")
L1Weight = sl_data['L1Weight']
L2Weight = sl_data['L2Weight']

_, _, spkOutPredictTest = applynets(inputUnitTest, L1Weight, L2Weight, TestSamples)
SL = {'spkOutPredictTest': spkOutPredictTest}



'''
raster plots
'''
# Create figure for raster plots
fig = plt.figure(figsize=(15/2.54, 17/2.54))  # Convert cm to inches
fig.patch.set_facecolor('white')

# Rat 01 Neuron 1 & 2 plots
# Assuming getTestRasters generates required variables
lR, hR, lRp, hRp, lRsp, hRsp, plot_lR, plot_hR, plot_lRp, plot_hRp, plot_lRsp, plot_hRsp = getTestRasters(testActions, testM1_truth, RL, SL)

# Neuron 1 Spikes
neuronIdx = 0  # Python uses 0-based indexing
ax1 = fig.add_axes([0.07, 0.85, 0.12, 0.05])
plotTestRasters(ax1, lR, lRp, lRsp, neuronIdx, modelColor) 
ax2 = fig.add_axes([0.20, 0.85, 0.12, 0.05])
plotTestRasters(ax2, hR, hRp, hRsp, neuronIdx, modelColor) 

ymax, ymin, h = get_yl(plot_lR, plot_hR, plot_lRp, plot_hRp, plot_lRsp, plot_hRsp, neuronIdx)

# ----- Neuron 1 Modulation Plots -----
ax3 = fig.add_axes([0.07, 0.74, 0.12, 0.1])  # Convert MATLAB Position to Python
l1,l2,l3 = plotModulation(ax3, plot_lR[neuronIdx,:], plot_lRp[neuronIdx,:], plot_lRsp[neuronIdx,:],
              modelColor, actionsColor[[0,1],:], ymin, ymax, h)

# Custom Y-axis labels
for y_val in np.arange(ymin, ymax, 0.2):  # MATLAB's ymin:0.2:ymax
    ax3.text(-0.52, y_val, f'{y_val:.1f}', 
            horizontalalignment='right', 
            verticalalignment='center',
            fontsize=6.5,
            usetex=True)

# Action labels
# Firing probability label
ax3.text(-0.82, (ymax + ymin)/2 - 0.07, 'firing probability',
        rotation=90,
        verticalalignment='center',
        horizontalalignment='center',
        fontsize=6.5,
        usetex=True)

# Condition labels
ax3.text(-0.25, -0.05, 'Rest',
        horizontalalignment='center',
        verticalalignment='top',
        color=(0.5, 0.5, 0.5),
        fontsize=7,
        usetex=True)

ax3.text(0.25, -0.05, 'Press',
        horizontalalignment='center',
        verticalalignment='top',
        color=actionsColor[1],
        fontsize=7,
        usetex=True)

ax3.text(0, -0.14, '(Low Trials)',
        horizontalalignment='center',
        verticalalignment='center',
        color='k',
        fontsize=7,
        usetex=True)

# Scale bar
ax3.plot([0, 0.2], [0.35, 0.35],  # X and Y coordinates
        color='k', 
        linewidth=1.5)
ax3.text(0.07, 0.36, '200 ms',
        fontsize=4,
        usetex=True)

# Right Modulation Plot
ax4 = fig.add_axes([0.20, 0.74, 0.12, 0.1])
plotModulation(ax4, plot_hR[neuronIdx,:], plot_hRp[neuronIdx,:], plot_hRsp[neuronIdx,:],
              modelColor, actionsColor[[0,2],:], ymin, ymax, h)

# Action labels (High Trials version)
ax4.text(-0.25, -0.05, 'Rest',
        horizontalalignment='center',
        verticalalignment='top',
        color=(0.5, 0.5, 0.5),
        fontsize=7,
        usetex=True)

ax4.text(0.25, -0.05, 'Press',
        horizontalalignment='center',
        verticalalignment='top',
        color=actionsColor[2],  # MATLAB's 3rd row becomes Python index 2
        fontsize=7,
        usetex=True)

ax4.text(0, -0.14, '(High Trials)',
        horizontalalignment='center',
        verticalalignment='center',
        color='k',
        fontsize=7,
        usetex=True)

# Create legend using proxy artists

legend_elements = [l1, l2, l3]
legend_elements[0].set_label('Recordings')
legend_elements[1].set_label('RLPP')
legend_elements[2].set_label('SLPP')

leg = fig.legend(handles=legend_elements,
               loc='upper center',
               bbox_to_anchor=(0.18, 0.95),  # Adjusted position
               ncol=3,
               frameon=False,
               fontsize=6.5,
               handlelength=1.5,
               columnspacing=1.5)

# Annotations
fig.text(0.2, 0.92, 'Rat 01 Neuron 1',  # Normalized figure coordinates
        fontsize=9,
        ha='center',
        va='top',
        usetex=True)

fig.text(0.04, 0.92, 'a(i)',
        fontweight='bold',
        fontsize=9,
        ha='left',
        va='top')

# ----- Neuron 2 Section -----
neuronIdx = 1  # Second neuron (0-based index)

# Raster plots
ax5 = fig.add_axes([0.385, 0.85, 0.12, 0.05])
plotTestRasters(ax5, lR, lRp, lRsp, neuronIdx, modelColor)
ax6 = fig.add_axes([0.515, 0.85, 0.12, 0.05])
plotTestRasters(ax6, hR, hRp, hRsp, neuronIdx, modelColor)

# Update y limits for Neuron 2
ymax, ymin, h = get_yl(plot_lR, plot_hR, plot_lRp, plot_hRp, 
                      plot_lRsp, plot_hRsp, neuronIdx)

# Modulation plots
ax7 = fig.add_axes([0.385, 0.74, 0.12, 0.1])
plotModulation(ax7, plot_lR[neuronIdx,:], plot_lRp[neuronIdx,:], plot_lRsp[neuronIdx,:],
              modelColor, actionsColor[[0,1],:], ymin, ymax, h)

# Y-axis labels
for y_val in np.linspace(ymin, ymax, 3):  # MATLAB's linspace equivalent
    ax7.text(-0.52, y_val, f'{y_val:.2f}',  # Format to 2 decimal places
            horizontalalignment='right',
            verticalalignment='center',
            fontsize=6.5,
            usetex=True)


ax8 = fig.add_axes([0.515, 0.74, 0.12, 0.1])
plotModulation(ax8, plot_hR[neuronIdx,:], plot_hRp[neuronIdx,:], plot_hRsp[neuronIdx,:],
              modelColor, actionsColor[[0,2],:], ymin, ymax, h)

# Neuron 2 annotations
fig.text(0.5, 0.92, 'Rat 01 Neuron 2',
        fontsize=9,
        ha='center',
        va='top',
        usetex=True)

fig.text(0.355, 0.92, 'a(ii)',
        fontweight='bold',
        fontsize=9,
        ha='left',
        va='top')




'''
show Modulation Depth
'''
# Load modulation depth data
mat_path = os.path.join('./Python_Ver/trained_results', 'modDep.mat')
if os.path.exists(mat_path):
    mat = loadmat(mat_path)
    modDep = mat['modDep']
else:
    # Fallback: compute modDep via your function
    bestFold = opt['testFold']
    modDep = getModulationDepth(bestFold, plot_lR, plot_hR,
                                 plot_lRp, plot_hRp, plot_lRsp, plot_hRsp)

# Configure LaTeX rendering for all text
plt.rc('text', usetex=True)
plt.rc('font', size=8)

# Create figure: size in cm -> inches (1 inch = 2.54 cm)
fig = plt.figure(figsize=(18/2.54, 8/2.54))
fig.patch.set_facecolor('white')

# Create 2x2 subplots
axes = [fig.add_subplot(2, 2, idx+1) for idx in range(4)]

# Loop over subplots 
for j in range(4):
    # Determine bin range limit: max absolute modulation, rounded up to nearest 0.05
    block = np.abs(modDep[j::4, :])
    maxMod = np.ceil(block.max() * 20) / 20
    bins = np.arange(-maxMod, maxMod + 0.05, 0.05)

    # Plot histograms for recordings (i indices: j+8, j, j+4)
    for i in [j+8, j, j+4]:
        ax = axes[j]
        # Histogram with translucent fill and no edge
        ax.hist(modDep[i, :].ravel(), bins=bins,
                facecolor=modelColor[i//4],
                edgecolor='none', alpha=0.3)
        # Ensure LaTeX tick labels and proper font size
        ax.tick_params(labelsize=8)

# Set titles and labels
titles = [r'Low Press - Low Rest', r'High Press - High Rest',
          r'High Press - Low Press', r'High Rest - Low Rest']
xlabels = [None, None,
           r'Difference in average firing probability (spikes/10 ms)', None]

ylabels = [None, None, r'Number of Neurons', None]

for idx, ax in enumerate(axes):
    ax.set_title(titles[idx], fontsize=9)
    if xlabels[idx]:
        ax.set_xlabel(xlabels[idx], fontsize=9)
    if ylabels[idx]:
        ax.set_ylabel(ylabels[idx], fontsize=8)
    ax.set_xlim(-0.7, 0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if idx == 0:
        handles = [plt.Rectangle((0,0),1,1, fc=modelColor[i//4], alpha=0.3) 
                   for i in [0,4,8]]
        leg = ax.legend(handles, ['Recordings', 'RLPP', 'SLPP'],
                        loc='upper left', frameon=False,
                        handlelength=1.5, handletextpad=0.5,
                        fontsize=8, bbox_to_anchor=(0.10, 0.98))
        leg.set_title(None)

# Overlay Gaussian fits
for j in range(4):
    for i in [j+8, j, j+4]:
        ax = axes[j]
        data = modDep[i, :].ravel()
        maxGau = np.ceil(np.abs(data).max() * 20) / 20
        x = np.arange(-maxGau, maxGau + 0.01, 0.01)
        gau = norm.pdf(x, loc=data.mean(), scale=data.std()) * len(data) * 0.05
        style = {'linewidth': 2, 'color': modelColor[i//4]}
        # Line styles by model group
        if i//4 == 0:
            style['linestyle'] = '--'
        elif i//4 == 1:
            style['linestyle'] = '-'
        else:
            style['linestyle'] = '-.'
        ax.plot(x, gau, **style)

# Annotations for subplot labels d(i), d(ii), etc.
labels = [r'd(i)', r'd(ii)', r'd(iii)', r'd(iv)']
positions = [(0.04, 0.96), (0.48, 0.96), (0.04, 0.50), (0.48, 0.50)]
for lbl, pos in zip(labels, positions):
    fig.text(pos[0], pos[1], lbl, fontsize=9,
             fontname='Times New Roman', fontweight='bold')

# Draw legend annotation lines manually (matching MATLAB positions)
# Coordinates adjusted to figure fraction
lines = [('--', modelColor[0], (0.085, 0.853)),
         ('-', modelColor[1], (0.085, 0.81)),
         ('-.', modelColor[2], (0.085, 0.767))]
for ls, col, (x, y) in lines:
    fig.lines.append(Line2D([x, x+0.035], [y, y], linestyle=ls,
                            color=col, linewidth=2, transform=fig.transFigure))

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()
