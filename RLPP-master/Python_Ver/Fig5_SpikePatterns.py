# show spike patterns by t-SNE
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from utils import *
from model import applynets, applynets_priori
from decoding import emulator_real

# load data
folds = sio.loadmat("trained_results/rat02_RL_results.mat")["folds"][0]

# define color
actionsColor = np.array([[  178, 178, 178],    [ 255, 165, 0],  [128, 0, 128 ]]) / 255;

# load data
DataName = "./Python_Ver/data/rat02.mat"
data = data_setup(DataName)
opt = opt_Setup_real(data)

np.random.seed(0)
inputEnsemble, M1_truth, Actions, Trials, opt = PreProcess(
    data["mPFC"], data["M1"], data["segment"], data["trialNo"], opt["decay_parameter"], opt
)
opt["testFold"] = 2 - 1
trainFolds = list(range(opt["foldNum"]))
trainFolds.remove(opt["testFold"])
opt["trainTrials"] = np.hstack([folds[i] for i in trainFolds])
opt["NumberOfTrainTrials"] = len(opt["trainTrials"])
opt["testTrials"] = folds[opt["testFold"]]
opt["NumberOfTestTrials"] = len(opt["testTrials"])

opt["Mode"] = "test"
testInput, testM1_truth, testActions, opt = DataLoader(
    inputEnsemble, M1_truth, Actions, Trials, opt
)
TestSamples = testInput.shape[1]
inputUnitTest = np.vstack([testInput, np.ones((1, TestSamples))])
_, _, _, M1_ensemble = emulator_real(
    testM1_truth, testActions, opt["M1index"], data["his"], data["modelName"]
)
Recordings = {"M1_ensemble": M1_ensemble}

# Get RL outputs
mat_data = sio.loadmat("./Python_Ver/trained_results/rat02_RL_results.mat")
L1WeightBestReward = mat_data["L1WeightBestReward"]
L2WeightBestReward = mat_data["L2WeightBestReward"]
MaxRewardEpisode = mat_data["MaxRewardEpisode"]

_, _, spkOutPredictTest = applynets_priori(
    inputUnitTest, L1WeightBestReward, L2WeightBestReward, TestSamples, MaxRewardEpisode,
    opt["prioriM"], opt["prioriN"]
)
_, _, _, M1_ensemble = emulator_real(
    spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"]
)
RL = {"M1_ensemble": M1_ensemble}

# Get SL outputs
mat_data = sio.loadmat("./Python_Ver/trained_results/rat02_Sup_results.mat")
L1Weight = mat_data["L1Weight"]
L2Weight = mat_data["L2Weight"]

_, _, spkOutPredictTest = applynets(inputUnitTest, L1Weight, L2Weight, TestSamples)
_, _, _, M1_ensemble = emulator_real(
    spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"]
)
SL = {"M1_ensemble": M1_ensemble}

# run tSNE
np.random.seed(0)

X = [
    TSNE().fit_transform(Recordings["M1_ensemble"][:, testActions > 0].T),
    TSNE().fit_transform(RL["M1_ensemble"][:, testActions > 0].T),
    TSNE().fit_transform(SL["M1_ensemble"][:, testActions > 0].T),
]
y = testActions[testActions > 0]
actions = ['Rest', 'Press Low', 'Press High']
y_str = np.array([actions[x - 1] for x in y.flatten()])


# show results
fig = plt.figure(figsize=(8,6))
ax1 = fig.add_axes([0.1, 0.1, 0.25, 0.3])

# Define colors and markers
colorSeq = np.unique(y) -1
sns.scatterplot(x=X[0][:, 0], y=X[0][:, 1], hue=y_str, palette=actionsColor[colorSeq, :].tolist(), ax=ax1, s=10, legend=False)

# Add annotations
strings = ['Rest', 'Press Low', 'Press High']
rect = plt.Rectangle((0.06, 0.882), 0.15, 0.07, transform=fig.transFigure, color='white', zorder=2)
fig.patches.append(rect)

for i in range(3):
    ellipse = plt.Circle((0.07, 0.505 + i * 0.02), 0.0075, transform=fig.transFigure, color=actionsColor[2 - i, :], ec='none', zorder=3)
    fig.patches.append(ellipse)
    fig.text(0.08, 0.50 + i * 0.02, strings[2 - i], fontsize=10, verticalalignment='center', fontname='Times New Roman')

ax1.set_title('Recorded Spike Pattern', fontname='Times New Roman', fontsize=12, fontweight='normal')
ax1.set_xticklabels(ax1.get_xticks(), fontname='Times New Roman')
ax1.set_yticklabels(ax1.get_yticks(), fontname='Times New Roman')
ax1.text(-0.48, 0.5, 'Rat 02', transform=ax1.transAxes, fontsize=12, fontname='Times New Roman')

# Second subplot - Spike Pattern by RLPP
ax2 = fig.add_axes([0.4, 0.1, 0.25, 0.3])
sns.scatterplot(x=X[1][:, 0], y=X[1][:, 1], hue=y_str, palette=actionsColor[colorSeq, :], ax=ax2, s=10, legend=False)
ax2.set_title('Spike Pattern by RLPP', fontname='Times New Roman', fontsize=12, fontweight='normal')
ax2.set_xticklabels(ax2.get_xticks(), fontname='Times New Roman')
ax2.set_yticklabels(ax2.get_yticks(), fontname='Times New Roman')

# Third subplot - Spike Pattern by SLPP
ax3 = fig.add_axes([0.7, 0.1, 0.25, 0.3])
sns.scatterplot(x=X[2][:, 0], y=X[2][:, 1], hue=y_str, palette=actionsColor[colorSeq, :], ax=ax3, s=10, legend=False)
ax3.set_title('Spike Pattern by SLPP', fontname='Times New Roman', fontsize=12, fontweight='normal')
ax3.set_xticklabels(ax3.get_xticks(), fontname='Times New Roman')
ax3.set_yticklabels(ax3.get_yticks(), fontname='Times New Roman')

plt.show()

