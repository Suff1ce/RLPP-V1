import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy import stats
import os
import statsmodels.api as sm

# Define color matrices
color_mat = np.array([
    [0,      0.4470, 0.7410],
    [0.8500, 0.3250, 0.0980],
    [0.9290, 0.6940, 0.1250],
    [0.4940, 0.1840, 0.5560],
    [0.4660, 0.6740, 0.1880],
    [0.3010, 0.7450, 0.9330]
])

modelColor = np.array([[250, 0, 0], [54, 56, 131], [103, 146, 70]]) / 255

# Create figure with specified dimensions
fig = plt.figure(figsize=(32/2.54, 15/2.54), facecolor='w')  # Convert cm to inches
fig.set_size_inches(32/2.54, 15/2.54)  # Explicitly set size

# Load Mutual Information data
if not os.path.exists("trained_results/mutualInformation.mat"):
    # Assuming getMutualInformation is a MATLAB script; need to implement separately
    raise FileNotFoundError("mutualInformation.mat not found and getMutualInformation not implemented")
else:
    data = loadmat("trained_results/mutualInformation.mat")
    M1_act = data['M1_act']  
    M1_mPFC = data['M1_mPFC']
    mPFC_act = data['mPFC_act'].squeeze()  # Remove singleton dimensions




# Create subplots
ax1 = plt.subplot(241)
ax2 = plt.subplot(242)
ax3 = plt.subplot(243)

# Plot data points
for ratIdx in range(6):  # 0-based index for 6 rats
    for M1Idx in range(len(M1_act[0, ratIdx][0])):  # Assuming M1_act structure
        # Subplot 1: M1_act[2] vs M1_act[1]
        ax1.plot(M1_act[2, ratIdx][0][M1Idx], M1_act[1, ratIdx][0][M1Idx], 
                '.', color='k')
        
        # Subplot 2: M1_act[0] vs M1_act[2]
        ax2.plot(M1_act[0, ratIdx][0][M1Idx], M1_act[2, ratIdx][0][M1Idx],
                '.', color='k')
        
        # Subplot 3: M1_act[0] vs M1_act[1]
        ax3.plot(M1_act[0, ratIdx][0][M1Idx], M1_act[1, ratIdx][0][M1Idx],
                '.', color='k')

# Format subplot 1
ax1.set_xlim([0, 0.8])
ax1.set_ylim([0, 0.8])
ax1.plot([0, 0.8], [0, 0.8], 'k--')
ax1.set_xlabel('$\\mathrm{MI(SLPP;~Movement)}$', fontsize=10)
ax1.set_ylabel('$\\mathrm{MI(RLPP;~Movement)}$', fontsize=10)
ax1.text(-0.2, 0.85, 'a', fontname='Times New Roman',
        fontweight='bold', fontsize=14, transform=ax1.transAxes)

# Format subplot 2
ax2.set_xlim([0, 0.8])
ax2.set_ylim([0, 0.8])
ax2.plot([0, 0.8], [0, 0.8], 'k--')
ax2.set_xlabel('$\\mathrm{MI(M1~Recordings;~Movement)}$', fontsize=10)
ax2.set_ylabel('$\\mathrm{MI(SLPP;~Movement)}$', fontsize=10)
ax2.text(-0.2, 0.85, 'b', fontname='Times New Roman',
        fontweight='bold', fontsize=14, transform=ax2.transAxes)

# Format subplot 3
ax3.set_xlim([0, 0.8])
ax3.set_ylim([0, 0.8])
ax3.plot([0, 0.8], [0, 0.8], 'k--')
ax3.set_xlabel('$\\mathrm{MI(M1~Recordings;~Movement)}$', fontsize=10)
ax3.set_ylabel('$\\mathrm{MI(RLPP;~Movement)}$', fontsize=10)
ax3.text(-0.2, 0.85, 'c', fontname='Times New Roman',
        fontweight='bold', fontsize=14, transform=ax3.transAxes)

# Statistical tests
# Convert cell arrays to numpy arrays
data_2 = np.concatenate([M1_act[1, ratIdx][0] for ratIdx in range(6)])
data_3 = np.concatenate([M1_act[2, ratIdx][0] for ratIdx in range(6)])
data_1 = np.concatenate([M1_act[0, ratIdx][0] for ratIdx in range(6)])

# Test RLPP vs SLPP
t_stat, p_val = stats.ttest_rel(data_2, data_3, alternative='greater')
if p_val < 0.05:
    print(f'RLPP significantly higher than SLPP, p = {p_val:.4f}')

# Test Recordings vs SLPP
t_stat, p_val = stats.ttest_rel(data_1, data_3, alternative='greater')
if p_val < 0.05:
    print(f'Recordings significantly higher than SLPP, p = {p_val:.4f}')

# Test Recordings vs RLPP
t_stat, p_val = stats.ttest_ind(data_1, data_2)
if p_val >= 0.05:
    print(f'Recordings and RLPP no significant differences, p = {p_val:.4f}')







# Adjust subplot positions for the first row
ax1 = plt.subplot(241)
ax2 = plt.subplot(242)
ax3 = plt.subplot(243)

# Get original positions
pos3 = ax3.get_position().bounds
pos2 = ax2.get_position().bounds
pos1 = ax1.get_position().bounds

# Adjust positions according to MATLAB logic
new_pos3 = (pos3[0] + pos2[0] - pos1[0] - 0.04, pos3[1], pos3[2], pos3[3])
ax3.set_position(new_pos3)

new_pos2 = ((new_pos3[0]/2 + (pos1[0]+0.04)/2), pos2[1], pos2[2], pos2[3])
ax2.set_position(new_pos2)

new_pos1 = (pos1[0]+0.04, pos1[1], pos1[2], pos1[3])
ax1.set_position(new_pos1)

print(' ')

'''
 Predicted M1-mPFC vs real M1-mPFC
'''
print('Compare the Mutual information between M1 and mPFC')

# Create subplots 5 and 6 (2nd row, 1st and 2nd columns)
ax5 = plt.subplot(245)
ax5.plot([0.1, 0.7], [0.1, 0.7], '--k')

ax6 = plt.subplot(246)
ax6.plot([0.1, 0.7], [0.1, 0.7], '--k')

points = {1: [], 2: [], 3: []}  # Using dictionary for MATLAB-like cell array

# Assuming M1_mPFC is loaded from the .mat file similar to M1_act
for modelIdx in [1, 2]:  # MATLAB's 2:3 corresponds to Python 1:2 (0-based)
    for ratIdx in range(6):  # 6 rats
        for M1Idx in range(len(M1_mPFC[0, ratIdx][0])):
            current_ax = plt.subplot(2, 4, modelIdx + 4)  # 5 and 6 subplots
            x_val = M1_mPFC[0, ratIdx][0][M1Idx]
            y_val = M1_mPFC[modelIdx, ratIdx][0][M1Idx]  # +1 for MATLAB to Python index
            
            current_ax.plot(x_val, y_val, '.', color=modelColor[modelIdx])
            points[modelIdx+1].append([x_val, y_val])

# Convert points to numpy arrays
for k in points:
    points[k] = np.array(points[k])

# Process RLPP (model 2)
xc = points[2][:, 0]
yc = points[2][:, 1]

# Linear regression and confidence interval
p, S = np.polyfit(xc, yc, 1, cov=True)
x_pred = np.unique(xc)
x_pred = np.insert(x_pred, 0, min(xc)-0.05)
x_pred = np.append(x_pred, max(xc)+0.05)
X = np.vstack([x_pred, np.ones(len(x_pred))]).T
Y = X @ p

# Calculate prediction interval
t_val = stats.t.ppf(0.975, len(xc)-2)
dy = t_val * np.sqrt(S[0,0]*(X[:,0]**2) + S[1,1] + 2*S[0,1]*X[:,0])

ax5.plot(x_pred, Y, '-', color=modelColor[1], lw=1.5)
ax5.fill_between(x_pred, Y-dy, Y+dy, color=modelColor[1], alpha=0.3)

# T-test
t_stat, p_val = stats.ttest_rel(points[2][:,0], points[2][:,1], alternative='less')
if p_val < 0.05:
    print(f'RLPP significantly higher than recordings, p = {p_val:.4f}')

ax5.set_xlim([0.15, 0.6])
ax5.set_ylim([0.15, 0.8])
ax5.set_xlabel('$\\mathrm{MI(M1~Recordings;~mPFC)}$', fontsize=10)
ax5.set_ylabel('$\\mathrm{MI(RLPP;~mPFC)}$', fontsize=10)
ax5.text(0.02, 0.85, 'd', fontname='Times New Roman',
        fontweight='bold', fontsize=14, transform=ax5.transAxes)

# Process SLPP (model 3)
xc = points[3][:, 0]
yc = points[3][:, 1]

# Calculate correlation
corr_coef = np.corrcoef(xc, yc)[0,1]
print(f'SLPP CC: {corr_coef:.4f}')

# Linear regression
p, S = np.polyfit(xc, yc, 1, cov=True)
x_pred = np.unique(xc)
x_pred = np.insert(x_pred, 0, min(xc)-0.05)
x_pred = np.append(x_pred, max(xc)+0.05)
X = np.vstack([x_pred, np.ones(len(x_pred))]).T
Y = X @ p

# Calculate prediction interval
dy = t_val * np.sqrt(S[0,0]*(X[:,0]**2) + S[1,1] + 2*S[0,1]*X[:,0])

ax6.plot(x_pred, Y, '-', color=modelColor[2], lw=1.5)
ax6.fill_between(x_pred, Y-dy, Y+dy, color=modelColor[2], alpha=0.3)

# Regression stats using statsmodels
X_sm = sm.add_constant(xc)
model = sm.OLS(yc, X_sm)
results = model.fit()
print(f'Regression on SLPP, p = {results.f_pvalue:.4f}')

# T-test
t_stat, p_val = stats.ttest_rel(points[3][:,0], points[3][:,1], alternative='greater')
if p_val < 0.05:
    print(f'SLPP significantly lower than recordings, p = {p_val:.4f}')

ax6.set_xlim([0.15, 0.6])
ax6.set_ylim([0.15, 0.6])
ax6.set_xlabel('$\\mathrm{MI(M1~Recordings;~mPFC)}$', fontsize=10)
ax6.set_ylabel('$\\mathrm{MI(SLPP;~mPFC)}$', fontsize=10)
ax6.text(0.02, 0.65, 'e', fontname='Times New Roman',
        fontweight='bold', fontsize=14, transform=ax6.transAxes)

# Adjust final subplot positions
ax5_pos = ax5.get_position().bounds
ax5.set_position([ax5_pos[0]-0.02, ax5_pos[1], ax5_pos[2], ax5_pos[3]])

ax6_pos = ax6.get_position().bounds
ax6.set_position([ax6_pos[0]-0.01, ax6_pos[1], ax6_pos[2], ax6_pos[3]])

print(' ')





'''
mPFC-behavior vs predicted M1-behavior
'''
print('Compare the Mutual information of mPFC-behavior and M1-behavior')

ax7 = plt.subplot(247)
ax8 = plt.subplot(248)
sp = {1: ax7, 2: ax8}

xl = [10, 0]
yl = [10, 0]

# Plot data for each model (RLPP and SLPP)
for modelIdx in [1, 2]:
    ax = sp[modelIdx]
    for ratIdx in range(6):
        # Error bar from min to max
        y_data = M1_act[modelIdx, ratIdx].squeeze()
        x_data = mPFC_act[ratIdx]
        ymin = np.min(y_data)
        ymax = np.max(y_data)
        ax.errorbar(x_data, ymin, yerr=[[0], [ymax - ymin]], color=color_mat[ratIdx])

        # Individual points
        for M1Idx in range(len(y_data)):
            ax.plot(x_data, y_data[M1Idx], 'o', color=color_mat[ratIdx],
                    markersize=4, markerfacecolor=color_mat[ratIdx])
    
    # Update axis limits
    temp_xlim = ax.get_xlim()
    temp_ylim = ax.get_ylim()
    xl = [min(xl[0], temp_xlim[0]) - 0.05, max(xl[1], temp_xlim[1]) + 0.05]
    yl = [min(yl[0], temp_ylim[0]), max(yl[1], temp_ylim[1])]

# Regression and statistical analysis
for modelIdx in [1, 2]:
    ax = sp[modelIdx]
    lines = [line for line in ax.get_lines()]
    xc = np.concatenate([line.get_xdata() for line in lines])
    yc = np.concatenate([line.get_ydata() for line in lines]).astype(np.float64)

    # Linear regression with confidence interval
    p, S = np.polyfit(xc, yc, 1, cov=True)
    x_pred = np.unique(xc)
    x_pred = np.insert(x_pred, 0, xl[0] + 0.05)
    x_pred = np.append(x_pred, xl[1] - 0.05)
    X = np.vstack([x_pred, np.ones(len(x_pred))]).T
    Y = X @ p

    # Confidence interval
    t_val = stats.t.ppf(0.975, len(xc) - 2)
    dy = t_val * np.sqrt(S[0,0]*(X[:,0]**2) + S[1,1] + 2*S[0,1]*X[:,0])

    ax.plot(x_pred, Y, '-', color=modelColor[modelIdx], linewidth=2)
    ax.fill_between(x_pred, Y - dy, Y + dy, color=modelColor[modelIdx], alpha=0.3)

    ax.plot([0., 0.8], [0., 0.8], '--k')
    ax.set_xlim([0, 0.8])
    ax.set_ylim([0, 0.8])
    ax.tick_params(labelsize=10)
    ax.tick_params(axis='both', which='both', direction='out')
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)

    # T-test
    t_stat, p_val = stats.ttest_rel(xc, yc, alternative='greater')
    if p_val < 0.05:
        model_name = 'RLPP' if modelIdx == 2 else 'SLPP'
        print(f'{model_name} M1 significantly lower than mPFC, p: {p_val:.4f}')

    # Ratio comparison
    ratio = np.sum(xc > yc) / len(xc)
    if modelIdx == 2:
        print(f'Ratio of RLPP M1 lower than mPFC: {ratio:.4f}')
    else:
        print(f'Ratio of SLPP M1 lower than mPFC: {ratio:.4f}')

# Labeling and adjusting positions
ax7.set_xlabel('${\\rm MI(mPFC;~Movement)}$', fontsize=10)
ax7.set_ylabel('${\\rm MI(RLPP;~Movement)}$', fontsize=10)
ax7.text(-0.2, 0.88, 'f', fontname='Times New Roman', fontweight='bold',
         fontsize=14, transform=ax7.transAxes)
ax7_pos = ax7.get_position().bounds
ax7.set_position([ax7_pos[0] + 0.01, ax7_pos[1], ax7_pos[2], ax7_pos[3]])

ax8.set_xlabel('${\\rm MI(mPFC;~Movement)}$', fontsize=10)
ax8.set_ylabel('${\\rm MI(SLPP;~Movement)}$', fontsize=10)
ax8.text(-0.2, 0.88, 'g', fontname='Times New Roman', fontweight='bold',
         fontsize=14, transform=ax8.transAxes)
ax8_pos = ax8.get_position().bounds
ax8.set_position([ax8_pos[0] + 0.02, ax8_pos[1], ax8_pos[2], ax8_pos[3]])

plt.show()