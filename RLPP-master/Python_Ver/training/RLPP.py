import numpy as np
import scipy.io
import os
from datetime import datetime
import string
import random

from utils import *
from model import *
from decoding import *


def RLPP(data, opt, inputEnsemble, M1Truth, Actions, Trials):
    """
    Reinforcement Learning for Predicting M1 Spikes.
    """

    # History records
    rewardHis = np.zeros(opt["maxEpisode"])
    crossEntropyHis = np.zeros(opt["maxEpisode"])
    MaxReward = -np.inf

    # Network weights initialization
    weightFromInputToHidden = 2 * np.random.rand(opt["hiddenUnitNum"], data["mPFCnum"] * opt["RelevantSpikes"] + 1) - 1
    weightFromHiddenToOutput = 2 * np.random.rand(data["M1num_pre"], opt["hiddenUnitNum"] + 1) - 1

    # Training
    if opt["verbose"] <= 3:
        print(f'----{opt["DataIndex"]} RL Train start----')

    opt["Mode"] = 'train'

    for episode in range(opt["maxEpisode"]):
        # Get batch data
        batchInput, batchM1Truth, batchActions, opt = DataLoader(inputEnsemble, M1Truth, Actions, Trials, opt)
        NumOfSamples = batchInput.shape[1]
        inputUnit = np.vstack([batchInput, np.ones((1, NumOfSamples))])

        # Forward pass to get spikes
        pOutput, hiddenUnit, spkOutPredict = applynets_priori(
            inputUnit, weightFromInputToHidden, weightFromHiddenToOutput, NumOfSamples, 
            episode, opt["prioriM"], opt["prioriN"]
        )

        # Emulator: get predicted motor
        if opt["DataIndex"] == 'Simulations':
            success, sucRate, motorPerform = emulator_simu(
                spkOutPredict, batchActions, opt["M1index"], data["his"], data["modelName"]
            )
        else:
            success, sucRate, motorPerform, _ = emulator_real(
                spkOutPredict, batchActions, opt["M1index"], data["his"], data["modelName"]
            )

        rewardHis[episode] = sucRate

        # Inner reward calculation for less frequent motor actions
        n_motor1 = np.sum(motorPerform == 1) + 1
        n_motor2 = np.sum(motorPerform == 2) + 1
        n_motor3 = np.sum(motorPerform == 3) + 1
        n_max = max(n_motor1, n_motor2, n_motor3)
        innerReward = (motorPerform == 1) * (n_max/n_motor1 - 1) + \
                      (motorPerform == 2) * (n_max/n_motor2 - 1) + \
                      (motorPerform == 3) * (n_max/n_motor3 - 1)

        # Discounted return calculation
        reward = success + opt["epsilon"] * (1 - episode / opt["maxEpisode"]) * innerReward
        temp = reward.copy()
        temp[np.isnan(reward)] = 0

        # Convolution for smoothing
        discountFilter = opt['discountFactor'] ** np.arange(opt['discountLength']-1, -1, -1)
        discountFilter /= opt['discountLength']
        fullConv = np.convolve(temp, discountFilter, mode='full')
        smoothedReward = fullConv[-len(reward):]
        smoothedReward[~np.isnan(reward)] = (smoothedReward[~np.isnan(reward)] - 
                                            np.mean(smoothedReward[~np.isnan(reward)])) / \
                                           np.std(smoothedReward[~np.isnan(reward)])
        smoothedReward[np.isnan(reward)] = 0

        # Cross entropy calculation
        crossEntropyHis[episode] = CrossEntropyError(
            np.vstack([1 - batchM1Truth, batchM1Truth]),
            np.vstack([1 - pOutput, pOutput]),
            NumOfSamples
        )

        # Update best weights
        if rewardHis[episode] >= MaxReward:
            MaxReward = rewardHis[episode]
            L2WeightBestReward = weightFromHiddenToOutput.copy()
            L1WeightBestReward = weightFromInputToHidden.copy()
            MaxRewardEpisode = episode
        
        if opt["verbose"] <= 0:
            print(f'{episode+1}/{opt["maxEpisode"]}...Error {crossEntropyHis[episode - 1]:.4f}...Reward {rewardHis[episode - 1]:.4f}')

        # Get gradient
        WeightDelta1, WeightDelta2 = getgradient(
            smoothedReward, pOutput, spkOutPredict, hiddenUnit, inputUnit,
            weightFromHiddenToOutput, NumOfSamples
        )

        # Learning rate adjustment
        if opt['DataIndex'] == 'Simulations':
            lr = 0.1 * (1 - episode/opt['maxEpisode']) + 0.5
        else:
            lr = 0.7 * (1 - episode/opt['maxEpisode']) + 0.5
        
        # Update weights
        weightFromHiddenToOutput += lr * (WeightDelta1 - 0 * weightFromHiddenToOutput)
        weightFromInputToHidden += lr * (WeightDelta2 - 0 * weightFromInputToHidden)

        # Reinitialize weights if necessary
        if (episode + 1) % 100 == 0:
            stds = np.std(pOutput, axis=1)
            min_std_idx = np.argmin(stds)
            if stds[min_std_idx] < 0.01:
                weightFromHiddenToOutput[min_std_idx, :] = 2 * np.random.rand(opt['hiddenUnitNum'] + 1) - 1

    # Test model
    opt["Mode"] = 'test'
    testInput, testM1Truth, testActions, opt = DataLoader(inputEnsemble, M1Truth, Actions, Trials, opt)
    NumOfSamples = testInput.shape[1]
    inputUnitTest = np.vstack([testInput, np.ones((1, NumOfSamples))])

    pOutputTest, hiddenUnitTest, spkOutPredictTest = applynets_priori(
        inputUnitTest, L1WeightBestReward, L2WeightBestReward, NumOfSamples, MaxRewardEpisode,
        opt["prioriM"], opt["prioriN"]
    )

    if opt["DataIndex"] == 'Simulations':
        _, testSucRate, motorPerformTest = emulator_simu(spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"])
    else:
        _, testSucRate, motorPerformTest, _ = emulator_real(spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"])
    
    if opt["verbose"] <= 3:
        print(f'===={opt["DataIndex"]} RL Test finish...Reward: {testSucRate:.4f}====')

    # Save results
    save_dict = {k: v for k, v in locals().items() if k not in ['data', 'inputEnsemble', 'M1Truth', 'Actions', 'Trials']}
    
    if opt['DataIndex'] == 'Simulations':
        targetFile = f'./Python_Ver/results/{opt["DataIndex"]}_RL_{opt["testFold"]}.mat'
    else:
        randstr = string.ascii_lowercase + string.digits
        randId = f"{datetime.now().strftime('%b%d_%H%M')}_{''.join(random.choices(randstr, k=4))}"
        targetFile = f'./Python_Ver/results/{opt["DataIndex"]}/RL_{opt["testFold"]}_{randId}.mat'
    
    os.makedirs(os.path.dirname(targetFile), exist_ok=True)
    scipy.io.savemat(targetFile, save_dict)
    
    return data, opt
