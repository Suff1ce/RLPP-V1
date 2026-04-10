import numpy as np
import scipy.io
import os
from datetime import datetime
import string
import random

from utils import *
from model import *
from decoding import *

def SupervisedLearning(data, opt, inputEnsemble, M1Truth, Actions, Trials):
    """
    Supervised learning function translated from MATLAB.
    """

    # History records
    rewardHis = np.zeros(opt["maxEpisode"])
    crossEntropyHis = np.zeros(opt["maxEpisode"])
    MaxReward = -np.inf
    MinError = np.inf

    # Initialize network weights
    weightFromInputToHidden = 2 * np.random.rand(opt["hiddenUnitNum"], data["mPFCnum"] * opt["RelevantSpikes"] + 1) - 1
    weightFromHiddenToOutput = 2 * np.random.rand(data["M1num_pre"], opt["hiddenUnitNum"] + 1) - 1

    # Training loop
    if opt["verbose"] <= 3:
        print(f'----{opt["DataIndex"]} Sup Train start----')

    opt["Mode"] = 'train'
    for episode in range(opt["maxEpisode"]):
        # Get batch data
        batchInput, batchM1Truth, batchActions, opt = DataLoader(inputEnsemble, M1Truth, Actions, Trials, opt)
        NumOfSamples = batchInput.shape[1]
        inputUnit = np.vstack((batchInput, np.ones((1, NumOfSamples))))

        # Forward propagation
        pOutput, hiddenUnit, spkOutPredict = applynets(
            inputUnit, weightFromInputToHidden, weightFromHiddenToOutput, NumOfSamples
        )

        # Emulator: get predicted motor
        if opt["DataIndex"] == 'Simulations':
            _, sucRate, _ = emulator_simu(spkOutPredict, batchActions, opt["M1index"], data["his"], data["modelName"])
        else:
            _, sucRate, _, _ = emulator_real(spkOutPredict, batchActions, opt["M1index"], data["his"], data["modelName"])

        rewardHis[episode] = sucRate

        # Store history and print log
        crossEntropyHis[episode] = CrossEntropyError(
            np.vstack((1 - batchM1Truth, batchM1Truth)), np.vstack((1 - pOutput, pOutput)), NumOfSamples
        )

        if crossEntropyHis[episode] <= MinError:
            MinError = crossEntropyHis[episode]
            L2Weight = weightFromHiddenToOutput.copy()
            L1Weight = weightFromInputToHidden.copy()
            MinErrorEpisode = episode

        if rewardHis[episode] >= MaxReward:
            MaxReward = rewardHis[episode]
            L2WeightBestReward = weightFromHiddenToOutput.copy()
            L1WeightBestReward = weightFromInputToHidden.copy()
            MaxRewardEpisode = episode

        if opt["verbose"] <= 0:
            print(f'{episode+1}/{opt["maxEpisode"]}...Error {crossEntropyHis[episode]:.4f}...Reward {rewardHis[episode]:.4f}')

        # Get gradient
        WeightDelta1, WeightDelta2 = getgradient_sup(
            pOutput, batchM1Truth, hiddenUnit, inputUnit, weightFromHiddenToOutput, NumOfSamples
        )

        # Gradient descent update
        if opt['DataIndex'] == 'Simulations':
            lr = 0.3
        else:
            lr = 1.1 * (1 - episode / opt["maxEpisode"]) + 0.1
        weightFromHiddenToOutput += lr * (WeightDelta1 - 0 * weightFromHiddenToOutput)
        weightFromInputToHidden += lr * (WeightDelta2 - 0 * weightFromInputToHidden)

    # Test model
    opt["Mode"] = 'test'
    testInput, testM1Truth, testActions, opt = DataLoader(inputEnsemble, M1Truth, Actions, Trials, opt)
    TestSamples = testInput.shape[1]
    inputUnitTest = np.vstack((testInput, np.ones((1, TestSamples))))
    pOutputTest, hiddenUnitTest, spkOutPredictTest = applynets(
        inputUnitTest, L1Weight, L2Weight, TestSamples
    )

    if opt["DataIndex"] == 'Simulations':
        _, testSucRate, motorPerformTest = emulator_simu(spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"])
    else:
        _, testSucRate, motorPerformTest, _ = emulator_real(spkOutPredictTest, testActions, opt["M1index"], data["his"], data["modelName"])
    
    if opt["verbose"] <= 3:
        print(f'===={opt["DataIndex"]} Sup Test finish...Reward: {testSucRate:.4f}====')

    # Save results
    save_dict = {k: v for k, v in locals().items() if k not in ['data', 'inputEnsemble', 'M1Truth', 'Actions', 'Trials']}
    
    if opt['DataIndex'] == 'Simulations':
        targetFile = f'./Python_Ver/results/{opt["DataIndex"]}_Sup_{opt["testFold"]}.mat'
    else:
        randstr = string.ascii_lowercase + string.digits
        randId = f"{datetime.now().strftime('%b%d_%H%M')}_{''.join(random.choices(randstr, k=4))}"
        targetFile = f'./Python_Ver/results/{opt["DataIndex"]}/Sup_{opt["testFold"]}_{randId}.mat'
    
    os.makedirs(os.path.dirname(targetFile), exist_ok=True)
    scipy.io.savemat(targetFile, save_dict)

    return data, opt

