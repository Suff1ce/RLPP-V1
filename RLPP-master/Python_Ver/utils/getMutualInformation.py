import numpy as np
import scipy.io as sio

def getMutualInformation(DataNameList, bestFold, MIcontinuous, gaussianSmooth):
    M1_act = [[], [], []]  # List of 3 lists to store results for each type
    M1_mPFC = [[], [], []]  # List of 3 lists to store results for each type
    mPFC_act = []

    # Loop through the data indices
    for dataIdx in range(6):
        # Load RL and SL results for the current data index
        RL = sio.loadmat(f'results/{DataNameList[dataIdx]}_RL_{bestFold[dataIdx]}.mat')
        SL = sio.loadmat(f'results/{DataNameList[dataIdx]}_Sup_{bestFold[dataIdx]}.mat')

        # Initialize arrays
        M1_act_real = np.zeros(len(RL['opt']['M1index'][0][0]))
        M1_act_RLPP = np.zeros(len(RL['opt']['M1index'][0][0]))
        M1_act_SLPP = np.zeros(len(RL['opt']['M1index'][0][0]))
        M1_mPFC_real_all = np.zeros((len(RL['opt']['M1index'][0][0]), RL['testInput'].shape[0]))
        M1_mPFC_RLPP_all = np.zeros((len(RL['opt']['M1index'][0][0]), RL['testInput'].shape[0]))
        M1_mPFC_SLPP_all = np.zeros((len(RL['opt']['M1index'][0][0]), RL['testInput'].shape[0]))
        M1_mPFC_real = np.zeros(RL['testInput'].shape[0])
        M1_mPFC_RLPP = np.zeros(RL['testInput'].shape[0])
        M1_mPFC_SLPP = np.zeros(RL['testInput'].shape[0])
        mPFC_act_rat = np.zeros(RL['testInput'].shape[0])

        # Compute mutual information for M1 activations and mPFC activations
        for M1idx in range(len(RL['opt']['M1index'][0][0])):
            M1_act_real[M1idx] = MIcontinuous(gaussianSmooth(RL['testM1_truth'][M1idx, RL['testActions'] > 0], 10), RL['testActions'][RL['testActions'] > 0])
            M1_act_RLPP[M1idx] = MIcontinuous(gaussianSmooth(RL['spkOutPredictTest'][M1idx, RL['testActions'] > 0], 10), RL['testActions'][RL['testActions'] > 0])
            M1_act_SLPP[M1idx] = MIcontinuous(gaussianSmooth(SL['spkOutPredictTest'][M1idx, RL['testActions'] > 0], 10), RL['testActions'][RL['testActions'] > 0])

            # Compute mPFC activations
            for n in range(RL['testInput'].shape[0]):
                M1_mPFC_real_all[M1idx, n] = MIcontinuous(gaussianSmooth(RL['testM1_truth'][M1idx, RL['testActions'] > 0], 10), RL['testInput'][n, RL['testActions'] > 0])
                M1_mPFC_RLPP_all[M1idx, n] = MIcontinuous(gaussianSmooth(RL['spkOutPredictTest'][M1idx, RL['testActions'] > 0], 10), RL['testInput'][n, RL['testActions'] > 0])
                M1_mPFC_SLPP_all[M1idx, n] = MIcontinuous(gaussianSmooth(SL['spkOutPredictTest'][M1idx, RL['testActions'] > 0], 10), RL['testInput'][n, RL['testActions'] > 0])

            # Sorting the values and calculating the mean for top 25%
            sorted_values = np.sort(M1_mPFC_real_all[M1idx, :])[::-1]
            M1_mPFC_real[M1idx] = np.mean(sorted_values[:int(0.25 * len(sorted_values))])

            sorted_values = np.sort(M1_mPFC_RLPP_all[M1idx, :])[::-1]
            M1_mPFC_RLPP[M1idx] = np.mean(sorted_values[:int(0.25 * len(sorted_values))])

            sorted_values = np.sort(M1_mPFC_SLPP_all[M1idx, :])[::-1]
            M1_mPFC_SLPP[M1idx] = np.mean(sorted_values[:int(0.25 * len(sorted_values))])

        # Compute the mPFC activation for the rat
        for n in range(RL['testInput'].shape[0]):
            mPFC_act_rat[n] = MIcontinuous(RL['testInput'][n, RL['testActions'] > 0], RL['testActions'][RL['testActions'] > 0])

        # Sorting and calculating the mean for mPFC activations
        sorted_values = np.sort(mPFC_act_rat)[::-1]
        mPFC_act.append(np.mean(sorted_values[:int(0.25 * len(sorted_values))]))

        # Store the results for this data index
        M1_act[0].append(M1_act_real)
        M1_act[1].append(M1_act_RLPP)
        M1_act[2].append(M1_act_SLPP)

        M1_mPFC[0].append(M1_mPFC_real)
        M1_mPFC[1].append(M1_mPFC_RLPP)
        M1_mPFC[2].append(M1_mPFC_SLPP)

    # Save the results into a .mat file
    sio.savemat('results/mutualInformation_testSet.mat', {
        'M1_act': M1_act,
        'M1_mPFC': M1_mPFC,
        'mPFC_act': mPFC_act
    })

