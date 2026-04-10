import numpy as np
import matplotlib.pyplot as plt
from utils import MIcontinuous, gaussianSmooth

def sortM1neurons(M1num, M1, actions):
    # Assuming M1, actions, and M1num are provided as NumPy arrays
    M1delayMI = np.zeros((M1num, 71))
    startIdx = 0
    endIdx = 50

    for delay in range(startIdx, endIdx + 1):
        if delay == 0:
            M1end = None
        else:
            M1end = - delay
        for n in range(M1num):
            M1delayMI[n, delay - startIdx] = MIcontinuous(
                gaussianSmooth(M1[: M1end, n], 5),
                gaussianSmooth(actions[delay :], 5)   # delay - startIdx:end + startIdx, startIdx=0
            )

    M1maxMI = np.max(M1delayMI[:, startIdx:endIdx + 1], axis=1)
    M1delay = np.argmax(M1delayMI[:, startIdx:endIdx + 1], axis=1)
    MIsorted = np.sort(M1maxMI)[::-1]
    M1order = np.argsort(M1maxMI)[::-1]

    plt.figure()
    plt.plot(MIsorted)
    plt.title("Sorted Mutual Information (close the window to continue)")
    plt.xlabel("M1 indexes")
    plt.ylabel("Mutual Information")
    plt.show()

    return M1maxMI, M1delay, MIsorted, M1order
