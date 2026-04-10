import numpy as np

def calculateTrialSucRate(success, test_actions):
    """
    Calculate the trial success rate based on success and test action sequences.
    
    Parameters:
    success (list or np.array): Binary array indicating success (1) or failure (0).
    test_actions (list or np.array): Action sequence where trial starts and ends.

    Returns:
    float: The success rate of trials.
    """
    trial_num = 0
    success_trial_num = 0
    start = None

    for time_idx in range(1, len(success)):
        if test_actions[time_idx - 1] == 0 and test_actions[time_idx] == 1:
            start = time_idx
            continue
        elif time_idx == len(success) - 1 or (test_actions[time_idx] > 1 and test_actions[time_idx + 1] == 0):
            if start is not None:
                stop = time_idx
                trial_num += 1
                success_count = np.sum(np.array(success[start:stop]) == 1)
                total_count = success_count + np.sum(np.array(success[start:stop]) == 0)
                if total_count > 0 and success_count / total_count > 0.7:
                    success_trial_num += 1

    return success_trial_num / trial_num if trial_num > 0 else 0
