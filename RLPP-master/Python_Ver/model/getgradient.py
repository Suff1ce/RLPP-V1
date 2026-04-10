import numpy as np

def getgradient(reward, p_output, spk_out_predict, hidden_unit, input_unit, 
                 weight_hidden_output, num_samples):
    """
    Compute gradient for updating neural network weights.

    Parameters:
    reward : ndarray
        Reward signal.
    p_output : ndarray
        Predicted output probabilities.
    spk_out_predict : ndarray
        Ground truth spike output.
    hidden_unit : ndarray
        Hidden layer activations.
    input_unit : ndarray
        Input layer activations.
    weight_hidden_output : ndarray
        Weights from hidden to output layer.
    num_samples : int
        Number of samples.

    Returns:
    weight_delta1 : ndarray
        Gradient for output layer weights.
    weight_delta2 : ndarray
        Gradient for hidden layer weights.
    """
    delta = reward * (spk_out_predict - p_output)
    weight_delta1 = (delta @ hidden_unit.T) / num_samples
    
    weight_delta2 = (hidden_unit * (1 - hidden_unit) * (weight_hidden_output.T @ delta)) @ input_unit.T / num_samples
    weight_delta2 = weight_delta2[:-1, :]  # Remove the last row 
    
    return weight_delta1, weight_delta2
