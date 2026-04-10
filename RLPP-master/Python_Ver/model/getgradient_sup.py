import numpy as np

def getgradient_sup(p_output, spk_out, hidden_unit, input_unit, 
                     weight_hidden_output, num_samples):
    """
    Compute supervised gradient for updating neural network weights.

    Parameters:
    p_output : ndarray
        Predicted output probabilities.
    spk_out : ndarray
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
    delta = spk_out - p_output

    weight_delta1 = np.dot(delta, hidden_unit.T) / num_samples
    
    temp = hidden_unit * (1 - hidden_unit) * np.dot(weight_hidden_output.T, delta)
    weight_delta2 = np.dot(temp, input_unit.T) / num_samples
    weight_delta2 = weight_delta2[:-1, :]   
    
    return weight_delta1, weight_delta2
