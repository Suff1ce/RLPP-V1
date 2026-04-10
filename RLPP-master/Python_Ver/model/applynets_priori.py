import numpy as np

def applynets_priori(input_unit, weight_input_hidden, weight_hidden_output, 
                     num_samples, episode, M, N):
    """
    Artificial Neural Network (ANN) to predict M1 spikes.

    Parameters:
    input_unit : ndarray
        Input layer activations.
    weight_input_hidden : ndarray
        Weights from input to hidden layer.
    weight_hidden_output : ndarray
        Weights from hidden to output layer.
    num_samples : int
        Number of samples.
    episode : float
        Scaling factor for standard deviation adjustment.
    M : float
        Prior knowledge factor M.
    N : float
        Prior knowledge factor N.

    Returns:
    p_output : ndarray
        Processed output probabilities.
    hidden_unit : ndarray
        Hidden layer activations.
    choose_action : ndarray
        Binary action selection based on probabilities.
    """
    def my_sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Compute hidden layer activations
    hidden_unit = np.vstack((my_sigmoid(np.dot(weight_input_hidden, input_unit)), 
                             np.ones((1, num_samples))))

    # Compute output layer activations
    output_unit = np.dot(weight_hidden_output, hidden_unit)
    p_output = my_sigmoid(output_unit)

    # Adjust probabilities with prior knowledge
    temp = (episode+1) * np.std(p_output, axis=1, keepdims=True)        # episode+1, as episode starts from 0, 25-04-01
    temp[temp < 1e-3] = 1e-3  # Prevent division by very small numbers
    p_output = (temp * p_output + M) / (temp + N)

    # Generate action selection based on probabilities
    choose_action = np.random.rand(*output_unit.shape) <= p_output

    return p_output, hidden_unit, choose_action
