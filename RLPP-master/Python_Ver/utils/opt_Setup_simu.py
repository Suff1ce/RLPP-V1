def opt_Setup_simu(data):
    """
    Set training options based on the input data.
    """
    opt = {}
    # Data description
    opt["DataIndex"] = data["DataIndex"]
    opt["mPFCchannels"] = data["mPFC"].shape[0]
    if not isinstance(data["M1num_pre"], int):
        data["M1num_pre"] = data["M1num_pre"].item()    # In case it is read as a numpy array
    opt["M1index"] = data["M1order"][:data["M1num_pre"]]

    # Hyperparameters for Hawkes process
    opt["decay_parameter"] = 150
    # Hyperparameters for training
    opt["foldNum"] = 5            # Use five-fold cross validation
    opt["maxEpisode"] = 5000      # Maximum number of iterations
    opt["batchSize"] = 20         # Number of trials per batch
    opt["RelevantSpikes"] = 50    # Use longer history in simulation data
    opt["hiddenUnitNum"] = 64

    # Hyperparameters for reward design
    opt["discountFactor"] = 0.98
    opt["discountLength"] = 20    # In simulation data we use shorter discount length, since the movement is determined by the instantaneous firing rate
    opt["epsilon"] = 1.0          # Coefficient for inner reward

    # Hyperparameters for priori knowledge of firing rate
    opt["prioriM"] = 0.2
    opt["prioriN"] = 1

    # Initial data loader
    opt["DataLoaderCursor"] = 1
    opt["trainTrials"] = 0
    opt["testTrials"] = 0
    opt["NumberOfAllTrials"] = 0
    opt["NumberOfTrainTrials"] = 0
    opt["NumberOfTestTrials"] = 0

    # Train model
    opt["Mode"] = ""              # 'train' or 'test'

    # Log control
    opt["verbose"] = 4            # Control the output logs

    return opt
