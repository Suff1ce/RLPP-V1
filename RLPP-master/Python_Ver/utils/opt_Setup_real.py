def opt_Setup_real(data):
    """
    Set training options based on the input data.
    """
    opt = {}
    # Data description
    opt["DataIndex"] = data["DataIndex"]
    opt["mPFCchannels"] = data["mPFC"].shape[0]
    if not isinstance(data["M1num_pre"], int):
        data["M1num_pre"] = data["M1num_pre"].item()
    opt["M1index"] = data["M1order"][:data["M1num_pre"]]
    
    # Hyperparameters for Hawkes process
    opt["decay_parameter"] = 150
    
    # Hyperparameters for training
    opt["foldNum"] = 5  # Use five-fold cross-validation
    opt["maxEpisode"] = int(5e3)  # Maximum number of iterations
    opt["batchSize"] = 20  # Number of trials per batch (20 trials ~ 30s)
    opt["RelevantSpikes"] = 5  # Past 5 mPFC spikes should cover ~500ms history
    opt["hiddenUnitNum"] = 64
    
    # Hyperparameters for reward design
    opt["discountFactor"] = 0.98
    opt["discountLength"] = 100
    opt["epsilon"] = 1.0  # Coefficient for inner reward
    
    # Hyperparameters for priori knowledge of firing rate
    opt["prioriM"] = 0.2
    opt["prioriN"] = 1.5
    
    # Initial data loader
    opt["DataLoaderCursor"] = 1
    opt["trainTrials"] = 0
    opt["testTrials"] = 0
    opt["NumberOfAllTrials"] = 0
    opt["NumberOfTrainTrials"] = 0
    opt["NumberOfTestTrials"] = 0
    
    # Train model mode (train or test)
    opt["Mode"] = ""
    
    # Log control
    opt["verbose"] = 4  # Control the output logs
    
    # Re-training
    opt["ReTrainTimes"] = 32  # Run multiple times to avoid local minimum
    
    return opt
