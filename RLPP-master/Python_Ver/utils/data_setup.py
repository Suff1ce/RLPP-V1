from scipy.io import loadmat

def data_setup(DataName):
    # Load specified variables from the MAT file
    data = loadmat(DataName, variable_names=["mPFC", "mPFCnum",
                                               "M1", "M1num", "M1order", "M1num_pre",
                                               "segment", "trialNo", "his", "modelName", "DataIndex"])
    # Transpose mPFC and M1 to match MATLAB orientation
    if data["mPFC"].shape[0] > data["mPFC"].shape[1]:
        data["mPFC"] = data["mPFC"].T
    if data["M1"].shape[0] > data["M1"].shape[1]:
        data["M1"] = data["M1"].T
    data["mPFCnum"] = data["mPFCnum"].item()
    data["M1num"]   = data["M1num"].item()

    if "M1num_pre" not in data.keys():
        data["M1num_pre"] = 2
    if not isinstance(data["M1num_pre"], int):
        data["M1num_pre"] = data["M1num_pre"].item() 
    data["his"] = data["his"].item()
    data["M1order"] = data["M1order"].flatten()
    data["segment"] = data["segment"].flatten()
    data["trialNo"] = data["trialNo"].flatten()

    # incase data is not a string, convert it to string
    if not isinstance(data["DataIndex"], str):
        data["DataIndex"] = data["DataIndex"].item()
    if "modelName" in data.keys() and isinstance(data["modelName"], list):
        data["modelName"] = data["modelName"][0]
    return data
