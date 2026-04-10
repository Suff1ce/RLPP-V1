function data = data_setup(DataName)
%% set data
data = load(DataName, ...
    "mPFC","mPFCnum",...
    "M1","M1num","M1order","M1num_pre",...
    "segment","trialNo","his","modelName","DataIndex");
data.mPFC = data.mPFC';
data.M1 = data.M1';

end
