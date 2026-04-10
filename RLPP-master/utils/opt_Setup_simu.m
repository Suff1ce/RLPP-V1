function opt = opt_Setup_simu(data)
%% set training options
% 
%% data description
opt.DataIndex = data.DataIndex;
opt.mPFCchannels = size(data.mPFC,1);
opt.M1index = data.M1order(1:data.M1num_pre);
%% hyperparameters for Hawkes process
opt.decay_parameter = 150;
%% hyperparameters for training
opt.foldNum = 5;            % use five-fold cross validation
opt.maxEpisode = 5e3;       % maximum number of iteration
opt.batchSize = 20;         % No of trials per batch
opt.RelevantSpikes = 50;   % in simulation data we use longer history
opt.hiddenUnitNum = 64;
%% hyperparameters for reward design
opt.discountFactor = 0.98;
opt.discountLength = 20;   % in simulation data we use shorter discount length, since the movement is determined by the instantaneous firing rate
opt.epsilon = 1.0;         % coefficient for inner reward
%% hyperparameters for priori knowledge of firing rate
opt.prioriM = 0.2;
opt.prioriN = 1;
%% initial data loader
opt.DataLoaderCursor = 1;
opt.trainTrials = 0;
opt.testTrials = 0;
opt.NumberOfAllTrials = 0;
opt.NumberOfTrainTrials = 0;
opt.NumberOfTestTrials = 0;
%% train model
opt.Mode = '';              % train or test
%% log control
opt.verbose = 4;            % control the output logs
end
