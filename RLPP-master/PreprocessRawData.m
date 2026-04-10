clear; clc; close all;
addpath utils/ preprocess/

% CHANGE this for six rats
DataIndex = 'rat01';
DataIndexShort = '01';

OriginData = ['raw_data/', DataIndex, '.mat'];
TargetData = ['data/', DataIndex, '.mat'];

%% Load data
if ~exist(TargetData, "file")
    [M1,mPFC,M1num,mPFCnum,M1channelName,mPFCchannelName,segment,trialNo,actions] = spikeTime2Train(OriginData);
else
    load(TargetData, "M1","mPFC","M1num","mPFCnum","segment","trialNo","actions");
end

%% Rank the neurons in decsending order based on the mutual information between neural activity and behavior
sortM1neurons;

%% prepare for decoder training
his = 60; % 600ms history for M1 ensemble
temp=segment((his+1):end);
y=temp(temp>0);
y_onehot=(y'==1:3)'; % only 3 types of behavior

x_full = zeros((his+1)*M1num,sum(temp>0));
for i=0:his
    M1ensemble_temp = M1(((his+1)-i):(end-i),:)';
    x_full(i*M1num+1:(i+1)*M1num,:) = M1ensemble_temp(:,temp>0);
end

%% search the best number of hidden units for the decoder
searchHiddenSize

%% select the proper hidden size & search for the top-N M1 neurons as decoder input
hiddenLayerSize = 2^3; % determined based on the results of last section for each rat
searchM1number

%% select the number of M1 neurons for transregional spike prediction & save the decoding model
M1num_pre = 2; % determined based on the results of last section for each rat

[~,bestIndex]=min(testPercentErrors(M1num_pre,:));
  disp(['Decoding accuracy: ', num2str(1-percentErrors(M1num_pre,bestIndex))]);
% save the best model
modelName = ['decodingModel_',DataIndexShort];
genFunction(netVec{M1num_pre,bestIndex},['decoding/',modelName]);

%% save preprocessed data for transregional spike prediction model traning
save(TargetData, ...
    "DataIndex","his", ...
    "M1","M1num","M1order","M1num_pre", ...
    "mPFC","mPFCnum", ...
    "segment","trialNo", ...
    "modelName");
