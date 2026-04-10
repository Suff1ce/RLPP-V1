clc;clear;close all;
addpath model/ utils/ decoding/ training/

if ~exist('./results', 'dir')
    mkdir('./results');
end

%% model training
DataName = "data/rat02.mat";

data = data_setup(DataName); % load data
data.M1num_pre = 4;
data.modelName = 'decodingModel_manual';
dta.his = 50;
data.DataIndex = [data.DataIndex, 'Manual'];

opt = opt_Setup_real(data); % set options

if ~exist(['./results/', opt.DataIndex,'/'], 'dir')
    mkdir(['./results/', opt.DataIndex,'/']);
end

rng('default');
% PreProcess: get mPFC ensemble for every time bin
[inputEnsemble,M1_truth,Actions,Trials,opt] = PreProcess( ...
    data.mPFC,data.M1,data.segment,data.trialNo,opt.decay_parameter,opt);

sch = 'RL';

for testFold = 1:opt.foldNum  % explore testFold in 1:5 for 5-fold cross validation
    opt.testFold = testFold;
    trainFolds = 1:opt.foldNum; trainFolds(testFold) = [];
    opt.trainTrials = cell2mat(opt.folds(trainFolds));
    opt.NumberOfTrainTrials = length(opt.trainTrials);
    opt.testTrials = opt.folds{testFold};
    opt.NumberOfTestTrials = length(opt.testTrials);

    disp(['----', opt.DataIndex,' ',sch,' fold ',num2str(opt.testFold),' Train start----']);
    parfor i = 1:opt.ReTrainTimes
        s = rng;
        RLPP(data,opt,s,inputEnsemble,M1_truth,Actions,Trials);
    end

    paths = dir(['results/',data.DataIndex,'/',sch,'_',num2str(opt.testFold),'_','*.mat']);
    SucRate = zeros(1,length(paths));
    testSuc = zeros(1,length(paths));
    for i=1:length(paths)
        load(['results/',data.DataIndex,'/',paths(i).name],"MaxReward","testSucRate")
        SucRate(i) = MaxReward;
        testSuc(i) = testSucRate;
    end
    [best,I] = max(SucRate);
    source = ['results/',data.DataIndex,'/',paths(I).name];
    destination = ['results/', data.DataIndex, '_', sch, '_', num2str(opt.testFold), '.mat'];
    copyfile(source, destination, 'f')
    disp([destination, ' testSuc: ', num2str(testSuc(I))])
end
