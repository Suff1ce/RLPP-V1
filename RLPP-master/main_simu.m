clc;clear;close all;
addpath model/ utils/ decoding/ training/ draw_figures/

if ~exist('./results', 'dir')
    mkdir('./results');
end

%% data synthesis
disp("Start data synthesis")
simuData;
drawnow;

%% model training
DataName = "data/simulations.mat";
data = data_setup(DataName); % load data
opt = opt_Setup_simu(data); % set options

rng('default');
% PreProcess: get mPFC ensemble for every time bin
[inputEnsemble,M1_truth,Actions,Trials,opt] = PreProcess( ...
    data.mPFC,data.M1,data.segment,data.trialNo,opt.decay_parameter,opt);
for RLorSL = [0 1]
    if RLorSL==0
        sch = 'RL';
    else
        sch = 'Sup';
    end

    for testFold = 1
        opt.testFold = testFold;
        trainFolds = 1:opt.foldNum; trainFolds(testFold) = [];
        opt.trainTrials = cell2mat(opt.folds(trainFolds));
        opt.NumberOfTrainTrials = length(opt.trainTrials);
        opt.testTrials = opt.folds{testFold};
        opt.NumberOfTestTrials = length(opt.testTrials);
    
        disp(['----', opt.DataIndex,' ',sch,' fold ',num2str(opt.testFold),' Train start----']);
        s = rng;
        if (RLorSL==0)
            RLPP(data,opt,s,inputEnsemble,M1_truth,Actions,Trials);
        else
            SupervisedLearning(data,opt,s,inputEnsemble,M1_truth,Actions,Trials);
        end
    end
end

%% show results
showSimuResults;
