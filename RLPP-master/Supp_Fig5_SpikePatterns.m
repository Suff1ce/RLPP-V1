%% show spike patterns by t-SNE
clear; clc;
addpath draw_figures/ utils/ model/ decoding/
load("trained_results/rat02_RL_results.mat", "folds")

%% define color
actionsColor = [178 178 178; 255 165 0; 128 0 128] / 255;

%% define figure
f = figure();
f.Units = 'centimeters';
f.Position = [1 1 15 15]; % [left bottom width height]
f.Color = 'w';

%% load data
DataName = "data/rat02.mat";
data = data_setup(DataName);
opt = opt_Setup_real(data);
rng('default');
[inputEnsemble,M1_truth,Actions,Trials,opt] = PreProcess( ...
    data.mPFC,data.M1,data.segment,data.trialNo,opt.decay_parameter,opt);
opt.testFold = 2;
trainFolds = 1:opt.foldNum; trainFolds(opt.testFold) = [];
opt.trainTrials = cell2mat(folds(trainFolds));
opt.NumberOfTrainTrials = length(opt.trainTrials);
opt.testTrials = folds{opt.testFold};
opt.NumberOfTestTrials = length(opt.testTrials);

opt.Mode = 'test';
[testInput,testM1_truth,testActions,opt] = DataLoader( ...
    inputEnsemble,M1_truth,Actions,Trials,opt);
TestSamples = length(testInput);
inputUnitTest = [testInput;ones(1,TestSamples)];
[~,~,~,M1_ensemble] = emulator_real(testM1_truth,testActions,opt.M1index,data.his,data.modelName);
Recordings.M1_ensemble = M1_ensemble;

%% Get RL outputs
load("trained_results/rat02_RL_results.mat", "L1WeightBestReward", "L2WeightBestReward", "MaxRewardEpisode")
[~, ~, spkOutPredictTest] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
[~,~,~,M1_ensemble] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
RL.M1_ensemble = M1_ensemble;

%% Get SL outputs
load("trained_results/rat02_Sup_results.mat", "L1Weight", "L2Weight")
[~, ~, spkOutPredictTest] = applynets(inputUnitTest,...
    L1Weight,L2Weight,TestSamples);
[~,~,~,M1_ensemble] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
SL.M1_ensemble = M1_ensemble;

%% run tSNE
rng('default')
X = cellfun(@(x) tsne(x'), [ ...
    {Recordings.M1_ensemble(:,testActions>0)}, ...
    {RL.M1_ensemble(:,testActions>0)}, ...
    {SL.M1_ensemble(:,testActions>0)} ...
], 'UniformOutput', false);
y = testActions(:,testActions>0);
actions = {'Rest', 'Press Low', 'Press High'};
y_str = arrayfun(@(x) actions{x}, y, 'UniformOutput', false);

%% show results
subplot(331)
colorSeq = unique(y, 'stable');
h = gscatter(X{1}(:,1),X{1}(:,2),y_str',actionsColor(colorSeq,:));
h(1).MarkerSize=1; h(2).MarkerSize=1; h(3).MarkerSize=1;

strings = {'Rest', 'Press Low', 'Press High'};
annotation('rectangle', [0.06 0.882 0.15 0.07], 'FaceColor', 'w');
hold on;
for i = 1:3
    annotation('ellipse', [0.07, 0.89+(i-1)*0.02, 0.015, 0.015], 'FaceColor', actionsColor(4-i,:), 'EdgeColor', 'none');
    annotation('textbox', [0.08, 0.885+(i-1)*0.02, 0.14, 0.015], 'String', strings{4-i}, ...
      'EdgeColor', 'none', 'FontSize', 10, 'Interpreter','latex', 'VerticalAlignment','middle');
end
hold off;
legend off
set(gca, 'TickLabelInterpreter', 'latex')
title('Recorded Spike Pattern','FontName','Times New Roman','Units','normalized', 'Position',[0.5 1.12], 'FontSize',12, 'FontWeight','normal')
text(-0.48, 0.5, 'Rat 02','FontName','Times New Roman','Units','normalized', 'FontSize',12)

subplot(332)
colorSeq = unique(y, 'stable');
h = gscatter(X{2}(:,1),X{2}(:,2),y_str',actionsColor(colorSeq,:));
h(1).MarkerSize=1; h(2).MarkerSize=1; h(3).MarkerSize=1;
legend off
set(gca, 'TickLabelInterpreter', 'latex')
title('Spike Pattern by RLPP','FontName','Times New Roman','Units','normalized', 'Position',[0.5 1.12], 'FontSize',12, 'FontWeight','normal')

subplot(333)
colorSeq = unique(y, 'stable');
h = gscatter(X{3}(:,1),X{3}(:,2),y_str',actionsColor(colorSeq,:));
h(1).MarkerSize=1; h(2).MarkerSize=1; h(3).MarkerSize=1;
legend off
set(gca, 'TickLabelInterpreter', 'latex')
title('Spike Pattern by SLPP','FontName','Times New Roman','Units','normalized', 'Position',[0.5 1.12], 'FontSize',12, 'FontWeight','normal')
