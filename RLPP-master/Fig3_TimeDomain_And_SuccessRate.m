%% show time domain of rat01, test set
clear; clc;
addpath draw_figures/ utils/ model/ decoding/
load("trained_results/rat01_RL_results.mat", "folds")

%% define color
modelColor = [250 0 0; 54 56 131; 103 146 70] / 255;
actionsColor = [178 178 178; 255 165 0; 128 0 128] / 255;

%% load data
DataName = "data/rat01.mat";
data = data_setup(DataName);
opt = opt_Setup_real(data);
rng('default');
[inputEnsemble,M1_truth,Actions,Trials,opt] = PreProcess( ...
    data.mPFC,data.M1,data.segment,data.trialNo,opt.decay_parameter,opt);
opt.testFold = 5;
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

%% Get RL outputs
load("trained_results/rat01_RL_results.mat", "L1WeightBestReward", "L2WeightBestReward", "MaxRewardEpisode")
[pOutputTest, ~, spkOutPredictTest] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
[success,testSucRate,motor_perform_test] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);

RL.motor_perform_test = motor_perform_test;
RL.pOutputTest = pOutputTest;
RL.spkOutPredictTest = spkOutPredictTest;
RL.testSucRate = testSucRate;
RL.success = success;

%% Get SL outputs
load("trained_results/rat01_Sup_results.mat", "L1Weight", "L2Weight")
[pOutputTest, ~, spkOutPredictTest] = applynets(inputUnitTest,...
    L1Weight,L2Weight,TestSamples);
[success,testSucRate,motor_perform_test] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);

SL.motor_perform_test = motor_perform_test;
SL.pOutputTest = pOutputTest;
SL.spkOutPredictTest = spkOutPredictTest;
SL.testSucRate = testSucRate;
SL.success = success;

%% Calculate Success rate
disp(['RL time-bin success rate of Rat 01: ', num2str(RL.testSucRate)])
disp(['RL trial success rate of Rat 01: ', num2str(calculateTrialSucRate(RL.success, testActions))])
disp(' ')
disp(['SL time-bin success rate of Rat 01: ', num2str(SL.testSucRate)])
disp(['SL trial success rate of Rat 01: ', num2str(calculateTrialSucRate(SL.success, testActions))])

%% Draw plots
f = figure();
f.Units = 'centimeters';
f.Position = [1 1 18 17]; % [left bottom width height]
f.Color = 'w';
timeIndex = 0.01:0.01:20;
timeRange = [0.01 20];
dataIndex = 1:2000;

%% Neural data
%% Neuron 1
% spikes
annotation('textbox',...
    [0.01 0.745 0.04 0.027],...
    'String',{'\textbf{a(i)}'},...
    'FontWeight','bold',...
    'FontSize',7, 'interpreter','latex',...
    'FitBoxToText','off',...
    'EdgeColor','none');
subplot('Position',[0.1 0.68 0.87 0.09])
plotSpikes(timeIndex, testM1_truth(1,dataIndex), 1.1, modelColor(1,:))
hold on
plotSpikes(timeIndex, RL.spkOutPredictTest(1,dataIndex), 1, modelColor(2,:))
plotSpikes(timeIndex, SL.spkOutPredictTest(1,dataIndex), 0.9, modelColor(3,:))
hold off
box off
set(gca, 'YColor', 'none')
set(gca, 'XColor', 'none')
set(gca, 'TickLength', [0 0])
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',6.5)
ylim([0.8 1.2])
xlim(timeRange)
title('\textbf{M1 Neuron 1}','interpreter','latex','FontSize',7,'FontWeight','bold', ...
    'Units','normalized', 'HorizontalAlignment','center','VerticalAlignment','middle','Position',[0.5 1.0 0])
ylabel('${\rm Spikes}$','FontSize',7,'units','normalized','interpreter','latex', 'Rotation',90, 'Color','k');
% firing probabilities
annotation('textbox',...
    [0.01 0.67 0.04 0.027],...
    'String',{'\textbf{a(ii)}'},...
    'FontWeight','bold',...
    'FontSize',7, 'interpreter','latex',...
    'FitBoxToText','off',...
    'EdgeColor','none');
subplot('Position',[0.1 0.57 0.87 0.10])
plot(timeIndex, smoothdata(testM1_truth(1,dataIndex),'gaussian',51), 'Color',modelColor(1,:), 'LineWidth',1);
hold on
plot(timeIndex, RL.pOutputTest(1,dataIndex), 'Color',modelColor(2,:), 'LineWidth',1)
plot(timeIndex, SL.pOutputTest(1,dataIndex), 'Color',modelColor(3,:), 'LineWidth',1)
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',6.5)
set(gca, 'XColor', 'k', 'LineWidth', 1)
xlabel('Time (s)','FontSize',7,'units','normalized','interpreter','latex');
ylabel('${\rm Firing ~Probability}$','FontSize',7,'units','normalized','interpreter','latex', 'Rotation',90);
xlim(timeRange)
legend({'Recordings', 'RLPP', 'SLPP'}, ...
    'interpreter','latex','Orientation','horizontal', ...
    'Position',[0.61,0.785,0.36,0.02],'Box','off','FontSize',7);
%% Neuron 2
% spikes
annotation('textbox',...
    [0.01 0.465 0.04 0.027],...
    'String',{'\textbf{a(iii)}'},...
    'FontWeight','bold',...
    'FontSize',7, 'interpreter','latex',...
    'FitBoxToText','off',...
    'EdgeColor','none');
subplot('Position',[0.1 0.4 0.87 0.09])
plotSpikes(timeIndex, testM1_truth(2,dataIndex), 1.1, modelColor(1,:))
hold on
plotSpikes(timeIndex, RL.spkOutPredictTest(2,dataIndex), 1, modelColor(2,:))
plotSpikes(timeIndex, SL.spkOutPredictTest(2,dataIndex), 0.9, modelColor(3,:))
hold off
box off
set(gca, 'YColor', 'none')
set(gca, 'XColor', 'none')
set(gca, 'TickLength', [0 0])
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',6.5)
ylim([0.8 1.2])
xlim(timeRange)
title('\textbf{M1 Neuron 2}','interpreter','latex','FontSize',7,'FontWeight','bold', ...
    'Units','normalized', 'HorizontalAlignment','center','VerticalAlignment','middle','Position',[0.5 1.0 0])
ylabel('${\rm Spikes}$','FontSize',7,'units','normalized','interpreter','latex', 'Rotation',90, 'Color','k');
% firing probabilities
annotation('textbox',...
    [0.01 0.39 0.04 0.027],...
    'String',{'\textbf{a(iv)}'},...
    'FontWeight','bold',...
    'FontSize',7, 'interpreter','latex',...
    'FitBoxToText','off',...
    'EdgeColor','none');
subplot('Position',[0.1 0.29 0.87 0.10])
plot(timeIndex, smoothdata(testM1_truth(2,dataIndex),'gaussian',51), 'Color',modelColor(1,:), 'LineWidth',1);
hold on
plot(timeIndex, RL.pOutputTest(2,dataIndex), 'Color',modelColor(2,:), 'LineWidth',1)
plot(timeIndex, SL.pOutputTest(2,dataIndex), 'Color',modelColor(3,:), 'LineWidth',1)
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',6.5)
set(gca, 'XColor', 'k', 'LineWidth',1)
xlabel('Time (s)','FontSize',7,'units','normalized','interpreter','latex');
ylabel('${\rm Firing ~Probability}$','FontSize',7,'units','normalized','interpreter','latex', 'Rotation',90);
xlim(timeRange)

%% Movements
%% Real Movements
annotation('textbox',...
    [0.01 0.2 0.04 0.027],...
    'String',{'\textbf{b}'},...
    'FontWeight','bold',...
    'FontSize',7, 'interpreter','latex',...
    'FitBoxToText','off',...
    'EdgeColor','none', 'HorizontalAlignment','center');
subplot('Position',[0.1 0.18 0.87 0.03])
plotActions(timeIndex, testActions(dataIndex), actionsColor, timeRange)
leg = legend({'Rest', 'Press Low', 'Press High'}, ...
    'interpreter','latex','Orientation','horizontal', ...
    'Position',[0.65,0.22,0.34,0.015],'Box','off','FontSize',7);
leg.ItemTokenSize(1) = 15;
text(0.8, 1.05, 'Recordings',...
    'interpreter','latex','FontSize',7,...
    'EdgeColor','none', 'Color', modelColor(1,:), ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment','right');
%% RLPP movements
subplot('Position',[0.1 0.14 0.87 0.03])
motor_perform = RL.motor_perform_test;
motor_perform(testActions==0) = 0;
plotActions(timeIndex, motor_perform(dataIndex), actionsColor, timeRange)
text(0.8, 1, 'RLPP',...
    'interpreter','latex','FontSize',7,...
    'EdgeColor','none', 'Color', modelColor(2,:), ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment','right');
%% SLPP movements
subplot('Position',[0.1 0.10 0.87 0.03])
motor_perform = SL.motor_perform_test;
motor_perform(testActions==0) = 0;
plotActions(timeIndex, motor_perform(dataIndex), actionsColor, timeRange)
text(0.8, 1, 'SLPP',...
    'interpreter','latex','FontSize',7,...
    'EdgeColor','none', 'Color', modelColor(3,:), ...
    'VerticalAlignment', 'middle', 'HorizontalAlignment','right');
%% X-axis
subplot('Position',[0.1 0.08 0.87 0.01])
plot(timeIndex, zeros(1,length(timeIndex)), 'k', 'LineWidth',0.2);
set(gca, 'YTickLabel', []); 
set(gca, 'YColor', 'none')
set(gca, 'XColor', 'k', 'LineWidth',1)
ylim([0,1])
box off
set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',6.5)
xlabel('Time (s)','FontSize',7,'units','normalized','interpreter','latex', 'Color','k');
line([0 0], [0 0.7], 'color', 'k')
ylabel('${\rm Movements}$','FontSize',7,'units','normalized','interpreter','latex', ...
    'Rotation',90, 'Color','k', 'Position',[-0.0347 6.5 0]);

%% Calculate successRate
if ~exist("trained_results/successRate.mat", "file")
    DataNameList = {'rat01','rat02','rat03','rat04','rat05','rat06'};

    testSucRateRL = zeros(6, 5);
    testSucRateSL = zeros(6, 5);
    testTrialSucRateRL = zeros(6, 5);
    testTrialSucRateSL = zeros(6, 5);
    for dataIdx = 1:6
        for i=1:5
            % RL results
            load(['results/', DataNameList{i}, '_RL_', num2str(i), '.mat'])
            testSucRateRL(dataIdx,i) = testSucRate;
            [~, ~, spkOutPredictTest] = applynets_priori(inputUnitTest,...
                L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
                opt.prioriM,opt.prioriN);
            [success,~,~] = emulator_real(spkOutPredictTest,testActions,opt.M1index,60,['decodingModel_', DataNameList{i}(4:5)]);
            testTrialSucRateRL(dataIdx,i) = calculateTrialSucRate(success, testActions);

            % SL results
            load(['results/', DataNameList{i}, '_Sup_', num2str(i), '.mat'])
            testSucRateSL(dataIdx,i) = testSucRate;
            [pOutputTest, ~, spkOutPredictTest] = applynets(inputUnitTest,...
                L1Weight,L2Weight,TestSamples);
            [success,~,~] = emulator_real(spkOutPredictTest,testActions,opt.M1index,60,['decodingModel_', DataNameList{i}(4:5)]);
            testTrialSucRateSL(dataIdx,i) = calculateTrialSucRate(success, testActions);
        end
    end

    testSucRate_mean = [mean(testSucRateRL, 2), mean(testSucRateSL, 2)];
    testSucRate_pos = [max(testSucRateRL,[],2)-mean(testSucRateRL, 2), max(testSucRateSL,[],2)-mean(testSucRateSL, 2)];
    testSucRate_neg = -[min(testSucRateRL,[],2)-mean(testSucRateRL, 2), min(testSucRateSL,[],2)-mean(testSucRateSL, 2)];
    testTrialSucRate_mean = [mean(testTrialSucRateRL, 2), mean(testTrialSucRateSL, 2)];
    testTrialSucRate_pos = [max(testTrialSucRateRL,[],2)-mean(testTrialSucRateRL, 2), max(testTrialSucRateSL,[],2)-mean(testTrialSucRateSL, 2)];
    testTrialSucRate_neg = -[min(testTrialSucRateRL,[],2)-mean(testTrialSucRateRL, 2), min(testTrialSucRateSL,[],2)-mean(testTrialSucRateSL, 2)];
    save trained_results/successRate.mat testSucRateRL         testSucRateSL ...
                                         testTrialSucRateRL    testTrialSucRateSL ...
                                         testSucRate_mean      testSucRate_pos     testSucRate_neg ...
                                         testTrialSucRate_mean testTrialSucRate_pos testTrialSucRate_neg
else
    load("trained_results/successRate.mat")
end

%% define figure
f = figure();
f.Units = 'centimeters';
f.Position = [1 1 18 4.5]; % [left bottom width height]
f.Color = 'w';

subplot(121)
b = bar(testSucRate_mean, "FaceColor", 'flat', 'EdgeColor','none');
b(1).CData = repmat(modelColor(2,:),6,1);
b(2).CData = repmat(modelColor(3,:),6,1);
[RLcoordinates, SLcoordinates] = b.XEndPoints;
hold on
for fold = 1:5
  plot(RLcoordinates+0.01*(fold-3), testSucRateRL(:,fold), 'k.', 'MarkerSize',3)
  plot(SLcoordinates+0.01*(fold-3), testSucRateSL(:,fold), 'k.', 'MarkerSize',3)
end

hold off
xticklabels({'${\rm Rat ~01}$', '${\rm Rat ~02}$', '${\rm Rat ~03}$', '${\rm Rat ~04}$', '${\rm Rat ~05}$', '${\rm Rat ~06}$'})
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
ylabel('Time-bin Success Rate','FontSize',6.5,'units','normalized','interpreter','latex', 'Rotation',90);
box off
for i=1:6
    [~,temp] = signrank(testSucRateRL(i,:), testSucRateSL(i,:), 'Tail','right');
    if ~temp
        continue;
    end
    hold on
    plot([RLcoordinates(i)-0.04 SLcoordinates(i)+0.04], 0.03+max(testSucRate_mean(i,:)+testSucRate_pos(i,:))*[1 1], 'k', 'LineWidth',2)
    plot(i, 0.07+max(testSucRate_mean(i,:)+testSucRate_pos(i,:)), 'k*')
end
hold off
annotation('textbox',[0.08 0.9 0.02 0.08], ...
    'String','\textbf{c}', 'interpreter','latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');

subplot(122)
b = bar(testTrialSucRate_mean, "FaceColor", 'flat', 'EdgeColor','none');
b(1).CData = repmat(modelColor(2,:),6,1);
b(2).CData = repmat(modelColor(3,:),6,1);
[RLcoordinates, SLcoordinates] = b.XEndPoints;
hold on
for fold = 1:5
  plot(RLcoordinates+0.01*(fold-3), testTrialSucRateRL(:,fold), 'k.', 'MarkerSize',3)
  plot(SLcoordinates+0.01*(fold-3), testTrialSucRateSL(:,fold), 'k.', 'MarkerSize',3)
end
hold off
xticklabels({'${\rm Rat ~01}$', '${\rm Rat ~02}$', '${\rm Rat ~03}$', '${\rm Rat ~04}$', '${\rm Rat ~05}$', '${\rm Rat ~06}$'})
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)
ylabel('${\rm Trial ~Success ~Rate}$','FontSize',6.5,'units','normalized','interpreter','latex', 'Rotation',90);
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
legend({'RLPP', 'SLPP'},'interpreter','latex','Orientation','horizontal','Position',[0.76 0.88 0.13 0.05], 'AutoUpdate','off', 'Box','off')
box off
ylim([0,1])
for i=1:6
    [~,temp] = signrank(testTrialSucRateRL(i,:), testTrialSucRateSL(i,:), 'Tail','right');
    if ~temp
        continue;
    end
    hold on
    plot([RLcoordinates(i)-0.04 SLcoordinates(i)+0.04], 0.03+max(testTrialSucRate_mean(i,:)+testTrialSucRate_pos(i,:))*[1 1], 'k', 'LineWidth',1)
    plot(i, 0.07+max(testTrialSucRate_mean(i,:)+testTrialSucRate_pos(i,:)), 'k*', 'MarkerSize',3)
end
hold off
annotation('textbox',[0.52 0.9 0.02 0.08], ...
    'String','\textbf{d}', 'interpreter','latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');
