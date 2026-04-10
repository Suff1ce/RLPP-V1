%% show time domain of rat02 under mannual designed decoder
clear; clc;
addpath draw_figures/ utils/ model/ decoding/
load("trained_results/rat02_RL_results.mat", "folds")

f = figure(1);
f.Units = 'centimeters';
f.Position = [4 4 18 12]; % [left bottom width height]
f.Color = 'w';

modelColor = [250 0 0; 54 56 131; 103 146 70] / 255;
actionsColor = [178 178 178; 255 165 0; 128 0 128] / 255;

%% load data
DataName = "data/rat02.mat";
data = data_setup(DataName);
data.M1num_pre = 4;
data.modelName = 'decodingModel_manual';
dta.his = 50;
data.DataIndex = [data.DataIndex, 'Manual'];

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

%% Get RL outputs
load("trained_results/rat02manual_RL_results.mat", "L1WeightBestReward", "L2WeightBestReward", "MaxRewardEpisode")
[pOutputTest, ~, spkOutPredictTest] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
[success,testSucRate,motor_perform_test] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);

%% show results
dataIndex = 1001:3500;
timeIndex = dataIndex/100;
timeRange = [10.01 35];

figure(1)
[~,I] = sort(opt.M1index);
pOutputTest = pOutputTest(I,:);
spikePredict = spkOutPredictTest(I,:);
subplot('Position',[0.13 0.93 0.7 0.02])
plotActions(timeIndex, testActions(dataIndex), actionsColor, timeRange)
t = title("\textbf{Rat 02 with manual decoder}", "FontSize",7, "Interpreter","latex");
t.Position(2) = t.Position(2) + 0.2;
text(timeRange(1)+0.5, 1, 'Recorded', Interpreter='latex',VerticalAlignment='middle', HorizontalAlignment='right', ...
  FontSize=6.5)
text(timeRange(1)-2, 1.5, '\textbf{a}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none')
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)

subplot('Position',[0.13 0.90 0.7 0.02])
motor_perform = motor_perform_test;
motor_perform(testActions==0) = 0;
plotActions(timeIndex, motor_perform(dataIndex), actionsColor, timeRange)
text(timeRange(1)+0.5, 1, 'Decoded', Interpreter='latex',VerticalAlignment='middle', HorizontalAlignment='right', ...
  FontSize=6.5)
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)

subplot('Position',[0.13 0.66 0.7 0.18])
plot(timeIndex, gaussianSmooth(pOutputTest(1,dataIndex), 5), "Color",[0.8500 0.3250 0.0980], 'LineWidth',1);
hold on
plot(timeIndex, gaussianSmooth(pOutputTest(3,dataIndex), 5), "Color", "blue", 'LineWidth',1); 
legend(["Artifical Neuron 1", "Artifical Neuron 3"],'interpreter','latex','Orientation','horizontal','AutoUpdate','off', "Box","off", ...
  'Position',[0.57 0.85 0.17 0.03], 'FontSize',6.5)
plotSpikes(timeIndex, spikePredict(1,dataIndex), 1.05, [0.8500 0.3250 0.0980])
plotSpikes(timeIndex, spikePredict(3,dataIndex), 0.92, "blue")
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
text(timeRange(1)-2, 1.1, '\textbf{b}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none')

subplot('Position',[0.13 0.41 0.7 0.18])
plot(timeIndex, gaussianSmooth(pOutputTest(2,dataIndex), 5), "Color","magenta", 'LineWidth',1);
hold on
plot(timeIndex, gaussianSmooth(pOutputTest(4,dataIndex), 5), "Color",[0.4660 0.6740 0.1880], 'LineWidth',1); 
legend(["Artifical Neuron 2", "Artifical Neuron 4"],'interpreter','latex','Orientation','horizontal','AutoUpdate','off', "Box","off", ...
  'Position',[0.57 0.6 0.17 0.03], 'FontSize',6.5)
plotSpikes(timeIndex, spikePredict(2,dataIndex), 1.03, "magenta")
plotSpikes(timeIndex, spikePredict(4,dataIndex), 0.9, [0.4660 0.6740 0.1880])
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
text(timeRange(1)-2, 1.1, '\textbf{c}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none')

subplot('Position',[0.13 0.16 0.7 0.18])
plot(timeIndex, gaussianSmooth(testM1_truth(1,dataIndex), 10), "Color",'k', 'LineWidth',1);
hold on
plot(timeIndex, gaussianSmooth(pOutputTest(1,dataIndex), 5), "Color",[0.8500 0.3250 0.0980], 'LineWidth',1);
legend(["M1 Neuron 1 recording", "Artificial Neuron 1"],'interpreter','latex','Orientation','horizontal','AutoUpdate','off', "Box","off", ...
  'Position',[0.56 0.35 0.17 0.03], 'FontSize',6.5)
plotSpikes(timeIndex, testM1_truth(1,dataIndex), 1.05, 'k')
plotSpikes(timeIndex, spikePredict(1,dataIndex), 0.92, [0.8500 0.3250 0.0980])
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex','FontSize',6.5)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
xlabel("Time (sec)", "FontSize",7, "Interpreter","latex")
ylabel("Firing probability", "FontSize",7, "Interpreter","latex")
text(timeRange(1)-2, 1.1, '\textbf{d}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none')

%% Similar modulation under different decoders
f = figure();
f.Units = 'centimeters';
f.Position = [17 4 20 17]; % [left bottom width height]
f.Color = 'w';

getPlotModulation = @(x) [
  smoothdata(squeeze(sum(x(:,:,1:51), 2))/size(x,2), 2, 'gaussian', 25) ...
  smoothdata(squeeze(sum(x(:,:,52:end), 2))/size(x,2), 2, 'gaussian', 25)
];

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

[lR,hR] = getRaster(testActions,testM1_truth);
plot_lR = getPlotModulation(lR);
plot_hR = getPlotModulation(hR);

load("trained_results/rat02_RL_results.mat", "L1WeightBestReward", "L2WeightBestReward", "MaxRewardEpisode")
[~, ~, selfDecoderPrediction] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
[lRp1,hRp1] = getRaster(testActions,selfDecoderPrediction);
plot_lRp1 = getPlotModulation(lRp1);
plot_hRp1 = getPlotModulation(hRp1);

load("trained_results/rat02decoder01_RL_results.mat", "L1WeightBestReward", "L2WeightBestReward", "MaxRewardEpisode")
[~, ~, anotherDecoderPrediction] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
[lRp2,hRp2] = getRaster(testActions,anotherDecoderPrediction);
plot_lRp2 = getPlotModulation(lRp2);
plot_hRp2 = getPlotModulation(hRp2);

neuronIdx = 2;
subplot('Position',[0.07 0.85 0.12 0.05])
plotTestRasters(lR, lRp1, lRp2, neuronIdx, ['k'; 'b'; 'c'])
subplot('Position',[0.20 0.85 0.12 0.05])
plotTestRasters(hR, hRp1, hRp2, neuronIdx, ['k'; 'b'; 'c'])

[ymax, ymin, h] = get_yl(plot_lR,plot_hR,plot_lRp1,plot_hRp1,plot_lRp2,plot_hRp2,neuronIdx);
subplot('Position',[0.07 0.74 0.12 0.1])
[l1, l2, l3] = plotModulation(plot_lR(neuronIdx,:),plot_lRp1(neuronIdx,:), ...
  plot_lRp2(neuronIdx,:),['k'; 'b'; 'c'],actionsColor([1 2],:),ymin,ymax,h);
for i=ymin:0.3:ymax % linspace(ymin,ymax,3)
  text(-0.54,i,num2str(i), 'HorizontalAlignment','right', 'Interpreter','latex', 'FontSize',10)
end
text(-0.82, (ymax+ymin)/2-0.07, 'firing probability', 'Rotation',90, ...
  'VerticalAlignment','middle', 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10)
text(-0.25,ymin-(ymax-ymin)/9,{'Rest'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color',[0.5 0.5 0.5])
text(0.25,ymin-(ymax-ymin)/9,{'Press'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color',actionsColor(2,:))
text(0,ymin-(ymax-ymin)/3,{'(Low Trials)'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color','k')
% add scale bar
hold on
line([-0.2 0], [0.5 0.5], 'LineWidth', 1.5, 'Color', 'k')
text(-0.4, 0.6, '200 ms', 'Interpreter', 'latex', 'FontSize', 10)
hold off
leg = legend([l1, l2, l3],{['Rat 02 Neuron ' num2str(neuronIdx), ' Recordings'], ...
    'RLPP with decoder 02', ...
    'RLPP with decoder 01'}, ...
    'interpreter','latex','Orientation','vertical', ...
    'Position',[0.04,0.93,0.32,0.03],'Box','off','FontSize',10);
leg.ItemTokenSize = [20 5];

subplot('Position',[0.20 0.74 0.12 0.1]);
plotModulation(plot_hR(neuronIdx,:),plot_hRp1(neuronIdx,:), ...
  plot_hRp2(neuronIdx,:),['k'; 'b'; 'c'],actionsColor([1 3],:),ymin,ymax,h);
text(-0.25,ymin-(ymax-ymin)/9,{'Rest'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color',[0.5 0.5 0.5])
text(0.25,ymin-(ymax-ymin)/9,{'Press'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color',actionsColor(3,:))
text(0,ymin-(ymax-ymin)/3,{'(High Trials)'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',10, 'Color','k')

%% Behavioral Performance under different decoders
load trained_results/successRate_DifferentDecoders.mat

subplot(3,3,[4 5])
hold on
means = mean(testSucRate, 3);
mins = min(testSucRate, [], 3);
maxs = max(testSucRate, [], 3);

b = bar(means, 'grouped');
[ngroups, nbars] = size(means);
for i = 1:nbars
    errorbar(b(i).XEndPoints, means(:,i), mins(:,i)-means(:,i), means(:,i)-maxs(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

for i = 1:nbars
  for j = 1:ngroups
    b(i).FaceColor = 'flat';
    if i==j
      b(i).CData(j,:) = modelColor(2,:);
    elseif i~=7
      b(i).CData(j,:) = [200, 220, 255]/255;
    else
      b(i).CData(j,:) = [1 1 1];
    end
  end
end

plot([0.55 6.45], [1/3 1/3], 'k--', 'LineWidth',2)
xticks(1:6)
xticklabels({'${\rm Rat ~01}$', '${\rm Rat ~02}$', '${\rm Rat ~03}$', '${\rm Rat ~04}$', '${\rm Rat ~05}$', '${\rm Rat ~06}$'})
set(gca, 'TickLabelInterpreter', 'latex','FontSize',12)
ylabel('Time-bin Success Rate','FontSize',12,'units','normalized','interpreter','latex', 'Rotation',90);
hold off
text(6.47, 0.3333, {'One Step'; 'Chance'; 'Rate'}, 'FontName','Times New Roman', ...
  'FontSize',10,'EdgeColor','none');

subplot(3,3,[7 8])
hold on
means = mean(testTrialSucRate, 3);
mins = min(testTrialSucRate, [], 3);
maxs = max(testTrialSucRate, [], 3);

b = bar(means, 'grouped');

for i = 1:nbars
  for j = 1:ngroups
    b(i).FaceColor = 'flat';
    if i==j
      b(i).CData(j,:) = modelColor(2,:);
    elseif i~=7
      b(i).CData(j,:) = [200, 220, 255]/255;
    else
      b(i).CData(j,:) = [1 1 1];
    end
  end
end

[ngroups, nbars] = size(means);
for i = 1:nbars
    errorbar(b(i).XEndPoints, means(:,i), mins(:,i)-means(:,i), means(:,i)-maxs(:,i), 'k', 'linestyle', 'none', 'LineWidth', 1);
end

xticks(1:6)
xticklabels({'${\rm Rat ~01}$', '${\rm Rat ~02}$', '${\rm Rat ~03}$', '${\rm Rat ~04}$', '${\rm Rat ~05}$', '${\rm Rat ~06}$'})
set(gca, 'TickLabelInterpreter', 'latex','FontSize',12)
ylabel('${\rm Trial ~Success ~Rate}$','FontSize',12,'units','normalized','interpreter','latex', 'Rotation',90);
plot([0.55 6.45], [0.02 0.02], 'k--', 'LineWidth',2)
hold off
text(6.43, 0.15, {'Multi-Step'; 'Chance'; 'Rate'}, 'FontName','Times New Roman', ...
  'FontSize',10,'EdgeColor','none');
