%% show modulation results
clear; clc;
addpath draw_figures/ utils/ model/ decoding/

%% load data
% Use Rat01 as an example. The modulation of other neurons can be plotted in 
% the similar way
load("trained_results/rat01_RL_results.mat", "folds")
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
[~, ~, spkOutPredictTest] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
% RL.pOutputTest = pOutputTest;
RL.spkOutPredictTest = spkOutPredictTest;

%% Get SL outputs
load("trained_results/rat01_Sup_results.mat", "L1Weight", "L2Weight")
[~, ~, spkOutPredictTest] = applynets(inputUnitTest,...
    L1Weight,L2Weight,TestSamples);
% SL.pOutputTest = pOutputTest;
SL.spkOutPredictTest = spkOutPredictTest;

%% define figure
f = figure();
f.Units = 'centimeters';
f.Position = [1 1 18 17]; % [left bottom width height]
f.Color = 'w';

%% define color
modelColor = [250 0 0; 54 56 131; 103 146 70] / 255;
actionsColor = [178 178 178; 255 165 0; 128 0 128] / 255;

%% Rat 01 Neuron 1 & 2
getTestRasters
% Neuron 1 Spikes
neuronIdx = 1;
subplot('Position',[0.07 0.85 0.12 0.05])
plotTestRasters(lR, lRp, lRsp, neuronIdx, modelColor);
subplot('Position',[0.20 0.85 0.12 0.05])
plotTestRasters(hR, hRp, hRsp, neuronIdx, modelColor);
[ymax, ymin, h] = get_yl(plot_lR,plot_hR,plot_lRp,plot_hRp,plot_lRsp,plot_hRsp,neuronIdx);
subplot('Position',[0.07 0.74 0.12 0.1])
[l1, l2, l3] = plotModulation(plot_lR(neuronIdx,:),plot_lRp(neuronIdx,:),plot_lRsp(neuronIdx,:),modelColor,actionsColor([1 2],:),ymin,ymax,h);
% add ticklabels
for i=ymin:0.2:ymax % linspace(ymin,ymax,3)
  text(-0.54,i,num2str(i), 'HorizontalAlignment','right', 'Interpreter','latex', 'FontSize',6.5)
end
text(-0.82, (ymax+ymin)/2-0.07, 'firing probability', 'Rotation',90, 'VerticalAlignment','middle', 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',6.5)
text(-0.25,-0.05,{'Rest'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color',[0.5 0.5 0.5])
text(0.25,-0.05,{'Press'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color',actionsColor(2,:))
text(0,-0.14,{'(Low Trials)'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color','k')
% add scale bar
hold on
line([0 0.2], [0.35 0.35], 'LineWidth', 1.5, 'Color', 'k')
text(0.07, 0.39, '200 ms', 'Interpreter', 'latex', 'FontSize', 6.5)
hold off
ax = subplot('Position',[0.20 0.74 0.12 0.1]);
plotModulation(plot_hR(neuronIdx,:),plot_hRp(neuronIdx,:),plot_hRsp(neuronIdx,:),modelColor,actionsColor([1 3],:),ymin,ymax,h);
leg = legend([l1, l2, l3],{'Recordings', 'RLPP', 'SLPP'}, ...
    'interpreter','latex','Orientation','horizontal', ...
    'Position',[0.04,0.93,0.32,0.03],'Box','off','FontSize',6.5);
leg.ItemTokenSize = [20 5];
text(-0.25,-0.05,{'Rest'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color',[0.5 0.5 0.5])
text(0.25,-0.05,{'Press'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color',actionsColor(3,:))
text(0,-0.14,{'(High Trials)'}, 'HorizontalAlignment','center', 'Interpreter','latex', 'FontSize',7, 'Color','k')
annotation('textbox',[0.105,0.91 0.18 0.02], ...
    'String','\textbf{Rat 01 Neuron 1}', ...
    'FontSize',7,'EdgeColor','none', ...
    'Interpreter', 'latex', 'HorizontalAlignment','center');
annotation('textbox',[0.04,0.91 0.04 0.027], ...
    'String',{'\textbf{a(i)}'}, 'Interpreter','latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');
% Neuron 2 Spikes
neuronIdx = 2;
subplot('Position',[0.385 0.85 0.12 0.05])
plotTestRasters(lR, lRp, lRsp, neuronIdx, modelColor);
subplot('Position',[0.515 0.85 0.12 0.05])
plotTestRasters(hR, hRp, hRsp, neuronIdx, modelColor);
[ymax, ymin, h] = get_yl(plot_lR,plot_hR,plot_lRp,plot_hRp,plot_lRsp,plot_hRsp,neuronIdx);
subplot('Position',[0.385 0.74 0.12 0.1])
plotModulation(plot_lR(neuronIdx,:),plot_lRp(neuronIdx,:),plot_lRsp(neuronIdx,:),modelColor,actionsColor([1 2],:),ymin,ymax,h);
for i=linspace(ymin,ymax,3)
  text(-0.54,i,num2str(i), 'HorizontalAlignment','right', 'Interpreter','latex', 'FontSize',6.5)
end
subplot('Position',[0.515 0.74 0.12 0.1])
plotModulation(plot_hR(neuronIdx,:),plot_hRp(neuronIdx,:),plot_hRsp(neuronIdx,:),modelColor,actionsColor([1 3],:),ymin,ymax,h);
annotation('textbox',[0.42,0.91 0.18 0.02], ...
    'String','\textbf{Rat 01 Neuron 2}', ...
    'FontSize',7,'EdgeColor','none', ...
    'Interpreter', 'latex', 'HorizontalAlignment','center');
annotation('textbox',[0.355,0.91 0.04 0.027], ...
    'String',{'\textbf{a(ii)}'}, 'Interpreter', 'latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');

%% Calculate Modulation
if ~exist("trained_results/modDep.mat", "file")
    getModulation
else
    load trained_results/modDep.mat
end

%% compare Modulation
f = figure(2);
f.Units = 'centimeters';
f.Position = [1 1 19 9]; % [left bottom width height]
f.Color = 'w';

for j=1:4
    maxMod = max(max(abs(modDep(j:4:12,:))));
    maxMod = ceil(maxMod*20)/20;
    for i=[j+8 j j+4]
        subplot(2,2,j)
        histogram(modDep(i,:),-maxMod:0.05:maxMod, 'FaceColor',modelColor(ceil(i/4),:), ...
            'EdgeAlpha',0, 'FaceAlpha',0.3)
        
        hold on
        set(gca, 'TickLabelInterpreter', 'latex', 'FontSize',8)
    end
end

subplot(2,2,1);
box off
title('\textbf{Low Press - Low Rest}','FontSize',7,'interpreter','latex')

subplot(2,2,2)
box off
title('\textbf{High Press - High Rest}','FontSize',7,'interpreter','latex')

subplot(2,2,3)
box off
title('\textbf{High Press - Low Press}','FontSize',7,'interpreter','latex')
xlabel({'Difference in average firing probability (spikes/10 ms)'},'FontSize',7,'interpreter','latex')
ylabel('Number of Neurons','FontSize',7,'interpreter','latex')

subplot(2,2,4)
box off
title('\textbf{High Rest - Low Rest}','FontSize',7,'interpreter','latex')
xlim([-0.4 0.4])

for j=1:4
    for i=[j+8 j j+4]
        subplot(2,2,j)
        maxGau = max(abs(modDep(i,:)));
        maxGau = ceil(maxGau*20)/20;
        gau_estimate = normpdf(-maxGau:0.01:maxGau,mean(modDep(i,:)), std(modDep(i,:))) ...
          *length(modDep(i,:))*0.05;
        if ceil(i/4) == 1
            plot(-maxGau:0.01:maxGau,gau_estimate,"Color",modelColor(ceil(i/4),:), 'LineWidth',2, 'LineStyle', '--')
        elseif ceil(i/4) == 2
            plot(-maxGau:0.01:maxGau,gau_estimate,"Color",modelColor(ceil(i/4),:), 'LineWidth',2)
        elseif ceil(i/4) == 3
            plot(-maxGau:0.01:maxGau,gau_estimate,"Color",modelColor(ceil(i/4),:), 'LineWidth',2, 'LineStyle', '-.')
        end
    end
end
hold off

annotation('textbox',[0.08,0.98 0.04 0.027], ...
    'String',{'\textbf{d(i)}'}, 'Interpreter', 'latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');
annotation('textbox',[0.52,0.98 0.04 0.027], ...
    'String',{'\textbf{d(ii)}'}, 'Interpreter', 'latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');
annotation('textbox',[0.08,0.50 0.04 0.027], ...
    'String',{'\textbf{d(iii)}'}, 'Interpreter', 'latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');
annotation('textbox',[0.52,0.50 0.04 0.027], ...
    'String',{'\textbf{d(iv)}'}, 'Interpreter', 'latex', ...
    'FontWeight','bold','FontSize',7,'EdgeColor','none');

for j = 1:4
    subplot(2,2,j)
    xlim([-0.7 0.7])
end

ax = subplot(2,2,2);
leg = legend([ax.Children(5) ax.Children(4) ax.Children(6)], ...
  'Recordings', 'RLPP', 'SLPP', 'interpreter', 'latex', 'Box','off', ...
  'AutoUpdate', 'off', 'Location','northwest', 'fontsize', 7);
leg.ItemTokenSize = [15 8];
leg.Position = [0.39 0.74 0.18 0.14];
a = annotation('line', [0.38 0.42], [0.853 0.853]);
a.LineStyle = '--';
a.Color = modelColor(1,:);
a.LineWidth = 2;

a = annotation('line', [0.38 0.42], [0.81 0.81]);
a.LineStyle = '-';
a.Color = modelColor(2,:);
a.LineWidth = 2;

a = annotation('line', [0.38 0.42], [0.767 0.767]);
a.LineStyle = '-.';
a.Color = modelColor(3,:);
a.LineWidth = 2;
