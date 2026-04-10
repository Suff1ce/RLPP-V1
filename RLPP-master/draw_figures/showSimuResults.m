clear; close all
legendPos = [0.38 0.95 0.53 0.04];
plotIndex = 1201:2400;
timeIndex = plotIndex/100;

%%
load results\Simulations_RL_1.mat

f = figure("Name","RLPP results");
f.Color = 'w';
subplot(3,1,1)
plot(timeIndex, gaussianSmooth(testM1_truth(1, plotIndex), 10), 'r', 'LineWidth',1.5)
hold on 
plot(timeIndex, pOutputTest(1, plotIndex), 'Color', [54 56 131]/255, 'LineWidth',1.5)
plotSpikes(timeIndex, testM1_truth(1, plotIndex), 1.4, 'r')
plotSpikes(timeIndex, spkOutPredictTest(1, plotIndex), 1.2, [54 56 131]/255)
box off
ylabel('M1 1')
legend(["Simulated M1 firing", "RL-generated M1 firing"], 'Position', legendPos, 'Orientation','horizontal')
subplot(3,1,2)
plot(timeIndex, gaussianSmooth(testM1_truth(2, plotIndex), 10), 'r', 'LineWidth',1.5)
hold on 
plot(timeIndex, pOutputTest(2, plotIndex), 'Color', [54 56 131]/255, 'LineWidth',1.5)
plotSpikes(timeIndex, testM1_truth(2, plotIndex), 1.4, 'r')
plotSpikes(timeIndex, spkOutPredictTest(2, plotIndex), 1.2, [54 56 131]/255)
box off
ylabel('M1 2')
subplot(3,1,3)
temp = testActions(plotIndex);
temp(temp~=3) = nan;
temp(temp==3) = 1.5;
plot(timeIndex, temp, 'r', 'LineWidth', 8);
hold on
temp = testActions(plotIndex);
temp(temp~=2) = nan;
temp(temp==2) = 1.5;
plot(timeIndex, temp, 'y', 'LineWidth', 8);
temp = testActions(plotIndex);
temp(temp~=1) = nan;
temp(temp==1) = 1.5;
plot(timeIndex, temp, 'b', 'LineWidth', 8);
text(timeIndex(1)+0.49, 1.9, 'Correct movements')

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=3) = nan;
temp(temp==3) = 0.5;
plot(timeIndex, temp, 'r', 'LineWidth', 8);

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=2) = nan;
temp(temp==2) = 0.5;
plot(timeIndex, temp, 'y', 'LineWidth', 8);

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=1) = nan;
temp(temp==1) = 0.5;
plot(timeIndex, temp, 'b', 'LineWidth', 8);
text(timeIndex(1)+0.49, 0.9, ['Predicted movements from RL outputs (Time-bin success rate: ' num2str(testSucRate, '%.2f'), ')'])

hold off
ylim([0 2])
legend(["Press High", "Press Low", "rest"], 'AutoUpdate','off', "Orientation", 'horizontal', 'Position',[0.43 0.31 0.48 0.04])
box off
xlabel('Time (sec)')
ylabel('Movements')

%%
clearvars -except legendPos plotIndex timeIndex
load results\Simulations_Sup_1.mat

f = figure("Name","Supervised learning results");
f.Color = 'w';
subplot(3,1,1)
plot(timeIndex, gaussianSmooth(testM1_truth(1, plotIndex), 10), 'r', 'LineWidth',1.5)
hold on 
plot(timeIndex, pOutputTest(1, plotIndex), 'Color', [103 146 70]/255, 'LineWidth',1.5)
plotSpikes(timeIndex, testM1_truth(1, plotIndex), 1.4, 'r')
plotSpikes(timeIndex, spkOutPredictTest(1, plotIndex), 1.2, [103 146 70]/255)
box off
ylabel('M1 1')
legend(["Simulated M1 firing", "SL-predicted M1 firing"], 'Position', legendPos, 'Orientation', 'horizontal')
subplot(3,1,2)
plot(timeIndex, gaussianSmooth(testM1_truth(2, plotIndex), 10), 'r', 'LineWidth',1.5)
hold on 
plot(timeIndex, pOutputTest(2, plotIndex), 'Color', [103 146 70]/255, 'LineWidth',1.5)
plotSpikes(timeIndex, testM1_truth(2, plotIndex), 1.4, 'r')
plotSpikes(timeIndex, spkOutPredictTest(2, plotIndex), 1.2, [103 146 70]/255)
box off
ylabel('M1 2')
subplot(3,1,3)
temp = testActions(plotIndex);
temp(temp~=3) = nan;
temp(temp==3) = 1.5;
plot(timeIndex, temp, 'r', 'LineWidth', 8);
hold on
temp = testActions(plotIndex);
temp(temp~=2) = nan;
temp(temp==2) = 1.5;
plot(timeIndex, temp, 'y', 'LineWidth', 8);
temp = testActions(plotIndex);
temp(temp~=1) = nan;
temp(temp==1) = 1.5;
plot(timeIndex, temp, 'b', 'LineWidth', 8);
text(timeIndex(1)+0.49, 1.9, 'Correct movements')

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=3) = nan;
temp(temp==3) = 0.5;
plot(timeIndex, temp, 'r', 'LineWidth', 8);

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=2) = nan;
temp(temp==2) = 0.5;
plot(timeIndex, temp, 'y', 'LineWidth', 8);

temp = motor_perform_test(plotIndex);
temp(testActions(plotIndex)==0) = nan;
temp(temp~=1) = nan;
temp(temp==1) = 0.5;
rest = plot(timeIndex, temp, 'b', 'LineWidth', 8);
text(timeIndex(1)+0.49, 0.9, ['Predicted movements from SL outputs  (Time-bin success rate: ' num2str(testSucRate, '%.2f'), ')'])

hold off
ylim([0 2])
legend(["Press High", "Press Low", "rest"], 'AutoUpdate','off', "Orientation", 'horizontal', 'Position',[0.43 0.31 0.48 0.04])
box off
xlabel('Time (sec)')
ylabel('Movements')