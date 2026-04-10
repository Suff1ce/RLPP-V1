clear; close all;
DataIndex = 'Simulations';
rng('default');

%% pre-define neural probabilities
% 10ms per time bin

% one mPFC neuron
rate_input = zeros(1, 2e4);
mPFCnum = 1;

% three M1 neuron, which only the first two are related to mPFC
rate_output = zeros(3, 2e4);
M1num = 3;
M1order = [1, 2, 3];
M1num_pre = 2; % only predict the first two

segment = zeros(1, 2e4); % movement of the "rat"
trialNo = zeros(1, 2e4); % trial No, from 1~200

%%
cursor = 1;
trialType = 1;
for trialIdx = 1:200 % simulate 200 trials

  startTime = randi([40, 50]); % start cue given after ([40 50])+50 time bins
  responseTime = randi([50, 60]); 
  pressTime = randi([80, 100]); % hold the level

  % trialType 1 means low trial, 2 means high trial 
  if trialType == 1
    trialType = 2;
  elseif trialType == 2
    trialType = 1;
  end
  
  % assume the mPFC firing changes 200 ms ahead of the press movement
  % baseline firing: 0.05, low lever: 0.3, high lever: 0.7
  mPFC_part = [0.05*ones(1, startTime+50+responseTime-20), (0.3+0.4*(trialType==2))*ones(1, 20+pressTime)];

  % M1 1: 0.8 for high lever, 0.1 for other time bins
  % M1 2: 0.7 for both lever, 0.05 for other time bins
  % M1 3: 0.5 for all the time
  M1_part = [0.1*ones(1, startTime+50+responseTime), (0.1+0.7*(trialType==2))*ones(1, pressTime);
    0.05*ones(1, startTime+50+responseTime), 0.7*ones(1, pressTime);
    0.5*ones(1, startTime+50+responseTime+pressTime)];
  
  segment_part = [zeros(1, startTime), ones(1, 50), zeros(1, responseTime), (1+trialType)*ones(1, pressTime)];
  
  trialNo_part = [zeros(1, startTime), trialIdx*ones(1, 50), zeros(1, responseTime), trialIdx*ones(1, pressTime)];
  
  rate_input(1, cursor+(1:length(mPFC_part))) = mPFC_part;
  rate_output(:, cursor+(1:length(mPFC_part))) = M1_part;
  segment(:, cursor+(1:length(mPFC_part))) = segment_part;
  trialNo(:, cursor+(1:length(mPFC_part))) = trialNo_part;
  
  cursor = cursor + length(mPFC_part);

end

rate_input = gaussianSmooth(rate_input, 10); % smooth the neural firing rate
mPFC = rate_input>rand(size(rate_input));%% Bernoulli generation
rate_output = gaussianSmooth(rate_output', 10)';  % smooth the neural firing rate
M1 = rate_output>rand(size(rate_output));%% Bernoulli generation

modelName = 'decodingModel_simulation';
his = 16;

%%
plotIndex = 1:1000;
timeIndex = plotIndex/100;
f = figure();
f.Color = 'w';
subplot(4,1,1)
plot(timeIndex, rate_input(plotIndex), 'k', 'LineWidth',1.5)
ylabel('mPFC')
box off
subplot(4,1,2)
plot(timeIndex, rate_output(1, plotIndex), 'k', 'LineWidth',1.5)
ylim([0 0.8])
box off
ylabel('M1 1')
subplot(4,1,3)
plot(timeIndex, rate_output(2, plotIndex), 'k', 'LineWidth',1.5)
ylim([0 0.8])
box off
ylabel('M1 2')
subplot(4,1,4)
plot(timeIndex, segment(plotIndex), 'k', 'LineWidth',1.5)
box off
ylabel('movements')

M1 = M1';
mPFC = mPFC';
save data\simulations.mat mPFC mPFCnum M1 M1num M1order M1num_pre segment trialNo his modelName DataIndex
