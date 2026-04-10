function [inputEnsemble,M1_truth,Actions,Trials,opt] = PreProcess(mPFC,M1,segment,trialNo,decay_parameter,opt)
%% get spike time train for each input channel
mPFC_Time = cell(opt.mPFCchannels,1);
for index=1:opt.mPFCchannels
mPFC_Time{index,1} = find(mPFC(index,:));
end

%% get start position of dataset
% find the time of n+1 spike (n is the number of relevant spikes) of each input channel, then use the max as start point
Start = max(cellfun(@(x) x(opt.RelevantSpikes)+1, mPFC_Time));

%% get output data
M1_truth = M1(opt.M1index,Start:end);
Actions  = segment(Start:end);
Trials  = trialNo(Start:end);

%% get each input channel's ensemble of three sets
% get ensembles
Input_hist = History(mPFC_Time,size(M1,2),opt);
% initial each set input
spikeEnsemble = cell(opt.mPFCchannels,1);
% get input for each channel & each set
for index = 1:opt.mPFCchannels
spikeEnsemble{index,1} = Input_hist{index,1}(:,Start:end);
end
inputEnsemble = cell2mat( ...
    cellfun(@(x) exp(-x/decay_parameter), spikeEnsemble, 'UniformOutput', false) ...
); % delete cell2mat if need cell format variable

%% get train / test trial indexes
Trials(Trials<=1) = nan; % sometimes the first few trials do not have enough history for prediction
AllTrialIndexes = unique(Trials(~isnan(Trials)));
opt.NumberOfAllTrials = length(AllTrialIndexes);
AllTrialIndexes = AllTrialIndexes(randperm(opt.NumberOfAllTrials));

foldTrialNum = floor(opt.NumberOfAllTrials/opt.foldNum);
foldStart = 1:foldTrialNum:(opt.foldNum*foldTrialNum);
foldStop  = [foldStart(2:end)-1, opt.foldNum*foldTrialNum];
opt.folds = cell(1,opt.foldNum);
opt.FoldTrialNumber = zeros(1,opt.foldNum);
for i=1:opt.foldNum
    opt.folds{i} = AllTrialIndexes(foldStart(i):foldStop(i));
    opt.FoldTrialNumber(i) = length(opt.folds{i});
end
end


%% Function History
% get ensemble of spike time
%    Input: xt - spike time train {channelNumber, 1}(1, time of m th spike)
%           Ny - total length of time bins of output signal
%           opt - opt.mPFCchannels: input channel number
%                 opt.RelevantSpikes: number of history length (number of past spikes suppossed to be relevant to current spike)
%    Output: x1 - input history ensemble {channelNumber, 1}(opt.RelevantSpikes+1-i th nearest spike from kth time bin)
%    Algorithm: Given m th spike time t_m and t_m+1 the spike time m+1,
%               from timebins t_m to (t_m+1)-1, the past relevant spikes are
%               same.
%               Thus, first take past relevant spike time to fill in the
%               ensemble t_m to (t_m+1)-1, then use current time deduct the
%               past spike time, we get the ensemble
%    Notes: The code simply use time bins t_m to t_m+1, because the t_m+1
%           will be rewrite in next iteration. As a result, although the
%           code is not same as the algorithm, but have same results
function x1 = History(xt,Ny,opt)
Ensemble = cellfun(@(x) size(x,2)-opt.RelevantSpikes, xt); % Ensemble is (number of spikes - relevant spike number) for each input channel
x1 = cell(opt.mPFCchannels,1); % create a history ensemble for each input channel
for jndex = 1:opt.mPFCchannels % for each channel
   x1{jndex,1} = ones(opt.RelevantSpikes,Ny)*inf; % Initial the x1
   
   % For time bin 1 to time bin xt{jndex,1}(opt.RelevantSpikes), no
   % enough past spikes are avilible, so do some special process
   for index= 1:opt.RelevantSpikes-1
       x1{jndex,1}(1:index,xt{jndex,1}(index):xt{jndex,1}(index+1)) = ...
           repmat(xt{jndex,1}(1,1:index)',1, xt{jndex,1}(index+1)-xt{jndex,1}(index)+1);
   end
   
   % deal with the ensemble part. all have enough relevant spike history
   for index = 1:Ensemble(jndex) % Traverse all ensemble indexes, same way as above, instead of a slide window
       x1{jndex,1}(:,xt{jndex,1}(index+opt.RelevantSpikes-1):xt{jndex,1}(index+opt.RelevantSpikes)) = ... % start from index=opt.RelevantSpikes
           repmat(xt{jndex,1}(1,index:index+opt.RelevantSpikes-1)',1, xt{jndex,1}(index+opt.RelevantSpikes)-xt{jndex,1}(index+opt.RelevantSpikes-1)+1);
   end
   
   % fill in the rest time bins, since index=Ensemble and there is no more xt{jndex,1}(index+1)
   x1{jndex,1}(:,xt{jndex,1}(end):end) =  repmat(xt{jndex,1}(1,end-opt.RelevantSpikes+1:end)',1,Ny-xt{jndex,1}(end)+1);
   
   x1{jndex,1} = repmat(1:Ny,opt.RelevantSpikes,1)-x1{jndex,1}; % current time index - past spike time index to get duration
end
end
