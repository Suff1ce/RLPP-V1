function [M1,mPFC,M1num,mPFCnum,M1channelName,mPFCchannelName,segment,trialNo,actions] = spikeTime2Train(filename)
%Discrete the original time series from Plexon to the spike trains and behavior labels
%   INPUT:
%       filename    - path to the original data
%   OUTPUT:
%       M1          - M1 spike trains. Matrix of TimeLength * M1num
%       mPFC        - mPFC spike trains. Matrix of TimeLength * NumOfNeurons
%       M1num       - Number of M1 neurons
%       mPFCnum     - Number of mPFC neurons
%       M1channelName   - Char matrix of M1num * 5
%       mPFCchannelName - Char matrix of M1num * 5
%       segment     - Label rest(1), high-press(2), and low-press(3) behaviors
%                     in successful trials
%       trialNo     - Label the number index of each segment. The index add
%                     1 per trial (1 rest + 1 press)
%       actions     - Mark all press-release movements by 1.

%% Load file
S = load(filename);
timebins  = 0.01; % 10ms time bins

%% Get number of M1 and mPFC neurons and the spike length
spikeLength = Inf;
M1num       = 0;
mPFCnum     = 0;
for channelNo = 1:32
  for sub = ['a', 'b', 'c', 'd'] % Maximum of 4 units per channel
    channelName = ['WB', num2str(channelNo, '%02d'), sub];
    if (~isfield(S, channelName))
      continue; % no such unit
    end
    if (channelNo <= 16)
      M1num     = M1num + 1;
    else
      mPFCnum   = mPFCnum + 1;
    end
    spikeLength = min(spikeLength, ceil(max(eval(['S.', channelName]))/timebins));
  end
end
% initial the spike trains
M1   = zeros(spikeLength, M1num);
mPFC = zeros(spikeLength, mPFCnum);

%% From spike time to spike train
M1i = 1; mPFCi = 1; % cursor for units
% initial channel names
M1channelName   = 'xxxxx'; 
mPFCchannelName = 'xxxxx';
for channelNo = 1:32
  for sub = ['a', 'b', 'c', 'd']
    channelName = ['WB', num2str(channelNo, '%02d'), sub];
    if (~isfield(S, channelName))
      continue;
    end
    % get spike train
    spikeTrain = accumarray(fix(eval(['S.', channelName])*1/timebins)+1, 1);
    if (channelNo <= 16) % first 16 is M1
      M1(:, M1i) = spikeTrain(1:spikeLength);
      M1channelName(M1i, :) = channelName;
      M1i = M1i+1;
    else % latter 16 is mPFC
      mPFC(:,mPFCi) = spikeTrain(1:spikeLength);
      mPFCchannelName(mPFCi, :) = channelName;
      mPFCi = mPFCi+1;
    end
  end
end

M1 = double(M1>0); % logic type seems not easy for caclating
mPFC = double(mPFC>0);

%% Event time to event train
[rest_time,press_low_time,press_high_time,press_release] = find_behavior_index( ...
    S.EVT01,S.EVT02,S.EVT03,S.EVT04,S.EVT05,S.EVT06,S.EVT07,S.EVT08);
% Assign values to segment
segment = zeros(1, spikeLength);
for i=1:length(rest_time)
  segment( ...
    floor(rest_time(i,1)/timebins):floor(rest_time(i,2)/timebins) ...
  )=1;
end
for i=1:length(press_low_time)
  segment( ...
    floor(press_low_time(i,1)/timebins):floor(press_low_time(i,2)/timebins) ...
  )=2;
end
for i=1:length(press_high_time)
  segment( ...
    floor(press_high_time(i,1)/timebins):floor(press_high_time(i,2)/timebins) ...
  )=3;
end
segment = segment(1:spikeLength); % the event recording may be longer than the spike recording
% Number the trials
trialNo = nan(1, spikeLength);
trials = 0;
for i=1:spikeLength
    if segment(i)>0
        if segment(i) == 1 && segment(i-1) == 0 % rising edge trigger
            trials = trials + 1;
        end
        trialNo(i) = trials;
    end
end
trialNo = trialNo(1:spikeLength);
trialNo(trialNo==0) = nan; % the very first trial may have no start part been recorded
% Assign values to actions
actions = zeros(1, spikeLength);
for i=1:length(press_release)
  actions( ...
    floor(press_release(i,1)/timebins):floor(press_release(i,2)/timebins) ...
  )=1;
end
actions = actions(1:spikeLength);

end
