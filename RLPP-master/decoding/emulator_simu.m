function [success,rate,motor_perform] = emulator_simu(spikes,motor_expect,indexes,his,modelName)
%% rearrange M1 order
[~,I] = sort(indexes);
spikes = spikes(I,:);
%% feedforward to get behavior
motor_perform = eval([modelName, '(spikes, his)']);
%% get reward
success = double(motor_perform==motor_expect);
success(motor_expect==0) = nan;
rate = sum(success(~isnan(success)))/length(success(~isnan(success)));
end
