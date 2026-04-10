function [success,rate,motor_perform,ensemble] = emulator_real(spikes,motor_expect,indexes,his,modelName)
%% rearrange M1 order
[~,I] = sort(indexes);
spikes = spikes(I,:);
%% get M1 spike ensemble
M1num = size(spikes,1);
ensemble = zeros((his+1)*M1num,length(spikes));
for i=0:his
    ensemble(i*M1num+1:(i+1)*M1num,:) = [zeros(M1num,i) spikes(:,1:(end-i))];
end
%% feedforward to get behavior
y = eval([modelName, '(ensemble)']);
motor_perform = vec2ind(y);
%% get reward
success = double(motor_perform==motor_expect);
success(motor_expect==0) = nan;
rate = sum(success(~isnan(success)))/length(success(~isnan(success)));
end
