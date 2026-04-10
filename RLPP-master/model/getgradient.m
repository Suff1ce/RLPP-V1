%% Get gradient
function [WeightDelta1,WeightDelta2] = getgradient(...
    reward,pOutput,spkOutPredict,hiddenUnit,inputUnit,...
    weightFromHiddenToOutput,NumOfSamples)
delta = reward.*(spkOutPredict-pOutput);
WeightDelta1 = delta*hiddenUnit'/NumOfSamples;
WeightDelta2 = (hiddenUnit.*(1-hiddenUnit).*(weightFromHiddenToOutput'*delta))...
    *inputUnit'/NumOfSamples;
WeightDelta2 = WeightDelta2(1:end-1,:);
end
