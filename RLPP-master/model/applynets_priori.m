%% ANN to predict M1 spikes
function [pOutput, hiddenUnit, chooseAction] = applynets_priori( ...
    inputUnit,weightFromInputToHidden,weightFromHiddenToOutput,NumOfSamples,episode,M,N)
hiddenUnit = [mySigmoid(weightFromInputToHidden*inputUnit);ones(1,NumOfSamples)];
outputUnit = weightFromHiddenToOutput*hiddenUnit;
pOutput = mySigmoid(outputUnit);
temp = episode*(std(pOutput,[],2));
temp(temp<1e-3) = 1e-3;
pOutput = (temp.*pOutput+M)./(temp+N);
chooseAction = rand(size(outputUnit)) <= pOutput;
end
