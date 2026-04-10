%% ANN to predict M1 spikes
function [pOutput, hiddenUnit, chooseAction] = applynets(inputUnit,weightFromInputToHidden,weightFromHiddenToOutput,NumOfSamples)
hiddenUnit = [mySigmoid(weightFromInputToHidden*inputUnit);ones(1,NumOfSamples)];
outputUnit = weightFromHiddenToOutput*hiddenUnit;
pOutput = mySigmoid(outputUnit);
chooseAction = rand(size(outputUnit)) <= pOutput;
end
