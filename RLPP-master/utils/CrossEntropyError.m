%% Cross Entropy
function Error = CrossEntropyError(Action,pOutput,NumOfSample)
Error = sum(sum(-Action.*log(pOutput)))/NumOfSample;
end
