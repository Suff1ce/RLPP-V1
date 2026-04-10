function [batchInput,batchM1_truth,batchActions,opt] = DataLoader(inputEnsemble,M1_truth,Actions,Trials,opt)
if strcmp(opt.Mode,'train')
    if (opt.DataLoaderCursor == 1) % shuffle when all trials have been trained once
        opt.trainTrials = opt.trainTrials(randperm(opt.NumberOfTrainTrials));
    end
    start = opt.DataLoaderCursor;
    stop = min(opt.NumberOfTrainTrials, opt.DataLoaderCursor+opt.batchSize-1);
    opt.DataLoaderCursor = mod(stop,opt.NumberOfTrainTrials)+1; % move the cursor forward
    trialIndexes = opt.trainTrials(start:stop);
elseif strcmp(opt.Mode,'test')
    trialIndexes = opt.testTrials;
elseif strcmp(opt.Mode,'all')
    trialIndexes = 1:opt.NumberOfTrainTrials;
end
timeIndexes = cell2mat(arrayfun(@(x) find(Trials==x), trialIndexes, 'UniformOutput', false));
timeIndexes = unique(timeIndexes + (-opt.discountLength:0)')';
batchInput    = inputEnsemble(:,timeIndexes);
batchM1_truth = M1_truth(:,timeIndexes);
batchActions  = Actions(timeIndexes);
end
