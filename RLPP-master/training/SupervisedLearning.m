function [data,opt,s] = SupervisedLearning(data,opt,s,inputEnsemble,M1_truth,Actions,Trials)
%Supervised learning
%   
rng(s);
%% history records
rewardHis = zeros(1,opt.maxEpisode);
crossEntropyHis = zeros(1,opt.maxEpisode);
MaxReward = -Inf;
MinError = Inf;

%% network weights
weightFromInputToHidden=2*rand(opt.hiddenUnitNum,data.mPFCnum*opt.RelevantSpikes+1)-1;
weightFromHiddenToOutput=2*rand(data.M1num_pre,opt.hiddenUnitNum+1)-1;

%% Train with supervised learning
if (opt.verbose<=3)
    disp(['----', opt.DataIndex,' Sup Train start----']);
end
opt.Mode = 'train';
for episode=1:opt.maxEpisode
    % get batch data
    [batchInput,batchM1_truth,batchActions,opt] = DataLoader( ...
        inputEnsemble,M1_truth,Actions,Trials,opt);
    NumOfSamples = length(batchInput);
    inputUnit = [batchInput;ones(1,NumOfSamples)];

    % forward to get spikes
    [pOutput, hiddenUnit, spkOutPredict] = applynets(inputUnit,...
        weightFromInputToHidden,weightFromHiddenToOutput,NumOfSamples);
    
    % emulator: get predict motor
    if strcmp(opt.DataIndex, 'Simulations')
        [~,sucRate,~] = emulator_simu( ...
            spkOutPredict,batchActions,opt.M1index,data.his,data.modelName);
    else
        [~,sucRate,~] = emulator_real( ...
            spkOutPredict,batchActions,opt.M1index,data.his,data.modelName);
    end
    rewardHis(episode) = sucRate;
    
    % store history and print log
    % cross-entropy between M1 predict rate and true spike train
    crossEntropyHis(episode) = CrossEntropyError(...
      [1-batchM1_truth;batchM1_truth],[1-pOutput;pOutput],NumOfSamples);
    if crossEntropyHis(episode)<=MinError
      MinError = crossEntropyHis(episode);
      L2Weight = weightFromHiddenToOutput;
      L1Weight = weightFromInputToHidden;
      MinErrorEpisode = episode;
    end
    % record weights for best reward
    if rewardHis(episode)>=MaxReward
      MaxReward = rewardHis(episode);
      L2WeightBestReward = weightFromHiddenToOutput;
      L1WeightBestReward = weightFromInputToHidden;
      MaxRewardEpisode = episode;
    end
    % print log
    if (opt.verbose<=0)
        disp(strcat(...
          num2str(episode),'/',num2str(opt.maxEpisode),...
          '...Error',num2str(crossEntropyHis(episode)),...
          '...Reward',num2str(rewardHis(episode))...
        ));
    end
    
    % get gradient
    [WeightDelta1,WeightDelta2] = getgradient_sup(...
        pOutput,batchM1_truth,hiddenUnit,inputUnit,...
        weightFromHiddenToOutput,NumOfSamples...
    );

    % gradient descent
    if strcmp(opt.DataIndex, 'Simulations')
        lr = 0.3;
    else
        lr = 1.1*(1-episode/opt.maxEpisode)+0.1;
    end
    weightFromHiddenToOutput = weightFromHiddenToOutput + ...
        lr*(WeightDelta1-0*weightFromHiddenToOutput);
    weightFromInputToHidden = weightFromInputToHidden + ...
        lr*(WeightDelta2-0*weightFromInputToHidden);
end

%% test model
opt.Mode = 'test';
[testInput,testM1_truth,testActions,opt] = DataLoader( ...
    inputEnsemble,M1_truth,Actions,Trials,opt);
TestSamples = length(testInput);
inputUnitTest = [testInput;ones(1,TestSamples)];
[pOutputTest, hiddenUnitTest, spkOutPredictTest] = applynets(inputUnitTest,...
    L1Weight,L2Weight,TestSamples);
if strcmp(opt.DataIndex, 'Simulations')
    [~,testSucRate,motor_perform_test] = emulator_simu(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
else
    [~,testSucRate,motor_perform_test] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
end

if (opt.verbose<=3)
    disp(strcat(...
      '====',opt.DataIndex,' Sup Test finish...Reward: ',num2str(testSucRate),'===='...
    ));
end

%% save results
clearvars("data","inputEnsemble","M1_truth","Actions","Trials");
if strcmp(opt.DataIndex, 'Simulations')
    targetFile = ['results/',opt.DataIndex,'_Sup_',num2str(opt.testFold),'.mat'];
    save(targetFile)
else
    randstr = ['a':'z' '0':'9'];
    randId = [datestr(datetime,'mmmdd_HHMM'),'_',randstr(randi(length(randstr),1,4))];
    targetFile = ['results/',opt.DataIndex,'/Sup_',num2str(opt.testFold),'_',randId,'.mat'];
    save(targetFile)
end

end
