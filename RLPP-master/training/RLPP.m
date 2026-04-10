function [data,opt,s] = RLPP(data,opt,s,inputEnsemble,M1_truth,Actions,Trials)
%Reinforcement learning
%   
rng(s);
%% history records
rewardHis = zeros(1,opt.maxEpisode);
crossEntropyHis = zeros(1,opt.maxEpisode);
MaxReward = -Inf;

%% network weights
weightFromInputToHidden=2*rand(opt.hiddenUnitNum,data.mPFCnum*opt.RelevantSpikes+1)-1;
weightFromHiddenToOutput=2*rand(data.M1num_pre,opt.hiddenUnitNum+1)-1;

%% Train
if (opt.verbose<=3)
    disp(['----', opt.DataIndex,' RL Train start----']);
end
opt.Mode = 'train';
for episode=1:opt.maxEpisode
    % get batch data
    [batchInput,batchM1_truth,batchActions,opt] = DataLoader( ...
        inputEnsemble,M1_truth,Actions,Trials,opt);
    NumOfSamples = length(batchInput);
    inputUnit = [batchInput;ones(1,NumOfSamples)];

    % forward to get spikes
    [pOutput, hiddenUnit, spkOutPredict] = applynets_priori(inputUnit,...
        weightFromInputToHidden,weightFromHiddenToOutput,NumOfSamples, ...
        episode,opt.prioriM,opt.prioriN);
    
    % emulator: get predict motor
    if strcmp(opt.DataIndex, 'Simulations')
        [success,sucRate,motor_perform] = emulator_simu( ...
            spkOutPredict,batchActions,opt.M1index,data.his,data.modelName);
    else
        [success,sucRate,motor_perform] = emulator_real( ...
            spkOutPredict,batchActions,opt.M1index,data.his,data.modelName);
    end

    rewardHis(episode) = sucRate;
    
    % inner reward for less likely appear motor
    n_motor1 = sum(motor_perform==1)+1;
    n_motor2 = sum(motor_perform==2)+1;
    n_motor3 = sum(motor_perform==3)+1;
    n_max = max([n_motor1, n_motor2, n_motor3]);
    innerReward = (motor_perform==1).*(n_max/n_motor1-1)+ ...
                  (motor_perform==2).*(n_max/n_motor2-1)+ ...
                  (motor_perform==3).*(n_max/n_motor3-1);
    
    % discounted return
    reward = success + opt.epsilon*(1-episode/opt.maxEpisode)*innerReward;
    temp = reward; temp(isnan(reward)) = 0;

    smoothed_reward = conv(temp, opt.discountFactor.^((opt.discountLength-1):-1:0)/opt.discountLength);
    smoothed_reward = smoothed_reward(end-length(reward)+1:end);
    smoothed_reward(~isnan(reward)) = normalize(smoothed_reward(~isnan(reward))')';
    smoothed_reward(isnan(reward)) = 0;
    
    % store history and print log
    % cross-entropy between M1 predict rate and true spike train
    crossEntropyHis(episode) = CrossEntropyError(...
      [1-batchM1_truth;batchM1_truth],[1-pOutput;pOutput],NumOfSamples);
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
    [WeightDelta1,WeightDelta2] = getgradient(...
        smoothed_reward,pOutput,spkOutPredict,hiddenUnit,inputUnit,...
        weightFromHiddenToOutput,NumOfSamples...
    );

    % gradient descent
    if strcmp(opt.DataIndex, 'Simulations')
        lr = 0.1*(1-episode/opt.maxEpisode)+0.5;
    else
        lr = 0.7*(1-episode/opt.maxEpisode)+0.5;
    end
    weightFromHiddenToOutput = weightFromHiddenToOutput + ...
        lr*(WeightDelta1-0*weightFromHiddenToOutput);
    weightFromInputToHidden = weightFromInputToHidden + ...
        lr*(WeightDelta2-0*weightFromInputToHidden);
            
    % when some of the neurons has very constant firing rate, we re-initialize part of the weights
    if (mod(episode,100)==0)
        [temp1,temp2] = min(std(pOutput,[],2));
        if (temp1 < 0.01)
            weightFromHiddenToOutput(temp2,:) = 2*rand(1,opt.hiddenUnitNum+1)-1;
        end
    end
end

%% test model
opt.Mode = 'test';
[testInput,testM1_truth,testActions,opt] = DataLoader( ...
    inputEnsemble,M1_truth,Actions,Trials,opt);
TestSamples = length(testInput);
inputUnitTest = [testInput;ones(1,TestSamples)];
[pOutputTest, hiddenUnitTest, spkOutPredictTest] = applynets_priori(inputUnitTest,...
    L1WeightBestReward,L2WeightBestReward,TestSamples,MaxRewardEpisode, ...
    opt.prioriM,opt.prioriN);
if strcmp(opt.DataIndex, 'Simulations')
    [~,testSucRate,motor_perform_test] = emulator_simu(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
else
    [~,testSucRate,motor_perform_test] = emulator_real(spkOutPredictTest,testActions,opt.M1index,data.his,data.modelName);
end

if (opt.verbose<=3)
    disp(strcat(...
      '====',opt.DataIndex,' RL Test finish...Reward: ',num2str(testSucRate),'===='...
    ));
end

%% save results
clearvars("data","inputEnsemble","M1_truth","Actions","Trials");
if strcmp(opt.DataIndex, 'Simulations')
    targetFile = ['results/',opt.DataIndex,'_RL_',num2str(opt.testFold),'.mat'];
    save(targetFile)
else
    randstr = ['a':'z' '0':'9'];
    randId = [datestr(datetime,'mmmdd_HHMM'),'_',randstr(randi(length(randstr),1,4))];
    targetFile = ['results/',opt.DataIndex,'/RL_',num2str(opt.testFold),'_',randId,'.mat'];
    save(targetFile)
end
