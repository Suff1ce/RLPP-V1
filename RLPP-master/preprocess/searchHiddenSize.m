x = x_full;
t = y_onehot;

NumOfPower = 8; % explore in [2, 4, 8, ..., 256]
NumOfIter = 24; % run multiple times
percentErrors=zeros(NumOfPower,NumOfIter);
testPercentErrors=zeros(NumOfPower, NumOfIter);
trainPercentErrors=zeros(NumOfPower, NumOfIter);
netVec = cell(NumOfPower, NumOfIter);

for i=1:NumOfPower
    trainFcn = 'trainscg';
    hiddenLayerSize = 2^i;
    parfor j=1:NumOfIter
        disp(['start ', num2str(2^i), ' Hidden units...', num2str(j), ' iter.']);
        net = patternnet(hiddenLayerSize, trainFcn);
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;
        net.performFcn = 'crossentropy';
        init(net);
        [net,tr] = train(net,x,t);
        y = net(x);
        tind = vec2ind(t);
        yind = vec2ind(y);
        percentErrors(i,j) = sum(tind ~= yind)/numel(tind)
        testPercentErrors(i,j) = sum(tind(tr.testMask{1}(1,:)==1) ~= yind(tr.testMask{1}(1,:)==1))/numel(tind(tr.testMask{1}(1,:)==1));
        trainPercentErrors(i,j) = sum(tind(tr.trainMask{1}(1,:)==1) ~= yind(tr.trainMask{1}(1,:)==1))/numel(tind(tr.trainMask{1}(1,:)==1));
        netVec{i,j} = net;
    end
end

%% show results
figure()
subplot(311);
errorbar(1:8, ...
    mean(percentErrors, 2), ...
    mean(percentErrors, 2)-min(percentErrors,[],2), ...
    max(percentErrors,[],2)-mean(percentErrors, 2));
hold on; line([0.5 8.5],[min(min(percentErrors)),min(min(percentErrors))]); hold off
xlim([0 9]); title('all percent Errors'); ylim([0 0.3])
subplot(312); 
errorbar(1:8, ...
    mean(testPercentErrors, 2), ...
    mean(testPercentErrors, 2)-min(testPercentErrors,[],2), ...
    max(testPercentErrors,[],2)-mean(testPercentErrors, 2));
hold on; line([0.5 8.5],[min(min(testPercentErrors)),min(min(testPercentErrors))]); hold off
xlim([0 9]); title('test percent Errors'); ylim([0 0.3])
subplot(313); 
errorbar(1:8, ...
    mean(trainPercentErrors, 2), ...
    mean(trainPercentErrors, 2)-min(trainPercentErrors,[],2), ...
    max(trainPercentErrors,[],2)-mean(trainPercentErrors, 2));
hold on; line([0.5 8.5],[min(min(trainPercentErrors)),min(min(trainPercentErrors))]); hold off
xlim([0 9]); title('train percent Errors'); xlabel('2^x of hidden units'); ylim([0 0.3])
