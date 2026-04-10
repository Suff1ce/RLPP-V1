NumOfIter = 24;
percentErrors=zeros(M1num,NumOfIter);
testPercentErrors=zeros(M1num, NumOfIter);
netVec = cell(M1num,NumOfIter);

for i=1:M1num
    x = X_PART(x_full, M1order(1:i), M1num, his);
    t = y_onehot;
    parfor j=1:NumOfIter
        disp(['start ', num2str(i), ' M1 neurons ', num2str(j), ' iter.']);
        net = patternnet(hiddenLayerSize, trainFcn);
        net.divideParam.trainRatio = 70/100;
        net.divideParam.valRatio = 15/100;
        net.divideParam.testRatio = 15/100;
        net.performFcn = 'crossentropy';  % Cross-Entropy
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

%%
figure()
subplot(311); 
errorbar(1:M1num, ...
    mean(percentErrors, 2), ...
    mean(percentErrors, 2)-min(percentErrors,[],2), ...
    max(percentErrors,[],2)-mean(percentErrors, 2));
xlim([0 M1num+1]); title('all percent Errors'); ylim([0 0.4])
subplot(312); 
errorbar(1:M1num, ...
    mean(testPercentErrors, 2), ...
    mean(testPercentErrors, 2)-min(testPercentErrors,[],2), ...
    max(testPercentErrors,[],2)-mean(testPercentErrors, 2));
xlim([0 M1num+1]); title('test percent Errors'); xlabel('No of M1 neurons');
ylim([0 0.4])
subplot(313); 
errorbar(1:M1num, ...
    mean(trainPercentErrors, 2), ...
    mean(trainPercentErrors, 2)-min(trainPercentErrors,[],2), ...
    max(trainPercentErrors,[],2)-mean(trainPercentErrors, 2));
xlim([0 M1num+1]); title('train percent Errors'); xlabel('No of M1 neurons');
ylim([0 0.4])
