DataNameList = {'rat01','rat02','rat03','rat04','rat05','rat06'};

modDep = zeros(12,0);
for dataIdx=1:6

temp = load(['results/', DataNameList{dataIdx}, '_RL_', num2str(fold(dataIdx)), '.mat']);
testActions = temp.testActions;
testM1_truth = temp.testM1_truth;
RL.spkOutPredictTest = spkOutPredictTest;
SL = load(['results/', DataNameList{dataIdx}, '_Sup_', num2str(fold(dataIdx)), '.mat'], "spkOutPredictTest");
getTestRasters

base = size(modDep, 2);
for n=1:size(lR, 1)
  nIdx = n+base;
  % real
  temp_1 = sum(squeeze(lR(n,:,:)))/size(lR,2);
  temp_2 = sum(squeeze(hR(n,:,:)))/size(hR,2);
  modDep(1,nIdx) = -(mean(temp_1(1:50))-mean(temp_1(52:101))); % low-rest
  modDep(2,nIdx) = -(mean(temp_2(1:50))-mean(temp_2(52:101))); % high-rest
  modDep(3,nIdx) = -(mean(temp_1(52:101))-mean(temp_2(52:101))); % high-low
  modDep(4,nIdx) = -(mean(temp_1(1:50))-mean(temp_2(1:50))); % high rest -low rest
  % RLPP
  temp_1 = sum(squeeze(lRp(n,:,:)))/size(lRp,2);
  temp_2 = sum(squeeze(hRp(n,:,:)))/size(hRp,2);
  modDep(5,nIdx) = -(mean(temp_1(1:50))-mean(temp_1(52:101))); % low-rest
  modDep(6,nIdx) = -(mean(temp_2(1:50))-mean(temp_2(52:101))); % high-rest
  modDep(7,nIdx) = -(mean(temp_1(52:101))-mean(temp_2(52:101))); % high-low
  modDep(8,nIdx) = -(mean(temp_1(1:50))-mean(temp_2(1:50))); % high rest -low rest
  % SLPP
  temp_1 = sum(squeeze(lRsp(n,:,:)))/size(lRsp,2);
  temp_2 = sum(squeeze(hRsp(n,:,:)))/size(hRsp,2);
  modDep(9,nIdx) = -(mean(temp_1(1:50))-mean(temp_1(52:101))); % low-rest
  modDep(10,nIdx) = -(mean(temp_2(1:50))-mean(temp_2(52:101))); % high-rest
  modDep(11,nIdx) = -(mean(temp_1(52:101))-mean(temp_2(52:101))); % high-low
  modDep(12,nIdx) = -(mean(temp_1(1:50))-mean(temp_2(1:50))); % high rest -low rest
end

end

save trained_results/modDep.mat modDep
