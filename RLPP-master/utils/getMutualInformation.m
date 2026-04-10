DataNameList = {'rat01','rat02','rat03','rat04','rat05','rat06'};

M1_act = cell(3, 6);
M1_mPFC = cell(3, 6);
mPFC_act = cell(1, 6);
for dataIdx=1:6

    RL = load(['results/', DataNameList{dataIdx}, '_RL_', num2str(fold(dataIdx)), '.mat']);
    SL = load(['results/', DataNameList{dataIdx}, '_Sup_', num2str(fold(dataIdx)), '.mat']);

    M1_act_real = zeros(1, length(RL.opt.M1index));
    M1_act_RLPP = zeros(1, length(RL.opt.M1index));
    M1_act_SLPP = zeros(1, length(RL.opt.M1index));
    M1_mPFC_real_all = zeros(length(RL.opt.M1index), size(RL.testInput,1));
    M1_mPFC_RLPP_all = zeros(length(RL.opt.M1index), size(RL.testInput,1));
    M1_mPFC_SLPP_all = zeros(length(RL.opt.M1index), size(RL.testInput,1));
    M1_mPFC_real = zeros(1, length(RL.opt.M1index));
    M1_mPFC_RLPP = zeros(1, length(RL.opt.M1index));
    M1_mPFC_SLPP = zeros(1, length(RL.opt.M1index));
    mPFC_act_rat = zeros(1, length(RL.opt.M1index));

    for M1idx = 1:length(RL.opt.M1index)
        M1_act_real(M1idx) = MIcontinuous(gaussianSmooth(RL.testM1_truth(M1idx,RL.testActions>0),10), RL.testActions(RL.testActions>0));
        M1_act_RLPP(M1idx) = MIcontinuous(gaussianSmooth(RL.spkOutPredictTest(M1idx,RL.testActions>0),10), RL.testActions(RL.testActions>0));
        M1_act_SLPP(M1idx) = MIcontinuous(gaussianSmooth(SL.spkOutPredictTest(M1idx,RL.testActions>0),10), RL.testActions(RL.testActions>0));
        parfor n=1:size(RL.testInput,1)
            M1_mPFC_real_all(M1idx, n) = MIcontinuous(gaussianSmooth(RL.testM1_truth(M1idx,RL.testActions>0),10), RL.testInput(n,RL.testActions>0));
            M1_mPFC_RLPP_all(M1idx, n) = MIcontinuous(gaussianSmooth(RL.spkOutPredictTest(M1idx,RL.testActions>0),10), RL.testInput(n,RL.testActions>0));
            M1_mPFC_SLPP_all(M1idx, n) = MIcontinuous(gaussianSmooth(SL.spkOutPredictTest(M1idx,RL.testActions>0),10), RL.testInput(n,RL.testActions>0));
        end
        M1_mPFC_real(M1idx) = max(M1_mPFC_real_all(M1idx,:));
        M1_mPFC_RLPP(M1idx) = max(M1_mPFC_RLPP_all(M1idx,:));
        M1_mPFC_SLPP(M1idx) = max(M1_mPFC_SLPP_all(M1idx,:));
    end
    parfor n=1:size(RL.testInput,1)
        mPFC_act_rat(n) = MIcontinuous(RL.testInput(n,RL.testActions>0), RL.testActions(RL.testActions>0));
    end
    mPFC_act{dataIdx} = max(mPFC_act_rat);

    [M1_act{:,dataIdx}] = deal(M1_act_real, M1_act_RLPP, M1_act_SLPP);
    [M1_mPFC{:,dataIdx}] = deal(M1_mPFC_real, M1_mPFC_RLPP, M1_mPFC_SLPP);

end
save results/mutualInformation.mat M1_act M1_mPFC mPFC_act
