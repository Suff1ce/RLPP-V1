function testTrialSucRate = calculateTrialSucRate(success, testActions)

trialNum = 0;
successTrialNum = 0;
for timeIdx=2:length(success)
    if testActions(timeIdx-1)==0 && testActions(timeIdx)==1
        start = timeIdx;
        continue
    elseif timeIdx==length(success) || (testActions(timeIdx)>1 && testActions(timeIdx+1)==0)
        stop = timeIdx;
        trialNum = trialNum + 1;
        if sum(success(start:stop)==1)/(sum(success(start:stop)==0)+sum(success(start:stop)==1))>0.7
            successTrialNum = successTrialNum + 1;
        end
    else
        continue
    end
end

testTrialSucRate = successTrialNum/trialNum;

end
