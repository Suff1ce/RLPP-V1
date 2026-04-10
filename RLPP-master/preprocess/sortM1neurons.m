% sort M1 neurons by mutual information
M1delayMI   = zeros(M1num, 71);
startIdx = 0;
endIdx   = 50;
for delay=startIdx:endIdx % delay vary from 0~500ms
    for n=1:M1num
         M1delayMI(n,delay+1-startIdx) = MIcontinuous( ...
             gaussianSmooth(M1(1-startIdx:end-delay+startIdx,n), 5)', ...
             gaussianSmooth(actions(delay+1-startIdx:end+startIdx), 5) ...
         );
    end
end
[M1maxMI, M1delay] = max(M1delayMI(:,1-startIdx:1-startIdx+endIdx), [], 2); % reasonable delay should be within 0~500ms
[MIsorted,M1order] = sort(M1maxMI, "descend");

figure()
plot(MIsorted)
title("Sorted Mutual Information")
xlabel("M1 indexes")
ylabel("Mutual Information")
