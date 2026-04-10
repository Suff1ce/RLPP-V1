function motor = decodingModel_simulation(spk, his)
N=his;
% get the movemean firing rate for the two M1 neurons
M1_1_mean = movmean(spk(1,:), [his-1, 0]);
M1_2_mean = movmean(spk(2,:), [his-1, 0]);

motor = (M1_2_mean<0.25 & M1_1_mean<0.25)*1+... % rest: when both are low firing
        (M1_2_mean>=0.25 & M1_1_mean<=0.25)*2+... % press low: when M1_2 is high and M1_1 is low
        (M1_2_mean>=0.25 & M1_1_mean>0.25)*3; % press high: when both are high firing
motor = motor(1:length(spk));
end
