function decoded_movements = decodingModel_manual(M1ensemble)
%% Mannual designed tunning function
% get each M1's firing rate
M1_fr = zeros(4, length(M1ensemble));
for i = 1:4
  M1_fr(i, :) = mean(M1ensemble(i:4:end,:));
end

% Rest: Neuron 1 low  & 2 low   & 3 high & 4 high
% Low:  Neuron 1 high & 2 any   & 3 low  & 4 any (Neuron 1 > 3)
% High: Neuron 1 any  & 2 high  & 3 any  & 4 low (Neuron 2 > 4)
decoded_movements = [
  0.3*ones(1,length(M1_fr))
  M1_fr(1,:) - M1_fr(3,:); 
  M1_fr(2,:) - M1_fr(4,:);
];

end
