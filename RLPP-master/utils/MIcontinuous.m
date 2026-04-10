function MI = MIcontinuous(signal1, signal2)
%Calcutlate the mutual information between signals
% Density estimated by Parzen window
%   INPUT:
%       signal1, signal2 - timeBins length vector
%   OUTPUT:
%       MI - mutual information. Scaler

%%
edges1 = linspace(min(signal1), max(signal1), 21);
edges2 = linspace(min(signal2), max(signal2), 21);

p_xy = histcounts2(signal1, signal2, edges1, edges2, 'Normalization', 'probability');
p_x  = histcounts(signal1, edges1, 'Normalization', 'probability');
p_y  = histcounts(signal2, edges2, 'Normalization', 'probability');

MI = nansum(p_xy .* log2(p_xy ./ (p_x' * p_y)), 'all');

end

%% use ksdensity
% function MI = MIcontinuous(signal1, signal2)
% % calcutlate the mutual information between signals
% % signal - timeBins length vector
% % MI - scaler
% 
% edges1 = linspace(min(signal1), max(signal1), 21)';
% edges2 = linspace(min(signal2), max(signal2), 21)';
% 
% p_x = ksdensity(signal1, edges1);
% p_x = p_x / sum(p_x);
% p_y = ksdensity(signal2, edges2);
% p_y = p_y / sum(p_y);
% 
% [x1,x2] = meshgrid(edges1, edges2);
% x1 = x1(:);
% x2 = x2(:);
% edges = [x1 x2];
% 
% p_xy = ksdensity([signal1 signal2], edges);
% p_xy = p_xy / sum(p_xy);
% p_xy = reshape(p_xy, 21, 21)';
% 
% MI = nansum(p_xy .* log2(p_xy ./ (p_x * p_y')), 'all');
% 
% end