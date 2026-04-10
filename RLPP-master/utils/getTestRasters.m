getPlotModulation = @(x) [
  smoothdata(squeeze(sum(x(:,:,1:51), 2))/size(x,2), 2, 'gaussian', 25) ...
  smoothdata(squeeze(sum(x(:,:,52:end), 2))/size(x,2), 2, 'gaussian', 25)
];

[lR,hR] = getRaster(testActions,testM1_truth);
plot_lR = getPlotModulation(lR);
plot_hR = getPlotModulation(hR);
[lRp,hRp] = getRaster(testActions,RL.spkOutPredictTest);
plot_lRp = getPlotModulation(lRp);
plot_hRp = getPlotModulation(hRp);
[lRsp,hRsp] = getRaster(testActions,SL.spkOutPredictTest);
plot_lRsp = getPlotModulation(lRsp);
plot_hRsp = getPlotModulation(hRsp);
