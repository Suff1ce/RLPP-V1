function plotTestRasters(R, Rp, Rsp, neuronIdx, color)
plotTrialNo = min(size(R,2), 20);
plotOneModel(squeeze(R(neuronIdx,1:plotTrialNo,:)), 1.03, color(1,:))
plotOneModel(squeeze(Rp(neuronIdx,1:plotTrialNo,:)), 1, color(2,:))
plotOneModel(squeeze(Rsp(neuronIdx,1:plotTrialNo,:)), 0.97, color(3,:))

hold off
box off
set(gca, 'YColor', 'white')
set(gca, 'XColor', 'white')
set(gca, 'TickLength', [0 0])
ylim([0.97 1.05])
end

function plotOneModel(raster, base, color)
for rasterIdx=1:size(raster,1)
    x = raster(rasterIdx,:)*(base+rasterIdx*0.001);
    plot(x, "|", "Color",color, "MarkerSize",0.5); hold on
end
end
