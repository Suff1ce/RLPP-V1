function plotSpikes(index, spikes, pos, color)
x = spikes*pos;
x(x==0) = nan;
plot(index, x, "|", "Color",color, "MarkerSize",6)
end
