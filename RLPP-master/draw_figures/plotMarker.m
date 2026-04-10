function plotMarker(x, y)

plot([x-0.08 x+0.08], [y y], 'k-', 'LineWidth',1.5);
plot(x, y, '^', 'Color','k', 'MarkerFaceColor', 'k', 'MarkerSize',8);

end