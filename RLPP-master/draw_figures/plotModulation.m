function [l1, l2, l3] = plotModulation(plot_R,plot_Rp,plot_Rsp,modelColor,actionsColor,ymin,ymax,h)

area(-0.5:0.01:0, h*ones(1,51), 'FaceColor',actionsColor(1,:), 'FaceAlpha',0.3, 'EdgeColor','none')
hold on
area(0:0.01:0.5,  h*ones(1,51), 'FaceColor',actionsColor(2,:), 'FaceAlpha',0.3, 'EdgeColor','none')

l1 = plot(-0.5:0.01:0, plot_R(1:51), 'Color', modelColor(1,:), 'LineWidth',2, 'LineStyle',':');
plot(0.01:0.01:0.5, plot_R(52:end), 'Color', modelColor(1,:), 'LineWidth',2, 'LineStyle',':')

l3 = plot(-0.5:0.01:0,plot_Rsp(1:51), 'Color', modelColor(3,:), 'LineWidth',2, 'LineStyle','-.');
plot(0.01:0.01:0.5, plot_Rsp(52:end), 'Color', modelColor(3,:), 'LineWidth',2, 'LineStyle','-.')

l2 = plot(-0.5:0.01:0,plot_Rp(1:51), 'Color', modelColor(2,:), 'LineWidth',1.5);
plot(0.01:0.01:0.5, plot_Rp(52:end), 'Color', modelColor(2,:), 'LineWidth',1.5)

hold off
ylim([ymin, ymax])
set(gca, 'YTickLabel',[])
set(gca, 'XTickLabel',[])
set(gca, 'YColor', 'none')
set(gca, 'XColor', 'none')
set(gca, 'TickLength', [0 0])

end
