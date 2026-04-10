function plotActions(index, actions, color, xRange)
temp = selectOneAction(actions, 1);
plot(index, temp, 'Color', color(1,:), 'LineWidth', 15);
hold on;
temp = selectOneAction(actions, 2);
plot(index, temp, 'Color', color(2,:), 'LineWidth', 15);
temp = selectOneAction(actions, 3);
plot(index, temp, 'Color', color(3,:), 'LineWidth', 15);
hold off
ylim([0.8 1.2])
xlim(xRange)
box off
set(gca, 'YColor', 'white')
set(gca, 'XColor', 'white')
set(gca, 'TickLength', [0 0])
end

function x = selectOneAction(actions, a)
x = 1*(actions==a); % pick out one actions
% change 010 into 111 to line up (If not, 010 can not be seen)
singleIdxes = strfind(x, [0 1 0]);
for i=1:length(singleIdxes)
    x(singleIdxes(i):(singleIdxes(i)+2)) = [1 1 1];
end
% do not plot 0 values
x(x==0)=nan;
end