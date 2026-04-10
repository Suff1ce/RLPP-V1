clear; clc;close all;
color_mat = [
    0      0.4470 0.7410;
    0.8500 0.3250 0.0980;
    0.9290 0.6940 0.1250;
    0.4940 0.1840 0.5560;
    0.4660 0.6740 0.1880;
    0.3010 0.7450 0.9330
];  
modelColor = [250 0 0; 54 56 131; 103 146 70] / 255;
f = figure();
f.Units = 'centimeters';
f.Position = [1 1 19 9]; % [left bottom width height]
f.Color = 'w';

%% Calculate Mutual Information
if ~exist("trained_results/mutualInformation.mat", "file")
    getMutualInformation
else
    load trained_results/mutualInformation.mat
end

%% M1-behavior compare
disp('Compare the Mutual information between M1 and behavior')
for ratIdx = 1:6
  for M1Idx = 1:length(M1_act{1,ratIdx})
    subplot(2,4,1)
    plot(M1_act{3,ratIdx}(M1Idx),M1_act{2,ratIdx}(M1Idx),'o', 'Color', [0.3 0.3 0.3], ...
      'MarkerSize',2, 'MarkerFaceColor',[0.3 0.3 0.3]); hold on

    subplot(2,4,2)
    plot(M1_act{1,ratIdx}(M1Idx),M1_act{3,ratIdx}(M1Idx),'o', 'Color', [0.3 0.3 0.3], ...
      'MarkerSize',2, 'MarkerFaceColor',[0.3 0.3 0.3]); hold on

    subplot(2,4,3)
    plot(M1_act{1,ratIdx}(M1Idx),M1_act{2,ratIdx}(M1Idx),'o', 'Color', [0.3 0.3 0.3], ...
      'MarkerSize',2, 'MarkerFaceColor',[0.3 0.3 0.3]); hold on

  end
end

subplot(2,4,1)
xlim([0 0.8]); ylim([0 0.8])
line([0 0.8], [0 0.8], ...
  'Color', 'k', 'lineStyle', '--', 'lineWidth', 1.5)
hold off

xlabel('${\rm MI(SLPP;~Movement)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(RLPP;~Movement)}$','Interpreter','latex','FontSize',7)
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)

text(-0.3, 1.07,'\textbf{a}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');

subplot(2,4,2)
xlim([0 0.8]); ylim([0 0.8])
line([0 0.8], [0 0.8], ...
  'Color', 'k', 'lineStyle', '--', 'lineWidth', 1.5)
hold off

xlabel('${\rm MI(M1~Recoridngs;~Movement)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(SLPP;~Movement)}$','Interpreter','latex','FontSize',7)
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)

text(-0.3, 1.07,'\textbf{b}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');

subplot(2,4,3)
xlim([0 0.8]); ylim([0 0.8])
line([0 0.8], [0 0.8], ...
  'Color', 'k', 'lineStyle', '--', 'lineWidth', 1.5)
hold off

xlabel('${\rm MI(M1~Recoridngs;~Movement)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(RLPP;~Movement)}$','Interpreter','latex','FontSize',7)
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)

text(-0.3, 1.07,'\textbf{c}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');

[H,p] = ttest(cell2mat(M1_act(2,:)),cell2mat(M1_act(3,:)), 'Tail','right');
if H==1
    disp(['RLPP significantly higher than SLPP, p = ', num2str(p)])
end
[H,p] = ttest(cell2mat(M1_act(1,:)),cell2mat(M1_act(3,:)), 'Tail','right');
if H==1
    disp(['Recordings significantly higher than SLPP, p = ', num2str(p)])
end
[H,p] = ttest(cell2mat(M1_act(1,:)),cell2mat(M1_act(2,:)));
if H==0
    disp(['Recordings and RLPP no significant differences, p = ', num2str(p)])
end

s1 = subplot(241);
s2 = subplot(242);
s3 = subplot(243);
s3.Position(1) = s3.Position(1) + s2.Position(1) - s1.Position(1) - 0.04;

s2.Position(1) = s3.Position(1)/2 + (s1.Position(1)+0.04)/2;
s1.Position(1) = s1.Position(1)+0.04;

disp(' ')

%% predicted M1-mPFC vs real M1-mPFC
disp('Compare the Mutual information between M1 and mPFC')
subplot(2,4,5)
plot([0.2 0.8],[0.2 0.8],'--k', 'lineWidth', 1.5)
hold on

subplot(2,4,6)
plot([0.2 0.8],[0.2 0.8],'--k', 'lineWidth', 1.5)
hold on


points = cell(1,3);
for modelIdx=2:3
for ratIdx = 1:6
  for M1Idx = 1:length(M1_act{1,ratIdx})
    subplot(2,4,modelIdx+3)
    l(modelIdx-1) = plot(M1_mPFC{1,ratIdx}(M1Idx), M1_mPFC{modelIdx,ratIdx}(M1Idx), 'o', 'Color',modelColor(modelIdx,:), 'MarkerSize',2, 'MarkerFaceColor',modelColor(modelIdx,:));
    points{modelIdx} = [points{modelIdx}; M1_mPFC{1,ratIdx}(M1Idx), M1_mPFC{modelIdx,ratIdx}(M1Idx)];
    hold on
  end
end
end

xc=points{2}(:,1);
yc=points{2}(:,2);
[p,S] = polyfit(xc,yc,1); 
[Y,dy] = polyconf(p,[min(xc)-0.05; unique(xc); max(xc)+0.05],S,'predopt','curve');
subplot(2,4,5)
plot([min(xc)-0.05; unique(xc); max(xc)+0.05],Y,'-', 'Color',modelColor(2,:), 'LineWidth',2)
fill([[min(xc)-0.05; unique(xc); max(xc)+0.05]; flipud([min(xc)-0.05; unique(xc); max(xc)+0.05])]', [Y-dy ; flipud(Y+dy)]', ...
  modelColor(2,:), 'FaceAlpha',0.3, 'EdgeAlpha',0)
[H,p] = ttest(points{2}(:,1),points{2}(:,2), 'Tail','left');
if H==1
    disp(['RLPP significantly higher than recordings, p = ', num2str(p)])
end
disp([num2str(sum(yc>xc)/length(yc)*100), '% neurons have larger MI values in RL'])
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
xlim([0.2 0.8]); ylim([0.2 1.1]);
xlabel('${\rm MI(M1~Recordings;~mPFC)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(RLPP;~mPFC)}$','Interpreter','latex','FontSize',7)
text(-0.3, 1.07,'\textbf{d}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');

xc=points{3}(:,1);
yc=points{3}(:,2);
disp(['SLPP CC: ', num2str(corr(xc, yc))])
[p,S] = polyfit(xc,yc,1); 
[Y,dy] = polyconf(p,[min(xc)-0.05; unique(xc); max(xc)+0.05],S,'predopt','curve');
[~,~,~,~,stats] = regress(yc, [xc ones(38, 1)]);
subplot(2,4,6)
plot([min(xc)-0.05; unique(xc); max(xc)+0.05],Y,'-', 'Color',modelColor(3,:), 'LineWidth',2)
fill([[min(xc)-0.05; unique(xc); max(xc)+0.05]; flipud([min(xc)-0.05; unique(xc); max(xc)+0.05])]', [Y-dy ; flipud(Y+dy)]', ...
  modelColor(3,:), 'FaceAlpha',0.3, 'EdgeAlpha',0)
disp(['Regression on SLPP, p = ', num2str(stats(3))])
[H,p] = ttest(points{3}(:,1),points{3}(:,2), 'Tail','right');
if H==1
    disp(['SLPP significantly lower than recordings, p = ', num2str(p)])
end
disp([num2str(sum(yc<xc)/length(yc)*100), '% neurons have larger MI values in SL'])
hold off
box off
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)
xlim([0.2 0.8]); ylim([0.2 1.1]);
xlabel('${\rm MI(M1~Recordings;~mPFC)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(SLPP;~mPFC)}$','Interpreter','latex','FontSize',7)
text(-0.3, 1.07,'\textbf{d}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');

s = subplot(245);
s.Position(1) = s.Position(1) - 0.02;
s = subplot(246);
s.Position(1) = s.Position(1) - 0.01;

disp(' ')

%% mPFC-behavior vs predicted M1-behavior
disp('Compare the Mutual information of mPFC-behavior and M1-behavior')
sp{2}=subplot(247);
sp{3}=subplot(248);

xl = [10 0];
yl = [10 0];

for modelIdx=2:3
subplot(sp{modelIdx})
for ratIdx = 1:6
  errorbar(mPFC_act{ratIdx}, ...
      min(M1_act{modelIdx,ratIdx}), ...
      0, ...
      max(M1_act{modelIdx,ratIdx}) - min(M1_act{modelIdx,ratIdx}), ...
      'Color',color_mat(ratIdx,:), 'LineWidth',1.5, 'CapSize',3.5)
  hold on
  for M1Idx = 1:length(M1_act{1,ratIdx})
    plot(mPFC_act{ratIdx}, M1_act{modelIdx,ratIdx}(M1Idx),'o', ...
        'Color', color_mat(ratIdx,:), 'MarkerSize',2, 'MarkerFaceColor',color_mat(ratIdx,:))
    hold on
  end
end
hold off
box off
temp = xlim; xl = [min(xl(1),temp(1))-0.05, max(xl(2), temp(2))+0.05];
temp = ylim; yl = [min(yl(1),temp(1)), max(yl(2), temp(2))];
end
for modelIdx=2:3
subplot(sp{modelIdx})
lh=findall(gca,'type','line');
% lh=lh(1:end-1);
xc=cell2mat(get(lh,'xdata'));
yc=cell2mat(get(lh,'ydata'));
for ratIdx = 1:6
  hold on
  plot(mPFC_act{ratIdx}, mean(M1_act{modelIdx,ratIdx}), 'Marker','x', ...
    'MarkerSize',6, 'MarkerFaceColor','k', 'MarkerEdgeColor','k')
  hold off
end
[p,S] = polyfit(xc,yc,1); 
[Y,dy] = polyconf(p,[min(xc)-0.05; unique(xc); max(xc)+0.05],S,'predopt','curve');
[b,~,~,~,stats] = regress(yc, [xc ones(38, 1)]);
hold on
plot([min(xc)-0.05; unique(xc); max(xc)+0.05],Y,'-', 'Color',modelColor(modelIdx,:), 'LineWidth',2)
fill([[min(xc)-0.05; unique(xc); max(xc)+0.05]; flipud([min(xc)-0.05; unique(xc); max(xc)+0.05])]', [Y-dy ; flipud(Y+dy)]', ...
  modelColor(modelIdx,:), 'FaceAlpha',0.3, 'EdgeAlpha',0)
if modelIdx==2
  disp(['slope of RLPP regression: ', num2str(p(1))])
  disp(['cc of RLPP: ', num2str(corr(xc,yc))])
else
  disp(['slope of SLPP regression: ', num2str(p(1))])
  disp(['cc of SLPP: ', num2str(corr(xc,yc))])
end

plot([0. 0.95],[0. 0.95],'--k', 'LineWidth',2); hold on
hold off
xlim([0. 0.9]); ylim([0 0.9])
set(gca, 'TickLabelInterpreter', 'latex','FontSize',7)
set(gca, 'XColor', 'k', 'YColor', 'k', 'LineWidth',1)

[H, p] = ttest(xc,yc, 'Tail','right');
if H==1
    if modelIdx==2
        disp(['RLPP M1 significantly lower than mPFC, p: ', num2str(p)])
    else
        disp(['SLPP M1 significantly lower than mPFC, p: ', num2str(p)])
    end
end

ratio = sum(xc>yc)/length(xc);
if modelIdx==2
    disp(['Ratio of RLPP M1 lower than mPFC: ', num2str(ratio)])
else
    disp(['Ratio of SLPP M1 lower than mPFC: ', num2str(ratio)])
end
end

subplot(sp{2})
xlabel('${\rm MI(mPFC;~Movement)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(RLPP;~Movement)}$','Interpreter','latex','FontSize',7)
text(-0.3, 1.07,'\textbf{f}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');
sp{2}.Position(1) = sp{2}.Position(1) + 0.01;
tmp = get(gca, 'Children');
set(gca, 'Children', [tmp(1:2);tmp(5:end);tmp(3:4)])

subplot(sp{3})
xlabel('${\rm MI(mPFC;~Movement)}$','Interpreter','latex','FontSize',7)
ylabel('${\rm MI(SLPP;~Movement)}$','Interpreter','latex','FontSize',7)
text(-0.3, 1.07,'\textbf{g}', 'Interpreter','latex', ...
    'FontSize',7,'EdgeColor','none', 'Units','normalized');
sp{3}.Position(1) = sp{3}.Position(1) + 0.02;
tmp = get(gca, 'Children');
set(gca, 'Children', [tmp(1:2);tmp(5:end);tmp(3:4)])