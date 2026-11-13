function plot_drl_diagnosis()
root=fileparts(fileparts(mfilename('fullpath')));folder=fullfile(root,'results','drl_diagnosis');
files={fullfile(root,'results','fixed_np40_nc12','raw.mat'), ...
    fullfile(root,'results','ddpg_np40_nc12','raw.mat'),fullfile(root,'results','td3_np40_nc12','raw.mat'), ...
    fullfile(folder,'high_y_high_psi_raw.mat'),fullfile(folder,'extended_qy15_raw.mat')};
names={'固定 MPC','DDPG-MPC','TD3-MPC','范围内固定权重探针','扩范围固定权重探针'};
colors=[.15 .15 .15;0 .4 .72;.82 .27 .1;.55 .2 .65;.1 .55 .25];styles={'-','--','-.',':','-'};
f=figure('Visible','off','Color','w','Position',[80 60 1400 940]);layout=tiledlayout(2,2,'Padding','compact','TileSpacing','compact');
ax1=nexttile;hold(ax1,'on');ax2=nexttile;hold(ax2,'on');ax3=nexttile;hold(ax3,'on');ax4=nexttile;hold(ax4,'on');
rmse=zeros(1,5);
for k=1:5
    loaded=load(files{k},'out');out=loaded.out;
    x=out.state_log.Data(:,1);e=out.error_log.Data;
    plot(ax1,x,e(:,1),styles{k},'Color',colors(k,:),'LineWidth',1.5,'DisplayName',names{k});
    plot(ax3,x,rad2deg(e(:,2)),styles{k},'Color',colors(k,:),'LineWidth',1.5,'DisplayName',names{k});
    m=metrics_to_distance(out,names{k},120);rmse(k)=m.lateral_rmse_m;
end
bar(ax2,rmse,'FaceColor','flat','CData',colors);xticks(ax2,1:5);
xticklabels(ax2,{'固定 MPC','DDPG','TD3','范围内探针','扩范围探针'});xtickangle(ax2,15);
title(ax1,'横向误差：当前 DRL 与固定参数探针');ylabel(ax1,'横向误差 / m');
title(ax2,'0–120 m 横向 RMSE（探针不是 DRL 成绩）');ylabel(ax2,'RMSE / m');
title(ax3,'航向误差：同时检查精度之间的权衡');ylabel(ax3,'航向误差 / °');
terms=readtable(fullfile(folder,'reward_decomposition_120m.csv'));
loss=terms.total_loss_120m;
shares=100*[terms.lateral,terms.growth,terms.course,loss-terms.lateral-terms.growth-terms.course]./loss;
bar(ax4,shares,'stacked');xticks(ax4,1:3);xticklabels(ax4,{'固定 MPC','DDPG','TD3'});
title(ax4,'当前奖励的惩罚贡献');ylabel(ax4,'占比 / %');ylim(ax4,[0 100]);
legend(ax4,{'横向误差','误差增长','航迹方向误差','其他项'},'FontName','宋体','Location','southoutside','Orientation','horizontal');
for ax=[ax1 ax2 ax3 ax4],ax.FontName='宋体';ax.FontSize=11;grid(ax,'on');box(ax,'on');end
for ax=[ax1 ax3]
    xlim(ax,[0 120]);xlabel(ax,'纵向位置 X / m');
    legend(ax,'show','FontName','宋体','FontSize',9,'Location','best');
end
title(layout,'DRL 效果诊断：权重范围与奖励分解','FontName','宋体','FontSize',17);
subtitle(layout,'固定基准和固定参数探针使用 Np=40、Nc=12；现有 DRL 的 Np 仍自适应；全部距离曲线至 120 m', ...
    'FontName','宋体','FontSize',11);
savefig(f,fullfile(folder,'diagnosis_120m.fig'));
exportgraphics(f,fullfile(folder,'diagnosis_120m.png'),'Resolution',200);
exportgraphics(f,fullfile(folder,'diagnosis_120m.pdf'),'ContentType','vector');close(f);
end
