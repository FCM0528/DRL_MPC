function plot_current_comparison()
% Publication/export plots from the actual deterministic comparison runs.
root=fileparts(fileparts(mfilename('fullpath')));s=read_mpc_settings();
folder=fullfile(root,'results',['comparison_' s.key]);
labels={s.fixed_label,['ddpg_' s.key],['td3_' s.key]};
names={'固定 MPC（40/12）','DDPG-MPC（300 回合）','TD3-MPC（300 回合）'};
colors=[0.15 0.15 0.15;0 0.40 0.72;0.82 0.27 0.10];styles={'-','--','-.'};
runs=cell(1,3);windowMetrics=table();distance=plot_station_limit();
for k=1:3
    data=load(fullfile(root,'results',labels{k},'raw.mat'),'out');runs{k}=data.out;
    windowMetrics=[windowMetrics;struct2table(metrics_to_distance(data.out,labels{k},distance))]; %#ok<AGROW>
end
writetable(windowMetrics,fullfile(folder,'metrics_120m.csv'));
f=figure('Visible','off','Color','w','Position',[80 60 1450 1050]);
layout=tiledlayout(f,3,2,'TileSpacing','compact','Padding','compact');
titles={'双移线轨迹','横向跟踪误差','航向误差','前轮转向指令','在线预测时域','跟踪误差指标'};
ylabs={'横向位置 Y / m','横向误差 / m','航向误差 / °','转角 / °','Np','误差 / m'};
for p=1:6
    ax=nexttile(layout);hold(ax,'on');
    if p<=5
        for k=1:3
            out=runs{k};x=out.state_log.Data(:,1);
            switch p
                case 1,y=out.state_log.Data(:,2);
                case 2,y=out.error_log.Data(:,1);
                case 3,y=rad2deg(out.error_log.Data(:,2));
                case 4,y=rad2deg(out.control_log.Data(:,1));
                case 5,y=out.control_log.Data(:,6);
            end
            if p==5
                stairs(ax,x,y,styles{k},'Color',colors(k,:),'LineWidth',1.5,'DisplayName',names{k});
            else
                plot(ax,x,y,styles{k},'Color',colors(k,:),'LineWidth',1.6,'DisplayName',names{k});
            end
        end
        if p==1
            plot(ax,runs{1}.state_log.Data(:,1),runs{1}.reference_log.Data(:,1),':', ...
                'Color',[0.35 0.60 0.25],'LineWidth',2,'DisplayName','CarSim 参考轨迹');
        end
        xlabel(ax,'纵向位置 X / m');xlim(ax,[0 distance]);
    else
        metrics=windowMetrics;
        bar(ax,[metrics.lateral_rmse_m metrics.max_abs_lateral_m]);
        xticks(ax,1:3);xticklabels(ax,{'固定 MPC','DDPG-MPC','TD3-MPC'});
    end
    title(ax,titles{p});ylabel(ax,ylabs{p});grid(ax,'on');box(ax,'on');
    ax.FontName='宋体';ax.FontSize=11;
    if p==6,legend(ax,{'RMSE','最大绝对误差'},'Location','best','FontName','宋体');
    else,legend(ax,'show','Location','best','FontName','宋体','FontSize',9);end
end
title(layout,'DLC SYX 70 km/h：固定 MPC 与 300 回合强化学习控制器', ...
    'FontName','宋体','FontSize',17);
subtitle(layout,'固定基准 Np=40、Nc=12；策略关闭探索噪声；曲线和误差统计范围均为 0–120 m', ...
    'FontName','宋体','FontSize',11);
saveplots(f,folder,'comparison_cn');close(f);

f=figure('Visible','off','Color','w','Position',[80 60 1300 800]);
layout=tiledlayout(f,2,1,'TileSpacing','compact','Padding','compact');
ax1=nexttile(layout);hold(ax1,'on');ax2=nexttile(layout);hold(ax2,'on');
for k=1:2
    algorithm={'ddpg','td3'};c=read_training_config();
    training=readtable(fullfile(root,'results',sprintf('training_%s_%s_ep300_seed%d',algorithm{k},s.key,c.training.seed),'training.csv'));
    assert(height(training)==300 && isequal(training.episode,(1:300)'));
    plot(ax1,training.episode,training.reward,'Color',0.65+0.35*colors(k+1,:), ...
        'HandleVisibility','off','LineWidth',0.5);
    plot(ax1,training.episode,movmean(training.reward,[19 0]),styles{k+1}, ...
        'Color',colors(k+1,:),'LineWidth',1.8,'DisplayName',upper(algorithm{k}));
    plot(ax2,training.episode,cumsum(training.steps),styles{k+1}, ...
        'Color',colors(k+1,:),'LineWidth',1.8,'DisplayName',upper(algorithm{k}));
end
title(ax1,'训练回报：浅线为逐回合值，粗线为最近 20 回合均值');ylabel(ax1,'回合累计奖励');
title(ax2,'实际累计策略交互次数');ylabel(ax2,'交互次数');
for ax=[ax1 ax2]
    ax.FontName='宋体';ax.FontSize=12;xlim(ax,[1 300]);grid(ax,'on');box(ax,'on');
    xlabel(ax,'训练回合');legend(ax,'show','Location','best','FontName','宋体');
end
title(layout,'DDPG / TD3 各 300 回合训练记录（种子 42）','FontName','宋体','FontSize',17);
saveplots(f,folder,'training_comparison');close(f);
writeReport(root,folder,s,names);
end
function saveplots(f,folder,name)
savefig(f,fullfile(folder,[name '.fig']));
exportgraphics(f,fullfile(folder,[name '.png']),'Resolution',200);
exportgraphics(f,fullfile(folder,[name '.pdf']),'ContentType','vector');
end
function writeReport(root,folder,s,names)
metrics=readtable(fullfile(folder,'metrics_120m.csv'));
fid=fopen(fullfile(folder,'COMPARISON.md'),'w','n','UTF-8');
cleanup=onCleanup(@()fclose(fid));
fprintf(fid,'# 300 回合训练与闭环对比\n\n');
fprintf(fid,'固定 MPC：Np=%d、Nc=%d；DDPG / TD3 各从新 Agent 开始实际训练 300 回合，种子 42。RL 的 Np 范围 %d–%d，Nc 保持 %d。\n\n',s.np,s.nc,s.min_np,s.max_np,s.nc);
fprintf(fid,'相同 DLC SYX 70 km/h CarSim 工况、相同参考轨迹；曲线与本表均使用 0–120 m 范围内的 MPC 采样点。评价关闭探索噪声，采用训练末回合 Agent。原始仿真仍运行至 150 m，全程指标另存 metrics.csv。\n\n');
fprintf(fid,'| 控制器 | 横向 RMSE / m | 最大绝对横向误差 / m | 航向 RMSE / rad | QP 失败数 | 到达 120 m | 统计范围 / m |\n');
fprintf(fid,'|---|---:|---:|---:|---:|---|---:|\n');
for k=1:3
    completed='否';if metrics.reached_range(k),completed='是';end
    fprintf(fid,'| %s | %.8f | %.8f | %.8f | %d | %s | %.4f |\n',names{k}, ...
        metrics.lateral_rmse_m(k),metrics.max_abs_lateral_m(k),metrics.heading_rmse_rad(k), ...
        metrics.qp_failures(k),completed,metrics.range_end_m(k));
end
fprintf(fid,'\n| 算法 | 实际回合数 | 实际策略交互数 | Actor 参数最大变化 |\n|---|---:|---:|---:|\n');
for algorithm={'ddpg','td3'}
    path=fullfile(root,'results',sprintf('training_%s_%s_ep300_seed42',algorithm{1},s.key));
    training=readtable(fullfile(path,'training.csv'));
    data=load(fullfile(path,'final_agent.mat'),'parameter_change');
    assert(height(training)==300 && data.parameter_change>0);
    fprintf(fid,'| %s | %d | %d | %.8g |\n',upper(algorithm{1}),height(training),sum(training.steps),data.parameter_change);
end
fprintf(fid,'\n- 闭环对比图：`comparison_cn.png`；另附 PDF 和可编辑 FIG。\n');
fprintf(fid,'- 相对固定 MPC 的横向 RMSE 变化：DDPG %+.2f%%，TD3 %+.2f%%（负值表示降低）。\n', ...
    100*(metrics.lateral_rmse_m(2)/metrics.lateral_rmse_m(1)-1), ...
    100*(metrics.lateral_rmse_m(3)/metrics.lateral_rmse_m(1)-1));
fprintf(fid,'- 训练曲线：`training_comparison.png`，浅线为逐回合奖励，粗线为最近 20 回合均值；另含实际累计交互数。\n');
fprintf(fid,'- 0–120 m 数值结果：`metrics_120m.csv`；全程指标另存 `metrics.csv`。每个控制器目录保留完整 `raw.mat` 和 `trace.csv`。\n\n');
fprintf(fid,'这是单工况、单种子训练与评价结果；300 回合不等于收敛或保证优于固定 MPC。原 47/7、500 回合结果保留为历史记录，不用于本次对比。\n');
end
