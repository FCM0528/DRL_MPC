function plot_correction_results()
root=fileparts(fileparts(mfilename('fullpath')));folder=fullfile(root,'results','prediction_correction');
names={'original_70','original_planned','corrected_planned','constrained_planned'};
labels={'原 MPC / 70 km/h','原 MPC / 速度规划','校正与补偿 / 速度规划','校正与补偿及约束 / 速度规划'};
outputs=cell(4,1);rows=table();
for k=1:4
    v=load(fullfile(folder,[names{k} '_raw.mat']));outputs{k}=v.out;
    v.metrics.distance_grid_rmse_m=sqrt(mean(interp1(v.out.state_log.Data(:,1),v.out.error_log.Data(:,1),0:.25:120,'linear').^2));
    rows=[rows;struct2table(v.metrics)]; %#ok<AGROW>
end
writetable(rows,fullfile(folder,'comparison_120m.csv'));colors=lines(4);
f=figure('Visible','off','Color','w','Position',[50 50 1500 900]);tiledlayout(2,3,'TileSpacing','compact');
for panel=1:6
    ax=nexttile;hold(ax,'on');
    for k=1:4
        out=outputs{k};x=out.state_log.Data(:,1);a=out.lateral_safety_log.Data;
        switch panel
            case 1,y=out.state_log.Data(:,2);
            case 2,y=out.error_log.Data(:,1);
            case 3,y=a(:,5);
            case 4,y=a(:,1);
            case 5,y=a(:,3)*180/pi;
            case 6,y=a(:,8);
        end
        plot(ax,x,y,'LineWidth',1.4,'Color',colors(k,:),'DisplayName',labels{k});
    end
    out=outputs{4};x=out.state_log.Data(:,1);a=out.lateral_safety_log.Data;
    switch panel
        case 1,plot(ax,x,out.reference_log.Data(:,1),'k--','DisplayName','原始参考');title(ax,'路径 / m');
        case 2,title(ax,'横向误差 / m');
        case 3,plot(ax,x,a(:,6),'k--','DisplayName','计划车速');plot(ax,x,a(:,7),'r:','DisplayName','局部曲率车速上限');title(ax,'车速 / km/h');
        case 4,plot(ax,x,a(:,2),'k--','DisplayName','最终工况侧向限值');plot(ax,x,-a(:,2),'k--','HandleVisibility','off');title(ax,'实测侧向加速度 / m/s²');
        case 5,yline(ax,5,'k--','DisplayName','侧偏角限值');yline(ax,-5,'k--','HandleVisibility','off');title(ax,'质心侧偏角 / °');
        case 6,title(ax,'预测约束归一化松弛量');
    end
    xlim(ax,[0 120]);xlabel(ax,'X / m');grid(ax,'on');legend(ax,'Location','best','FontSize',8);
end
sgtitle('固定 Np=40、Nc=12、相同权重；区分降速与控制器校正的作用');set(findall(f,'-property','FontName'),'FontName','宋体');
savefig(f,fullfile(folder,'comparison_120m.fig'));exportgraphics(f,fullfile(folder,'comparison_120m.png'),'Resolution',160);close(f);disp(rows);
fprintf('CORRECTION_PLOTS_COMPLETE\n');
end
