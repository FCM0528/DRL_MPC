function finalize_nominal_speed()
% Verify the saved model's ordinary Run path, without any speed override.
root=setup_matlab_mpc();folder=fullfile(root,'results','nominal_speed_correction');m='DRL_MPC_MATLAB';
open_system(fullfile(root,'models',[m '.slx']));p=evalin('base','matlab_mpc_parameters');
assert(~p.speedPlanActive&&p.np==40&&p.nc==12);
assert(contains(get_param(Simulink.ID.getFullName([m ':1']),'SIMFILE'),[filesep 'case' filesep]));
notes=find_system(m,'FindAll','on','Type','annotation');
for h=notes'
    a=get_param(h,'Object');
    if contains(a.Text,'计划目标'),a.Text='实际车速 / 原目标70 / 曲率能力诊断（不调速，km/h）';a.FontName='宋体';end
    if contains(a.Text,'参考可行性超限比'),a.Text='QP 松弛量 / 曲率需求可行性诊断（不调速）';a.FontName='宋体';end
end
save_system(m);out=sim(Simulink.SimulationInput(m));
v=load(fullfile(folder,'current70_raw.mat'),'out','metrics');
difference=max(abs(out.state_log.Data-v.out.state_log.Data),[],'all');assert(difference<1e-6);
assert(v.metrics.qp_failures==0&&v.metrics.reached_range);
baseline=readtable(fullfile(folder,'original70_metrics.csv'));
scopes=find_system(m,'LookUnderMasks','all','FollowLinks','off','BlockType','Scope');assert(numel(scopes)==10);
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
data=struct('passed',true,'verification_scope','original 70 km/h case restored, compiled and simulated', ...
    'model',m,'default_profile',3,'speed_planning_enabled',false,'speed_target_kmh',70, ...
    'speed_validation_band_kmh',[68 72],'speed_validation_band_is_design_check',true, ...
    'np',40,'nc',12,'scopes',numel(scopes),'controller_mex_loaded',false, ...
    'default_entry_state_max_difference',difference,'tracking_rmse_improved',v.metrics.lateral_rmse_m<baseline.lateral_rmse_m, ...
    'actual_ay_within_soft_design_limit',v.metrics.max_ay_excess_mps2<1e-6, ...
    'constraint_limitations','soft predicted state constraints; actual excess is reported separately', ...
    'metrics',v.metrics,'corrected_environment_training_episodes',0,'real_time_qualified',false);
fid=fopen(fullfile(folder,'acceptance.json'),'w');fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));fclose(fid);
plotResults();fprintf('NOMINAL_SPEED_DEFAULT_ENTRY_PASS\n');
    function plotResults()
        names={'original70','constrained70','current70'};labels={'原 MPC / 原70工况','上轮修正恢复70 / 原70工况','本轮同速修正 / 原70工况'};
        f=figure('Visible','off','Color','w','Position',[50 50 1450 850]);tiledlayout(2,3,'TileSpacing','compact');
        outputs=cell(3,1);rows=table();
        for j=1:3
            z=load(fullfile(folder,[names{j} '_raw.mat']),'out');outputs{j}=z.out;
            q=metrics_to_distance(z.out,names{j},120);s=z.out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=z.out.lateral_safety_log.Data(mask,:);
            q.min_speed_kmh=min(s(mask,3))*3.6;q.max_speed_kmh=max(s(mask,3))*3.6;q.max_ay_mps2=max(abs(a(:,1)));
            q.max_ay_excess=max(0,max(abs(a(:,1))-a(:,2)));q.max_beta_deg=max(abs(a(:,3)))*180/pi;q.max_slack=max(a(:,8));
            q.distance_grid_rmse_m=sqrt(mean(interp1(s(:,1),z.out.error_log.Data(:,1),0:.25:120).^2));
            rows=[rows;struct2table(q)]; %#ok<AGROW>
        end
        writetable(rows,fullfile(folder,'current_comparison_120m.csv'));
        for panel=1:6
            ax=nexttile;hold(ax,'on');
            for j=1:3
                z=outputs{j};x=z.state_log.Data(:,1);
                switch panel
                    case 1,y=z.state_log.Data(:,2);
                    case 2,y=z.error_log.Data(:,1);
                    case 3,y=z.state_log.Data(:,3)*3.6;
                    case 4,y=z.lateral_safety_log.Data(:,1);
                    case 5,y=z.lateral_safety_log.Data(:,3)*180/pi;
                    case 6,y=z.lateral_safety_log.Data(:,8);
                end
                plot(ax,x,y,'LineWidth',1.4,'DisplayName',labels{j});
            end
            z=outputs{3};x=z.state_log.Data(:,1);a=z.lateral_safety_log.Data;
            switch panel
                case 1,plot(ax,x,z.reference_log.Data(:,1),'k--','DisplayName','原始参考');title(ax,'实际路径 / m');
                case 2,title(ax,'横向误差 / m');
                case 3,yline(ax,70,'k--','DisplayName','原车速目标70');title(ax,'纵向速度 / km/h（未启用降速）');ylim(ax,[68 72]);
                case 4,plot(ax,x,a(:,2),'k--','DisplayName','软约束设计限值');plot(ax,x,-a(:,2),'k--','HandleVisibility','off');title(ax,'实测侧向加速度 / m/s²');
                case 5,yline(ax,5,'k--','DisplayName','设计限值');yline(ax,-5,'k--','HandleVisibility','off');title(ax,'质心侧偏角 / °');
                case 6,title(ax,'侧向预测约束归一化松弛量');
            end
            xlim(ax,[0 120]);xlabel(ax,'X / m');grid(ax,'on');legend(ax,'Location','best','FontSize',8);
        end
        sgtitle('相同原70 km/h工况，Np=40、Nc=12；纵向速度参与预测，不作降速规划');
        set(findall(f,'-property','FontName'),'FontName','宋体');savefig(f,fullfile(folder,'comparison_120m.fig'));
        exportgraphics(f,fullfile(folder,'comparison_120m.png'),'Resolution',160);close(f);
    end
end
