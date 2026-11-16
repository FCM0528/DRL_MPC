function comparison=compare_matlab_controllers(agentFile,corrected)
% Evaluate new native training, matched fixed MPC and an explicit ablation.
arguments
    agentFile (1,:) char=''
    corrected (1,1) logical=false
end
root=setup_matlab_mpc();if isempty(agentFile)
    training='matlab_td3_ep300_seed42';if corrected,training='matlab_td3_corrected70_ep300_seed42';end
    agentFile=fullfile(root,'results',training,'final_agent.mat');
end
[agent,metadata]=load_paper_agent(agentFile);m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
folder=fullfile(root,'results','matlab_comparison');if corrected,folder=fullfile(root,'results','nominal_speed_correction','td3_comparison');end
if ~isfolder(folder),mkdir(folder);end
profile=0;caseName='case';if corrected,profile=3;end
trainFolder=fileparts(agentFile);t=readtable(fullfile(trainFolder,'training.csv'));
ends=10:10:height(t);scores=zeros(size(ends));
for k=1:numel(ends),scores(k)=mean(t.reward(ends(k)-9:ends(k)));end
if isempty(ends),episode=height(t);score=mean(t.reward);selectedFile=agentFile;
else,[score,index]=max(scores);episode=ends(index);selectedFile=fullfile(trainFolder,'checkpoints',sprintf('Agent%d.mat',episode));end
labels={'matched_fixed','native_td3_final','same_final_policy_lambda_zero','native_td3_reward_selected'};
comparison=table();outputs=cell(1,4);
for k=1:4
    mode=k-1;if k==4,agent=load_paper_agent(selectedFile);mode=1;end
    assignin('base','agent',agent);in=Simulink.SimulationInput(m);in=in.setVariable('controller_mode',mode);
    in=in.setVariable('matlab_mpc_parameters',matlabMpcProfile(root,profile,false));
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim',caseName,'simfile.sim'));out=sim(in);
    metrics=metrics_to_distance(out,labels{k},120);show=out.state_log.Data(:,1)>=0&out.state_log.Data(:,1)<=120;
    metrics.lateral_mae_m=mean(abs(out.error_log.Data(show,1)));
    metrics.heading_mae_rad=mean(abs(out.error_log.Data(show,2)));
    metrics.max_abs_front_command_deg=rad2deg(max(abs(out.control_log.Data(show,1))));
    assert(metrics.qp_failures==0&&metrics.reached_range);
    comparison=[comparison;struct2table(metrics)];outputs{k}=out; %#ok<AGROW>
    save(fullfile(folder,[labels{k} '_raw.mat']),'out','metrics','-v7.3');
end
writetable(comparison,fullfile(folder,'metrics_120m.csv'));
selection=struct('criterion','maximum previous-10-episode mean training reward among saved checkpoints', ...
    'episode',episode,'training_mean_reward',score,'checkpoint_file',selectedFile,'model',m, ...
    'trained_with','MATLAB/Simulink MPC','metrics',table2struct(comparison(4,:)));
fid=fopen(fullfile(folder,'checkpoint_selection.json'),'w');fprintf(fid,'%s\n',jsonencode(selection,'PrettyPrint',true));fclose(fid);
save(fullfile(folder,'provenance.mat'),'agentFile','metadata','selectedFile','episode');
names={'固定 MPC','原生训练最终策略','最终策略 λ 置零（消融）',sprintf('原生训练奖励选取第%d回合',episode)};
colors=lines(4);f=figure('Visible','off','Color','w','Position',[50 50 1450 900]);tiledlayout(2,3,'TileSpacing','compact');
titles={'实际路径与参考 / m','横向误差 / m','航向误差 / °','前轮转角 / °','实际 λ','实际 q_y'};
for panel=1:6
    ax=nexttile;hold(ax,'on');
    if panel==1,plot(ax,outputs{1}.state_log.Data(:,1),outputs{1}.reference_log.Data(:,1),'k--','LineWidth',2,'DisplayName','参考轨迹');end
    for k=1:4
        s=outputs{k};x=s.state_log.Data(:,1);
        switch panel
            case 1,y=s.state_log.Data(:,2);
            case 2,y=s.error_log.Data(:,1);
            case 3,y=rad2deg(s.error_log.Data(:,2));
            case 4,y=rad2deg(s.control_log.Data(:,1));
            case 5,y=s.paper_lambda_log.Data;
            case 6,y=s.control_log.Data(:,7);
        end
        plot(ax,x,y,'Color',colors(k,:),'LineWidth',1.4,'DisplayName',names{k});
    end
    title(ax,titles{panel});xlabel(ax,'X / m');xlim(ax,[0 120]);grid(ax,'on');legend(ax,'show','Location','best','FontSize',8);
end
sgtitle('MATLAB/Simulink 原生控制器：同工况，关闭探索噪声');set(findall(f,'-property','FontName'),'FontName','宋体');
savefig(f,fullfile(folder,'comparison_120m.fig'));exportgraphics(f,fullfile(folder,'comparison_120m.png'),'Resolution',180);close(f);
f=figure('Visible','off','Color','w','Position',[50 50 1200 500]);
plot(t.episode,t.reward,'Color',[.7 .8 .9]);hold on;plot(t.episode,movmean(t.reward,[9 0]),'LineWidth',2);
xlabel('训练回合');ylabel('每回合累计奖励');title('原生 MATLAB/Simulink 环境中的实际训练');
legend('逐回合奖励','过去10回合均值','Location','best');grid on;set(findall(f,'-property','FontName'),'FontName','宋体');
savefig(f,fullfile(folder,'training.fig'));exportgraphics(f,fullfile(folder,'training.png'),'Resolution',180);close(f);
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
disp(comparison);fprintf('MATLAB_NATIVE_COMPARISON_PASS\n');
end
