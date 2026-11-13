function comparison=compare_controllers(ddpgFile,td3File)
root=setup();open_system(fullfile(root,'models','DRL_MPC.slx'));
settings=read_mpc_settings();c=read_training_config();
if nargin<1,ddpgFile=fullfile(root,'results',sprintf('training_ddpg_%s_ep%d_seed%d',settings.key,c.training.episodes,c.training.seed),'final_agent.mat');end
if nargin<2,td3File=fullfile(root,'results',sprintf('training_td3_%s_ep%d_seed%d',settings.key,c.training.episodes,c.training.seed),'final_agent.mat');end
files={'',ddpgFile,td3File};
for k=2:3
    assert(isfile(files{k}),'Current configuration requires a newly trained Agent: %s',files{k});
    meta=load(files{k},'model_settings');
    assert(isfield(meta,'model_settings') && isequal(meta.model_settings,settings),'Agent configuration does not match the current MPC.');
end
labels={settings.fixed_label,['ddpg_' settings.key],['td3_' settings.key]};comparison=table();outputs=cell(1,3);
for k=1:3
    if k>1,loaded=load(files{k},'agent');agent=loaded.agent;agent.UseExplorationPolicy=false;assignin('base','agent',agent);end
    in=Simulink.SimulationInput('DRL_MPC');in=in.setVariable('controller_mode',double(k>1));in=in.setVariable('steering_interface_test',0);
    out=sim(in);metrics=export_run(out,labels{k});comparison=[comparison;struct2table(metrics)];outputs{k}=out;
end
folder=fullfile(root,'results',['comparison_' settings.key]);if ~isfolder(folder),mkdir(folder);end
writetable(comparison,fullfile(folder,'metrics.csv'));save(fullfile(folder,'comparison.mat'),'comparison','files');
f=figure('Visible','off','Position',[100 100 1200 800]);tiledlayout(3,2);
for p=1:6
    nexttile;hold on;
    for k=1:3
        out=outputs{k};show=out.state_log.Data(:,1)<=plot_station_limit();
        t=out.state_log.Time(show);s=out.state_log.Data(show,:);d=out.control_log.Data(show,:);e=out.error_log.Data(show,:);
        switch p
            case 1,plot(s(:,1),s(:,2));xlabel('X (m)');ylabel('Y (m)');
            case 2,plot(s(:,1),e(:,1));xlabel('X (m)');ylabel('Lateral error (m)');
            case 3,plot(t,rad2deg(e(:,2)));ylabel('Heading error (deg)');
            case 4,plot(t,rad2deg(d(:,1)));ylabel('Steering command (deg)');
            case 5,stairs(t,d(:,6));ylabel('Np');
            case 6,plot(t,rad2deg(s(:,11)));ylabel('Sideslip (deg)');
        end
    end
    if p==1,show=outputs{1}.state_log.Data(:,1)<=plot_station_limit();plot(outputs{1}.state_log.Data(show,1),outputs{1}.reference_log.Data(show,1),'k--');legend([labels,{'DLC SYX'}]);else,legend(labels);end
    if p<=2,xlim([0 plot_station_limit()]);end
    grid on;
end
savefig(f,fullfile(folder,'comparison.fig'));exportgraphics(f,fullfile(folder,'comparison.png'),'Resolution',180);close(f);disp(comparison);
end
