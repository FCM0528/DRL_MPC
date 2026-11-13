function [agent,stats]=train_controller(algorithm,episodes,seeds,label)
root=setup();c=read_training_config();
model_settings=read_mpc_settings();
if nargin<2||isempty(episodes),episodes=c.training.episodes;end
if nargin<3||isempty(seeds),seeds=c.training.seed;end
if nargin<4,label='training';end
open_system(fullfile(root,'models','DRL_MPC.slx'));
for seed=seeds
    folder=fullfile(root,'results',sprintf('%s_%s_%s_ep%d_seed%d',label,algorithm,model_settings.key,episodes,seed));if ~isfolder(folder),mkdir(folder);end
    diary(fullfile(folder,'training.log'));cleanup=onCleanup(@()diary('off'));
    device='cpu';if gpuDeviceCount('available')>0,device='gpu';end
    agent=create_agent(algorithm,seed,device);assignin('base','agent',agent);assignin('base','controller_mode',1);
    env=make_environment();validateEnvironment(env);
    before=getLearnableParameters(getActor(agent));
    options=rlTrainingOptions('MaxEpisodes',episodes,'MaxStepsPerEpisode',250, ...
        'StopTrainingCriteria','EpisodeCount','StopTrainingValue',episodes, ...
        'Plots','none','Verbose',true,'UseParallel',false, ...
        'SaveAgentCriteria','EpisodeFrequency','SaveAgentValue',max(1,min(10,episodes)), ...
        'SaveAgentDirectory',fullfile(folder,'checkpoints'));
    evaluator=rlEvaluator('EvaluationFrequency',6,'NumEpisodes',1);
    stats=train(agent,env,options,'Evaluator',evaluator);
    agent.UseExplorationPolicy=false;
    after=getLearnableParameters(getActor(agent));parameter_change=0;
    for k=1:numel(before),parameter_change=max(parameter_change,max(abs(gather(double(before{k}))-gather(double(after{k}))),[],'all'));end
    save(fullfile(folder,'final_agent.mat'),'agent','stats','parameter_change','c','seed','episodes','model_settings','-v7.3');
    T=table(stats.EpisodeIndex(:),stats.EpisodeReward(:),stats.EpisodeSteps(:),'VariableNames',{'episode','reward','steps'});
    writetable(T,fullfile(folder,'training.csv'));
    f=figure('Visible','off');plot(T.episode,T.reward);xlabel('Episode');ylabel('Return');grid on;savefig(f,fullfile(folder,'training.fig'));exportgraphics(f,fullfile(folder,'training.png'),'Resolution',160);close(f);
    fprintf('Completed %s: %d episodes; actor max parameter change %.8g\n',algorithm,episodes,parameter_change);
    assignin('base','agent',agent);in=Simulink.SimulationInput('DRL_MPC');in=in.setVariable('controller_mode',1);out=sim(in);export_run(out,sprintf('%s_%s_%s_ep%d_seed%d_evaluation',label,algorithm,model_settings.key,episodes,seed));
    clear cleanup
end
end
