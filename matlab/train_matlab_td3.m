function train_matlab_td3(episodes,seed)
% Train a new TD3 agent against the MATLAB/Simulink MPC implementation.
arguments
    episodes (1,1) double {mustBeInteger,mustBePositive}=300
    seed (1,1) double {mustBeInteger,mustBeNonnegative}=42
end
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
accepted=jsondecode(fileread(fullfile(root,'results','nominal_speed_correction','acceptance.json')));assert(accepted.passed);
folder=fullfile(root,'results',sprintf('matlab_td3_corrected70_ep%d_seed%d',episodes,seed));
assert(~isfile(fullfile(folder,'final_agent.mat')),'Refusing to overwrite an existing training result.');
if ~isfolder(folder),mkdir(folder);end
diary(fullfile(folder,'training.log'));cleanup=onCleanup(@()diary('off'));
agent=create_paper_agent(seed);assignin('base','agent',agent);
obs=rlNumericSpec([6 1]);act=rlNumericSpec([3 1],'LowerLimit',-ones(3,1),'UpperLimit',ones(3,1));
env=rlSimulinkEnv(m,Simulink.ID.getFullName([m ':124']),obs,act);env.UseFastRestart='off';env.ResetFcn=@reset;
validateEnvironment(env);before=getLearnableParameters(getActor(agent));
options=rlTrainingOptions('MaxEpisodes',episodes,'MaxStepsPerEpisode',500,'Plots','none', ...
    'StopTrainingCriteria','EpisodeCount','StopTrainingValue',episodes,'Verbose',true, ...
    'SaveAgentCriteria','EpisodeFrequency','SaveAgentValue',10,'SaveAgentDirectory',fullfile(folder,'checkpoints'));
stats=train(agent,env,options);after=getLearnableParameters(getActor(agent));change=0;
for k=1:numel(before),change=max(change,max(abs(double(gather(after{k}))-double(gather(before{k}))),[],'all'));end
settings=paper_settings();settings.end_distance_m=120;settings.model=m;settings.implementation='MATLAB/Simulink';
settings.prediction_revision='prediction_correction_nominal70_v1';settings.speed_coordination=false;
save(fullfile(folder,'final_agent.mat'),'agent','stats','settings','change','-v7.3');
writetable(table(stats.EpisodeIndex(:),stats.EpisodeReward(:),stats.EpisodeSteps(:), ...
    'VariableNames',{'episode','reward','steps'}),fullfile(folder,'training.csv'));
assert(numel(stats.EpisodeIndex)==episodes&&change>0);
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
summary=struct('episodes',episodes,'interactions',sum(stats.EpisodeSteps),'actor_max_abs_change',change, ...
    'seed',seed,'model',m,'implementation','MATLAB/Simulink','controller_mex_loaded',false,'np',40,'nc',12, ...
    'prediction_revision','prediction_correction_nominal70_v1','coordinated_case',false);
fid=fopen(fullfile(folder,'summary.json'),'w');fprintf(fid,'%s\n',jsonencode(summary,'PrettyPrint',true));fclose(fid);
accepted.native_training_completed=true;accepted.training_episodes=episodes;accepted.training_interactions=summary.interactions;
accepted.corrected_environment_training_episodes=episodes;
fid=fopen(fullfile(root,'results','nominal_speed_correction','acceptance.json'),'w');fprintf(fid,'%s\n',jsonencode(accepted,'PrettyPrint',true));fclose(fid);
fprintf('MATLAB_NATIVE_TRAINING_PASS: %d episodes, %d interactions, actor change %.9g\n',episodes,sum(stats.EpisodeSteps),change);
compare_matlab_controllers(fullfile(folder,'final_agent.mat'),true);
    function in=reset(in)
        in=in.setVariable('controller_mode',1);in=in.setVariable('matlab_mpc_parameters',matlabMpcProfile(root,3,false));
        in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));
    end
end
