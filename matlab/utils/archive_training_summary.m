function archive_training_summary(pattern)
root=fileparts(fileparts(mfilename('fullpath')));
if nargin<1
    settings=read_mpc_settings();c=read_training_config();
    pattern=sprintf('training_*_%s_ep%d_seed%d',settings.key,c.training.episodes,c.training.seed);
end
folders=dir(fullfile(root,'results',pattern));
for entry=folders'
        if ~entry.isdir,continue;end
        algorithm=regexp(entry.name,'(?:smoke|training)_(ddpg|td3)_','tokens','once');if isempty(algorithm),continue;end
        folder=fullfile(entry.folder,entry.name);
        file=fullfile(folder,'final_agent.mat');if ~isfile(file),continue;end
        loaded=load(file,'stats','parameter_change','seed','episodes','agent');
        T=readtable(fullfile(folder,'training.csv'));
        summary=struct('algorithm',algorithm{1},'seed',loaded.seed, ...
            'configured_episodes',loaded.episodes,'completed_episodes',height(T), ...
            'interactions',sum(T.steps),'actor_max_parameter_change',loaded.parameter_change, ...
            'sample_time_s',loaded.agent.SampleTime,'learning_warmup_interactions',550);
        fid=fopen(fullfile(folder,'summary.json'),'w');fprintf(fid,'%s',jsonencode(summary,'PrettyPrint',true));fclose(fid);
        fid=fopen(fullfile(folder,'training.log'),'w');
        fprintf(fid,'%s seed=%d episodes=%d interactions=%d\n',algorithm{1},loaded.seed,height(T),sum(T.steps));
        for k=1:height(T),fprintf(fid,'Episode %d reward=%.17g steps=%d\n',T.episode(k),T.reward(k),T.steps(k));end
        fprintf(fid,'Actor maximum parameter change %.17g\n',loaded.parameter_change);fclose(fid);
end
end
