function diagnose_drl()
% Read-only audit of existing agents, returns, observations and exact rewards.
root=setup();folder=fullfile(root,'results','drl_diagnosis');if ~isfolder(folder),mkdir(folder);end
names={'fixed_np40_nc12','ddpg_np40_nc12','td3_np40_nc12'};
allTerms=table();allMetrics=table();
for k=1:3
    data=load(fullfile(root,'results',names{k},'raw.mat'),'out');out=data.out;
    s=out.state_log.Data;d=out.control_log.Data;t=out.state_log.Time;
    % Reward_Termination receives Previous_Control_Result (one MPC delay).
    % Use the command that actually drove the transition, not the new command.
    settings=read_mpc_settings();d=[[0 1 0 0 0 settings.baseline'];d(1:end-1,:)];
    info=mpc_mex('info');limit=info(1);terms=zeros(numel(t),12);exact=zeros(numel(t),1);
    previousSteer=0;previousTuning=read_mpc_settings().baseline;
    for i=1:numel(t)
        ref=mpc_mex('reference',s(i,1));ey=s(i,2)-ref(1);
        beta=atan2(s(i,4),s(i,3));course=atan2(sin(s(i,5)+beta-ref(2)),cos(s(i,5)+beta-ref(2)));
        xdot=s(i,3)*cos(s(i,5))-s(i,4)*sin(s(i,5));
        ydot=s(i,3)*sin(s(i,5))+s(i,4)*cos(s(i,5));
        growth=max(0,abs(ey+.25*(ydot-ref(4)*xdot))-abs(ey))/.2;
        rate=abs(d(i,1)-previousSteer)/(deg2rad(100)*.04);bounded=min(rate,2);
        old=previousTuning(2:4)';current=d(i,7:9);
        spans=.5*log([13800/4600 5600/1400 27000/6750]);
        change=max(-2,min(2,log(current./old)./spans));
        heading=max(-20,min(20,course/.1));
        terms(i,:)=[2*(ey/.2)^2,3*(max(0,abs(ey)-.25)/.25)^2, ...
            .1*(sqrt(1+heading^2)-1)/(sqrt(2)-1),.6*growth^2, ...
            .03*max(-1,min(1,d(i,1)/limit))^2, ...
            .05*(bounded^2+4*(rate-bounded)), ...
            .2*max(0,abs(beta)/deg2rad(5)-1)^2, ...
            .002*min(1,abs(d(i,6)-previousTuning(1))/38)^2,0,0,.001*mean(change.^2),0];
        exact(i)=-mpc_mex('reward',s(i,:)',d(i,1),previousSteer,d(i,6:9)',previousTuning)/10;
        previousSteer=d(i,1);previousTuning=d(i,6:9)';
    end
    assert(max(abs(sum(terms,2)-exact))<1e-9,'Reward decomposition does not match the native reward');
    inRange=s(:,1)<=120 & t>0;totals=sum(terms(inRange,:),1)/2;
    termNames={'lateral','peak','course','growth','steer','steer_rate','sideslip','np_change','vy','yaw_error','weight_change','weight_deviation'};
    row=cell2table([names(k),num2cell(totals),{sum(totals)}], ...
        'VariableNames',[{'controller'},termNames,{'total_loss_120m'}]);
    allTerms=[allTerms;row]; %#ok<AGROW>
    allMetrics=[allMetrics;struct2table(metrics_to_distance(out,names{k},120))]; %#ok<AGROW>
    writematrix([t s(:,1) terms],fullfile(folder,[names{k} '_reward_terms.csv']));
    if k>1
        stored=sum(out.reward_log.Data(3:2:end));reconstructed=-sum(exact(2:end))/2;
        fprintf('%s reward sum: stored=%.9g reconstructed=%.9g diff=%.3g\n',names{k},stored,reconstructed,stored-reconstructed);
        assert(abs(stored-reconstructed)<1e-8,'Reward transition alignment mismatch');
        pairs=reshape(exact(2:end),2,[]);
        writematrix([out.reward_log.Time(3:2:end),out.reward_log.Data(3:2:end),-mean(pairs,1)'], ...
            fullfile(folder,[names{k} '_reward_alignment.csv']));
    end
end
writetable(allTerms,fullfile(folder,'reward_decomposition_120m.csv'));
writetable(allMetrics,fullfile(folder,'metrics_120m.csv'));
for algorithm={'ddpg','td3'}
    file=fullfile(root,'results',sprintf('training_%s_np40_nc12_ep300_seed42',algorithm{1}),'final_agent.mat');
    trained=load(file,'agent','stats');stats=trained.stats;
    T=table(stats.EpisodeIndex(:),stats.EpisodeReward(:),stats.EvaluationStatistic(:),stats.EpisodeQ0(:), ...
        'VariableNames',{'episode','training_return','evaluation_return','Q0'});
    writetable(T,fullfile(folder,[algorithm{1} '_learning_audit.csv']));
    critics=getCritic(trained.agent);
    if numel(critics)==2
        p1=getLearnableParameters(critics(1));p2=getLearnableParameters(critics(2));difference=0;
        for i=1:numel(p1),difference=max(difference,max(abs(double(gather(p1{i}))-double(gather(p2{i}))),[],'all'));end
        fprintf('TD3 critic maximum difference: %.9g\n',difference);
    end
end
disp(allTerms);disp(allMetrics);
end
