function test_mex_pipeline(forceFailure)
if nargin<1,forceFailure=false;end
root=setup();data=readmatrix(fullfile(root,'results','adaptive_replay_states.csv'));
name='mex_pipeline_parity';expectedFile='adaptive_replay_original.csv';
if forceFailure
    mpc_mex('init',fullfile(root,'configs','forced_qp_failure.ini'),fullfile(root,'carsim','case'));
    cleanup=onCleanup(@()setup()); %#ok<NASGU>
    name='mex_pipeline_failure_parity';expectedFile='forced_qp_failure_original.csv';
end
expected=readmatrix(fullfile(root,'results',expectedFile));
settings=read_mpc_settings();actual=zeros(size(expected));mpc_mex('reset');prev=0;previous_tuning=settings.baseline;
stages={'linearization','reference','cost','constraints','qp'};
for k=1:size(data,1)
    state=data(k,1:20)';tuning=data(k,21:24)';
    if k==1,state(7)=0;else,state(7)=(data(k,3)-data(k-1,3))/.04;end
    assert(mpc_mex('stage_operatingpoint',state,tuning)==1);
    for j=1:numel(stages),assert(mpc_mex(['stage_' stages{j}],1)==1);end
    d=mpc_mex('stage_steering',1);
    actual(k,:)=[d(1:5);mpc_mex('prediction',state,d(1));mpc_mex('observe',state,tuning(1),prev);mpc_mex('reference',state(1));mpc_mex('reward',state,d(1),prev,tuning,previous_tuning)]';
    prev=d(1);previous_tuning=tuning;
end
error=max(abs(actual-expected),[],1);
assert(error(1)<1e-10 && all(error(2:3)==0) && all(error(4:5)<1e-7));
assert(all(error(6:13)<1e-9)&&all(error(14:27)<1e-9)&&all(error(28:34)<1e-12)&&error(35)<1e-8);
if forceFailure,assert(nnz(actual(:,2)==0)>0 && all(isfinite(actual),'all'));end
save(fullfile(root,'results',[name '.mat']),'expected','actual','error');
summary=[error(1:5),max(error(6:13)),max(error(14:27)),max(error(28:34)),error(35)];
writetable(array2table(summary,'VariableNames',{'steer_rad','converged','iterations','primal','dual','predicted_state','observation','reference','reward'}),fullfile(root,'results',[name '.csv']));
disp(summary);
end
