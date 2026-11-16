function validate_matlab_closed_loop()
% Regression using current native agents and saved native simulation outputs.
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
folder=fullfile(root,'results','matlab_native_validation');comparison=fullfile(root,'results','matlab_comparison');
agent=load_paper_agent(fullfile(root,'results','matlab_td3_ep300_seed42','final_agent.mat'));
selection=jsondecode(fileread(fullfile(comparison,'checkpoint_selection.json')));
labels={'matched_fixed','native_td3_final','same_final_policy_lambda_zero','native_td3_reward_selected'};
rows=table();differences=zeros(4,3);
for k=1:4
    mode=k-1;if k==4,agent=load_paper_agent(selection.checkpoint_file);mode=1;end
    assignin('base','agent',agent);in=Simulink.SimulationInput(m);in=in.setVariable('controller_mode',mode);
    in=in.setVariable('matlab_mpc_parameters',matlabMpcProfile(root,0,false));
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));out=sim(in);
    v=load(fullfile(comparison,[labels{k} '_raw.mat']),'out');previous=v.out;
    assert(isequal(size(out.state_log.Data),size(previous.state_log.Data)));
    differences(k,:)=[max(abs(out.state_log.Data-previous.state_log.Data),[],'all'), ...
        max(abs(out.control_log.Data(:,1)-previous.control_log.Data(:,1))), ...
        max(abs(out.reward_log.Data-previous.reward_log.Data),[],'all')];
    metrics=metrics_to_distance(out,labels{k},120);assert(metrics.qp_failures==0&&metrics.reached_range);
    rows=[rows;struct2table(metrics)]; %#ok<AGROW>
end
assert(all(differences(:,1)<1e-6)&&all(differences(:,2)<1e-8)&&all(differences(:,3)<.001));
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
writetable(rows,fullfile(folder,'regression_120m.csv'));
data=struct('passed',true,'reference','saved native controller comparison runs','model',m, ...
    'state_steer_reward_max_difference',differences,'controller_mex_loaded',false,'range_start_m',0,'range_end_m',120);
fid=fopen(fullfile(folder,'regression_after_cleanup.json'),'w');fprintf(fid,'%s\n',jsonencode(data,'PrettyPrint',true));fclose(fid);
disp(rows);fprintf('MATLAB_NATIVE_CLOSED_LOOP_REGRESSION_PASS\n');
end
