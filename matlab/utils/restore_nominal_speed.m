function restore_nominal_speed()
% Restore the user's original speed scenario and compare at that speed only.
root=setup_matlab_mpc();folder=fullfile(root,'results','nominal_speed_correction');if ~isfolder(folder),mkdir(folder);end
m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
set_param(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));
set_param(m,'PostLoadFcn',sprintf('addpath(''%s''); setup_matlab_mpc;',root));save_system(m);
fprintf('ORIGINAL_70_KMH_DEFAULT_RESTORED\n');
profiles=[0 1 3];names={'original70','corrected70','constrained70'};rows=table();
for k=1:3
    p=matlabMpcProfile(root,profiles(k),false);in=Simulink.SimulationInput(m);
    in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));
    tic;out=sim(in);elapsed=toc;s=out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=out.lateral_safety_log.Data(mask,:);
    metrics=metrics_to_distance(out,names{k},120);metrics.elapsed_s=elapsed;
    metrics.min_speed_kmh=min(s(mask,3))*3.6;metrics.max_speed_kmh=max(s(mask,3))*3.6;
    metrics.distance_grid_rmse_m=sqrt(mean(interp1(s(:,1),out.error_log.Data(:,1),0:.25:120).^2));
    metrics.max_abs_ay_mps2=max(abs(a(:,1)));metrics.max_ay_excess_mps2=max(0,max(abs(a(:,1))-a(:,2)));
    metrics.max_beta_deg=max(abs(a(:,3)))*180/pi;metrics.max_slack=max(a(:,8));
    assert(metrics.reached_range&&metrics.min_speed_kmh>68&&metrics.max_speed_kmh<72,'Nominal speed band or distance check failed.');
    if k==1
        v=load(fullfile(root,'results','matlab_comparison','matched_fixed_raw.mat'),'out');
        assert(max(abs(out.state_log.Data-v.out.state_log.Data),[],'all')<1e-6);
    end
    save(fullfile(folder,[names{k} '_raw.mat']),'out','metrics','p','-v7.3');writetable(struct2table(metrics),fullfile(folder,[names{k} '_metrics.csv']));
    rows=[rows;struct2table(metrics)];disp(struct2table(metrics));fprintf('NOMINAL_SPEED_CASE_COMPLETE %s\n',names{k}); %#ok<AGROW>
end
writetable(rows,fullfile(folder,'comparison_120m.csv'));fprintf('NOMINAL_SPEED_COMPARISON_COMPLETE\n');
end
