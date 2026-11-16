function run_correction_experiments(indices)
% CarSim closed-loop ablations, all distance metrics restricted to [0,120].
if nargin==0,indices=1:6;end
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
folder=fullfile(root,'results','prediction_correction');
sldiagviewer.diary(fullfile(folder,'simulation_diagnostics.txt'));
names={'original_70','corrected_70','observer_70','original_planned','corrected_planned','constrained_planned'};
profiles=[0 1 2 0 1 3];planned=[false false false true true true];rows=table();
for k=indices
    p=matlabMpcProfile(root,profiles(k),planned(k));in=Simulink.SimulationInput(m);
    in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
    caseName='case';if planned(k),caseName='coordinated_case';end
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim',caseName,'simfile.sim'));
    tic;out=sim(in);elapsed=toc;metrics=metrics_to_distance(out,names{k},120);
    s=out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=out.lateral_safety_log.Data(mask,:);
    distanceErrors=interp1(s(:,1),out.error_log.Data(:,1),0:.25:120,'linear');
    metrics.distance_grid_rmse_m=sqrt(mean(distanceErrors.^2));
    metrics.elapsed_s=elapsed;metrics.min_speed_kmh=min(s(mask,3))*3.6;metrics.max_speed_kmh=max(s(mask,3))*3.6;
    metrics.max_abs_ay_mps2=max(abs(a(:,1)));metrics.max_ay_excess_mps2=max(0,max(abs(a(:,1))-a(:,2)));
    metrics.max_abs_beta_deg=max(abs(a(:,3)))*180/pi;metrics.max_slack=max(a(:,8));
    metrics.max_reference_excess=max(a(:,9));metrics.max_steer_rate_deg_s=max(abs(diff(out.control_log.Data(mask,1))))/.04*180/pi;
    [~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
    if k==1
        v=load(fullfile(root,'results','matlab_comparison','matched_fixed_raw.mat'),'out');
        difference=max(abs(out.state_log.Data-v.out.state_log.Data),[],'all');assert(difference<1e-6);
        fprintf('BASELINE_STATE_REGRESSION_MAX %.12g\n',difference);
    end
    save(fullfile(folder,[names{k} '_raw.mat']),'out','metrics','p','-v7.3');
    writetable(struct2table(metrics),fullfile(folder,[names{k} '_metrics.csv']));disp(struct2table(metrics));
    assert(metrics.reached_range,'Vehicle did not reach 120 m.');
    rows=[rows;struct2table(metrics)]; %#ok<AGROW>
    fprintf('CORRECTION_CASE_COMPLETE %s\n',names{k});
end
writetable(rows,fullfile(folder,'latest_experiment_metrics.csv'));
fprintf('CORRECTION_EXPERIMENTS_COMPLETE\n');
end
