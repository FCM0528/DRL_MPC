function run_drl_ablations()
% Controlled real CarSim experiments; do not modify the nominal INI/model.
root=setup();open_system(fullfile(root,'models','DRL_MPC.slx'));
folder=fullfile(root,'results','drl_diagnosis');if ~isfolder(folder),mkdir(folder);end
cleanup=onCleanup(@()setup());
copyfile(fullfile(root,'configs','default.ini'),fullfile(folder,'default.ini'));
copyfile(fullfile(root,'configs','carsim_python_rl.ini'),fullfile(folder,'carsim_python_rl.ini'));
names={'nominal','tracking_corner','high_y_high_psi','high_y_high_r','np12','np32','np50', ...
    'rate_change_half','rate_change_tenth','rate_change_zero','feedforward_penalty_zero','rate_tenth_and_tracking_corner'};
tunings=[40 9200 2800 13500;40 13800 1400 6750;40 13800 5600 6750;40 13800 2800 27000; ...
    12 9200 2800 13500;32 9200 2800 13500;50 9200 2800 13500; ...
    40 9200 2800 13500;40 9200 2800 13500;40 9200 2800 13500;40 9200 2800 13500;40 13800 1400 6750];
rateChanges=[960000 960000 960000 960000 960000 960000 960000 480000 96000 0 960000 96000];
steerPenalties=33000*ones(1,numel(names));steerPenalties(11)=0;
results=table();
for k=1:numel(names)
    ini=fullfile(folder,[names{k} '.ini']);fid=fopen(ini,'w');
    fprintf(fid,'[inherit]\nbase = %s\n[weights]\nr_delta_change = %.17g\nr_steer = %.17g\n', ...
        'carsim_python_rl.ini',rateChanges(k),steerPenalties(k));fclose(fid);
    mpc_mex('init',ini,fullfile(root,'carsim','case'));
    in=Simulink.SimulationInput('DRL_MPC');in=in.setVariable('controller_mode',0);
    in=in.setVariable('mpc_baseline_tuning',tunings(k,:)');
    in=in.setVariable('mpc_initial_control_result',[0;1;0;0;0;tunings(k,:)']);
    out=sim(in);
    row=struct2table(metrics_to_distance(out,names{k},120));
    row.np=tunings(k,1);row.nc=12;row.q_y=tunings(k,2);row.q_psi=tunings(k,3);row.r_delta=tunings(k,4);
    row.r_delta_change=rateChanges(k);row.r_steer=steerPenalties(k);
    results=[results;row]; %#ok<AGROW>
    save(fullfile(folder,[names{k} '_raw.mat']),'out','-v7.3');
    writetable(results,fullfile(folder,'ablations_120m.csv'));
    fprintf('ABLATION %d/%d %s RMSE=%.9g peak=%.9g QP=%d\n',k,numel(names),names{k},row.lateral_rmse_m,row.max_abs_lateral_m,row.qp_failures);
end
disp(results);
end
