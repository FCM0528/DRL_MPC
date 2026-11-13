function run_extended_weight_probe()
% Diagnostic parameter-space probe, not a trained RL result or a new default.
root=setup();folder=fullfile(root,'results','drl_diagnosis');
cleanup=onCleanup(@()setup());
names={'extended_qy2','extended_qy5','extended_qy10','extended_qy5_low_r','extended_qy10_low_r','extended_qy15'};
qy=[18400 46000 92000 46000 92000 138000];rd=[13500 13500 13500 1350 1350 13500];
results=table();
for k=1:numel(names)
    ini=fullfile(folder,[names{k} '.ini']);fid=fopen(ini,'w');
    fprintf(fid,['[inherit]\nbase = carsim_python_rl.ini\n[rl]\n' ...
        'q_y_values = 4600,6000,7500,8250,9200,18400,46000,92000,138000\n' ...
        'r_delta_values = 270,1350,3375,6750,13500,15250,17000,21500,27000\n']);fclose(fid);
    mpc_mex('init',ini,fullfile(root,'carsim','case'));
    tuning=[40;qy(k);2800;rd(k)];
    in=Simulink.SimulationInput('DRL_MPC');in=in.setVariable('controller_mode',0);
    in=in.setVariable('mpc_baseline_tuning',tuning);
    in=in.setVariable('mpc_initial_control_result',[0;1;0;0;0;tuning]);out=sim(in);
    row=struct2table(metrics_to_distance(out,names{k},120));row.q_y=qy(k);row.r_delta=rd(k);row.np=40;row.nc=12;
    results=[results;row]; %#ok<AGROW>
    save(fullfile(folder,[names{k} '_raw.mat']),'out','-v7.3');
    writetable(results,fullfile(folder,'extended_weights_120m.csv'));
    fprintf('EXTENDED %s RMSE=%.9g peak=%.9g QP=%d\n',names{k},row.lateral_rmse_m,row.max_abs_lateral_m,row.qp_failures);
end
disp(results);
end
