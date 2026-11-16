function run_nominal_speed_acceptance()
root=setup_matlab_mpc(3,false);m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
out=run_corrected_preview();p=matlabMpcProfile(root,3,false);metrics=metrics_to_distance(out,'current70',120);
s=out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=out.lateral_safety_log.Data(mask,:);
metrics.min_speed_kmh=min(s(mask,3))*3.6;metrics.max_speed_kmh=max(s(mask,3))*3.6;
metrics.distance_grid_rmse_m=sqrt(mean(interp1(s(:,1),out.error_log.Data(:,1),0:.25:120).^2));
metrics.max_ay_mps2=max(abs(a(:,1)));metrics.max_ay_excess_mps2=max(0,max(abs(a(:,1))-a(:,2)));
metrics.max_beta_deg=max(abs(a(:,3)))*180/pi;metrics.max_slack=max(a(:,8));
metrics.max_steer_rate_deg_s=max(abs(diff(out.control_log.Data(mask,1))))/.04*180/pi;
baseline=readtable(fullfile(root,'results','nominal_speed_correction','original70_metrics.csv'));
metrics.rmse_change_vs_baseline_pct=100*(1-metrics.lateral_rmse_m/baseline.lateral_rmse_m);
assert(metrics.reached_range&&metrics.qp_failures==0&&metrics.min_speed_kmh>68&&metrics.max_speed_kmh<72);
assert(metrics.lateral_rmse_m<baseline.lateral_rmse_m&&metrics.max_abs_lateral_m<baseline.max_abs_lateral_m);
save(fullfile(root,'results','nominal_speed_correction','current70_raw.mat'),'out','metrics','p','-v7.3');
writetable(struct2table(metrics),fullfile(root,'results','nominal_speed_correction','current70_metrics.csv'));disp(struct2table(metrics));
fprintf('NOMINAL_SPEED_ACCEPTANCE_PASS\n');
end
