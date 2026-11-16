function check_fixed_speed_design()
% Assess physical-capacity bounds and rate regularization at unchanged speed.
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));folder=fullfile(root,'results','nominal_speed_correction');
profiles=[3 1 1 3 3];rateScale=[1 .5 .25 .5 .25];names={'physical_capacity','rate_half','rate_quarter','physical_half','physical_quarter'};rows=table();
for k=1:numel(profiles)
    p=matlabMpcProfile(root,profiles(k),false);p.rateChange=p.rateChange*rateScale(k);
    % The old 0.95 factor was an added design margin, not a measured limit.
    % Here use the same measured-table/mu capacity without adding that margin.
    if profiles(k)==3,p.safetyUtilization=1;end
    in=Simulink.SimulationInput(m);in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));
    out=sim(in);s=out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=out.lateral_safety_log.Data(mask,:);metrics=metrics_to_distance(out,names{k},120);
    metrics.min_speed_kmh=min(s(mask,3))*3.6;metrics.max_speed_kmh=max(s(mask,3))*3.6;
    metrics.max_ay_mps2=max(abs(a(:,1)));metrics.max_ay_excess=max(0,max(abs(a(:,1))-a(:,2)));
    metrics.max_beta_deg=max(abs(a(:,3)))*180/pi;metrics.max_slack=max(a(:,8));
    metrics.max_steer_rate_deg_s=max(abs(diff(out.control_log.Data(mask,1))))/.04*180/pi;
    metrics.rate_change_weight=p.rateChange;metrics.safety_utilization=p.safetyUtilization;
    save(fullfile(folder,[names{k} '_raw.mat']),'out','metrics','p','-v7.3');rows=[rows;struct2table(metrics)];disp(struct2table(metrics)); %#ok<AGROW>
end
writetable(rows,fullfile(folder,'fixed_speed_design.csv'));fprintf('FIXED_SPEED_DESIGN_COMPLETE\n');
end
