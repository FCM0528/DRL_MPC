function check_nominal_feedforward()
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));folder=fullfile(root,'results','nominal_speed_correction');rows=table();
for k=1:3
    p=matlabMpcProfile(root,3,false);p.safetyUtilization=1;p.rateChange=960000/(2^(k-1));p.consistentFeedforward=true;p.maxIterations=20000;
    in=Simulink.SimulationInput(m);in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));
    out=sim(in);s=out.state_log.Data;mask=s(:,1)>=0&s(:,1)<=120;a=out.lateral_safety_log.Data(mask,:);name=sprintf('consistent_ff_%d',k);
    metrics=metrics_to_distance(out,name,120);metrics.min_speed_kmh=min(s(mask,3))*3.6;metrics.max_speed_kmh=max(s(mask,3))*3.6;
    metrics.max_ay_mps2=max(abs(a(:,1)));metrics.max_ay_excess=max(0,max(abs(a(:,1))-a(:,2)));
    metrics.max_beta_deg=max(abs(a(:,3)))*180/pi;metrics.max_slack=max(a(:,8));metrics.rate_change_weight=p.rateChange;
    metrics.max_steer_rate_deg_s=max(abs(diff(out.control_log.Data(mask,1))))/.04*180/pi;
    save(fullfile(folder,[name '_raw.mat']),'out','metrics','p','-v7.3');rows=[rows;struct2table(metrics)];disp(struct2table(metrics)); %#ok<AGROW>
end
writetable(rows,fullfile(folder,'consistent_feedforward.csv'));fprintf('CONSISTENT_FEEDFORWARD_COMPLETE\n');
end
