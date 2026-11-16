function check_observer_ablation()
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
folder=fullfile(root,'results','prediction_correction');rows=table();
for k=1:3
    p=matlabMpcProfile(root,2,true);name='no_observer';p.observerEnabled=false;
    if k>=2,p.observerEnabled=true;p.observerGain=[.03;0;.03];name='low_gain_observer';end
    if k==3,p.safetyEnabled=true;p.maxRate=p.designSteerRate;name='low_gain_constrained';end
    in=Simulink.SimulationInput(m);in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
    in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','coordinated_case','simfile.sim'));
    out=sim(in);metrics=metrics_to_distance(out,name,120);
    metrics.distance_grid_rmse_m=sqrt(mean(interp1(out.state_log.Data(:,1),out.error_log.Data(:,1),0:.25:120).^2));
    save(fullfile(folder,[name '_raw.mat']),'out','metrics','p','-v7.3');rows=[rows;struct2table(metrics)];disp(struct2table(metrics)); %#ok<AGROW>
end
writetable(rows,fullfile(folder,'observer_ablation.csv'));fprintf('OBSERVER_ABLATION_COMPLETE\n');
end
