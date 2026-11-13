function metrics=run_fixed_mpc()
root=setup();open_system(fullfile(root,'models','DRL_MPC.slx'));
in=Simulink.SimulationInput('DRL_MPC');
in=in.setVariable('controller_mode',0);
out=sim(in);
settings=read_mpc_settings();metrics=export_run(out,settings.fixed_label);
assert(metrics.qp_failures==0,'Fixed MPC contains failed QPs');
assert(metrics.completed,'CarSim did not reach the common evaluation endpoint');
end
