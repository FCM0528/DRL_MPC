function env=make_environment()
obs=rlNumericSpec([14 1]);obs.Name='road_preview_v2';
act=rlNumericSpec([4 1],'LowerLimit',-ones(4,1),'UpperLimit',ones(4,1));act.Name='adaptive_mpc';
scope=migration_id('DRL_Adaptive_Tuning');[~,path]=migration_id('RL_Agent',scope);
env=rlSimulinkEnv('DRL_MPC',path,obs,act);
env.UseFastRestart='off';
env.ResetFcn=@resetEnvironment;
end
function in=resetEnvironment(in)
mpc_mex('reset');in=in.setVariable('controller_mode',1);
end
