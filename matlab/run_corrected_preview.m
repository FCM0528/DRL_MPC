function out=run_corrected_preview(profile,coordinated)
% Run the corrected fixed-weight MPC; old TD3 agents belong to the old model.
arguments
    profile (1,1) double {mustBeMember(profile,0:3)}=3
    coordinated (1,1) logical=false
end
assert(~coordinated,'NominalSpeed:NoSpeedPlanning','当前入口保持原70 km/h工况；已撤回大幅降速规划。');
root=setup_matlab_mpc();m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));
p=matlabMpcProfile(root,profile,coordinated);in=Simulink.SimulationInput(m);
in=in.setVariable('matlab_mpc_parameters',p);in=in.setVariable('controller_mode',0);
caseName='case';
in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim',caseName,'simfile.sim'));
out=sim(in);metrics=metrics_to_distance(out,'corrected_preview',120);
assert(metrics.qp_failures==0&&metrics.reached_range,'Preview acceptance checks failed. Inspect QP_Status.');
[~,mexfiles]=inmem('-completenames');assert(~any(contains(mexfiles,{'mpc_paper_mex','mpc_mex'})));
assignin('base','matlab_corrected_out',out);disp(struct2table(metrics));
end
