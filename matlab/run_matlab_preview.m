function out=run_matlab_preview(controller)
% Run the native MATLAB MPC with fixed, final or reward-selected tuning.
arguments
    controller (1,:) char {mustBeMember(controller,{'fixed','final','selected'})}='fixed'
end
root=setup_matlab_mpc();open_matlab_model();m='DRL_MPC_MATLAB';mode=0;agentFile='';
if ~strcmp(controller,'fixed')
    mode=1;
    if strcmp(controller,'selected')
        selection=jsondecode(fileread(fullfile(root,'results','matlab_comparison','checkpoint_selection.json')));agentFile=selection.checkpoint_file;
    else,agentFile=fullfile(root,'results','matlab_td3_ep300_seed42','final_agent.mat');end
    agent=load_paper_agent(agentFile);assignin('base','agent',agent);
end
[~,beforeMex]=inmem('-completenames');
in=Simulink.SimulationInput(m);in=in.setVariable('controller_mode',mode);
in=in.setVariable('matlab_mpc_parameters',matlabMpcProfile(root,0,false));
in=in.setBlockParameter(Simulink.ID.getFullName([m ':1']),'SIMFILE',fullfile(root,'carsim','case','simfile.sim'));out=sim(in);
metrics=metrics_to_distance(out,['matlab_' controller],120);assert(metrics.qp_failures==0&&metrics.reached_range);
[~,mexfiles]=inmem('-completenames');newMex=setdiff(mexfiles,beforeMex);
assert(~any(contains(newMex,{'mpc_paper_mex','mpc_mex'})),'The preview unexpectedly loaded a legacy controller MEX.');
folder=fullfile(root,'results','matlab_preview');if ~isfolder(folder),mkdir(folder);end
save(fullfile(folder,[controller '_raw.mat']),'out','metrics','agentFile','-v7.3');
writetable(struct2table(metrics),fullfile(folder,[controller '_metrics_120m.csv']));
assignin('base','matlab_preview_out',out);disp(struct2table(metrics));
end
