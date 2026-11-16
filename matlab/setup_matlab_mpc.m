function root=setup_matlab_mpc(profile,coordinated)
% Initialize the MATLAB/Simulink controller. Does not load a controller MEX.
arguments
    profile (1,1) double {mustBeMember(profile,0:3)}=3
    coordinated (1,1) logical=false
end
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'utils'),fullfile(root,'rl'),fullfile(root,'controllers','matlab'));
addpath('D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\Matlab');
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'build','matlab_cache'), ...
    'CodeGenFolder',fullfile(root,'build','matlab_codegen'),'createDir',true);
parameters=matlabMpcProfile(root,profile,coordinated);assignin('base','matlab_mpc_parameters',parameters);
assignin('base','mpc_baseline_tuning',[40;9200;2800;13500]);
assignin('base','mpc_initial_control_result',[0;1;0;0;0;40;9200;2800;13500]);
assignin('base','controller_mode',0);assignin('base','steering_interface_test',0);assignin('base','steering_test_deg',0);
assignin('base','agent',create_paper_agent());
end
