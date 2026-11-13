function root = setup()
% Initialize only this migration; do not change MATLAB's global search path.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'build'),fullfile(root,'utils'),fullfile(root,'rl'),fullfile(root,'tests'));
addpath('D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\Matlab');
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'build','cache'), ...
    'CodeGenFolder',fullfile(root,'build','codegen'),'createDir',true);
mpc_mex('init',fullfile(root,'configs','carsim_python_rl.ini'),fullfile(root,'carsim','case'));
settings=read_mpc_settings();baseline=mpc_mex('map',zeros(4,1));
assert(isequal(baseline,settings.baseline));
assignin('base','mpc_baseline_tuning',baseline);
assignin('base','mpc_initial_control_result',[0;1;0;0;0;baseline]);
assignin('base','steering_test_deg',0);
assignin('base','steering_interface_test',0);
assignin('base','controller_mode',0);
key=sprintf('%s_min%d_max%d',settings.key,settings.min_np,settings.max_np);
same=evalin('base','exist(''mpc_configuration_key'',''var'')') && strcmp(evalin('base','mpc_configuration_key'),key);
if ~same || ~evalin('base','exist(''agent'',''var'')'),assignin('base','agent',create_agent('ddpg',42));end
assignin('base','mpc_configuration_key',key);
end
