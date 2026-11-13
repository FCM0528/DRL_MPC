function root = setup()
% Initialize only this migration; do not change MATLAB's global search path.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'build'),fullfile(root,'utils'),fullfile(root,'rl'),fullfile(root,'tests'));
addpath('D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\Matlab');
Simulink.fileGenControl('set','CacheFolder',fullfile(root,'build','cache'), ...
    'CodeGenFolder',fullfile(root,'build','codegen'),'createDir',true);
mpc_mex('init',fullfile(root,'configs','carsim_python_rl.ini'),fullfile(root,'carsim','case'));
assignin('base','steering_test_deg',0);
assignin('base','steering_interface_test',0);
assignin('base','controller_mode',0);
if ~evalin('base','exist(''agent'',''var'')'),assignin('base','agent',create_agent('ddpg',42));end
end
