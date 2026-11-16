function open_matlab_model()
% Open the verified MATLAB/Simulink implementation of the MPC algorithm.
root=setup_matlab_mpc();file=fullfile(root,'results','matlab_native_validation','acceptance.json');
assert(isfile(file),'Native MATLAB/Simulink model validation is not complete.');
accepted=jsondecode(fileread(file));assert(accepted.passed);
m='DRL_MPC_MATLAB';open_system(fullfile(root,'models',[m '.slx']));set_param(m,'ZoomFactor','FitSystem');
end
