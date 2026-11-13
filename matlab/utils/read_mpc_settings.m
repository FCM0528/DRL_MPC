function settings=read_mpc_settings()
% One source for model constants, experiment folders and validation baselines.
root=fileparts(fileparts(mfilename('fullpath')));
base=fileread(fullfile(root,'configs','default.ini'));
rl=fileread(fullfile(root,'configs','carsim_python_rl.ini'));
settings.np=number(base,'np');settings.nc=number(base,'nc');
settings.min_np=number(rl,'min_np');settings.max_np=number(rl,'max_np');
settings.baseline=[settings.np;number(base,'q_y');number(base,'q_psi');number(base,'r_delta')];
assert(settings.nc<=settings.min_np && settings.min_np<settings.np && settings.np<settings.max_np);
settings.key=sprintf('np%d_nc%d',settings.np,settings.nc);
settings.fixed_label=['fixed_' settings.key];
end
function value=number(text,key)
token=regexp(text,['(?m)^' key '\s*=\s*([0-9.eE+-]+)\s*$'],'tokens','once');
assert(~isempty(token),'Missing configuration key %s',key);value=str2double(token{1});
end
