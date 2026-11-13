function apply_current_configuration()
% Apply configuration through SATK; numeric constants live in setup/INI files.
root=setup();open_system(fullfile(root,'models','DRL_MPC.slx'));
scope=migration_id('DRL_Adaptive_Tuning');neutral=migration_id('Neutral_Parameters',scope);
result=model_edit('DRL_MPC',scope,jsonencode({struct('op','configure','target',neutral,'params',struct('Value','mpc_baseline_tuning'))}),'incremental');
assert(~contains(result,'status: partial'),result);
delay=migration_id('Previous_Control_Result');
result=model_edit('DRL_MPC','root',jsonencode({struct('op','configure','target',delay,'params',struct('InitialCondition','mpc_initial_control_result'))}),'incremental');
assert(~contains(result,'status: partial'),result);
refine_main_view();tidy_mpc_view();fix_local_spacing();annotate_model_chinese();
check=model_check('DRL_MPC','root');assert(contains(check,'status: healthy'),check);
set_param('DRL_MPC','SimulationCommand','update');save_system('DRL_MPC');
settings=read_mpc_settings();
print('-sDRL_MPC','-dpng',fullfile(root,'results',['model_overview_' settings.key '.png']));
print('-sDRL_MPC/LTV_MPC_Controller','-dpng',fullfile(root,'results',['mpc_subsystem_' settings.key '.png']));
end
