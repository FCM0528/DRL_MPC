function finalize_model()
root=setup();model='DRL_MPC';open_system(fullfile(root,'models',[model '.slx']));
copyfile(fullfile(root,'models',[model '.slx']),fullfile(root,'backups',['before_final_layout_' char(datetime('now','Format','yyyyMMdd_HHmmss')) '.slx']));
expand_controller_pipeline();refine_drl_layout();
plant=migration_id('CarSim_Vehicle_Plant');plantPath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(plant,'blk_')]);
if isempty(find_system(plantPath,'SearchDepth',1,'Name','Raw_CarSim_Channels'))
    ops={struct('op','add_block','type','ToWorkspace','name','Raw_CarSim_Channels','ref','raw','params',struct('VariableName','raw_carsim','SaveFormat','Timeseries')),struct('op','connect','target','blk_1.y1 -> #raw.u1')};
    disp(model_edit(model,plant,jsonencode(ops),'incremental'));
end
consumers={'Reference_Trajectory_Preview','Tracking_Error_Calculation','LTV_MPC_Controller','DRL_Adaptive_Tuning','Reward_Termination','Monitoring_Evaluation'};
for k=1:numel(consumers)
    scope=migration_id(consumers{k});input=migration_id('Input_1',scope);
    inputPath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(input,'blk_')]);
    if strcmp(get_param(inputPath,'IsBusElementPort'),'on'),continue;end
    ops={struct('op','replace_block','target',input,'type','simulink/Ports & Subsystems/In Bus Element','params',struct('Port','1','PortName','Vehicle_State','Element',''))};
    out=model_edit(model,scope,jsonencode(ops),'incremental');assert(~contains(out,'status: partial'),out);
end
scope=migration_id('CarSim_State_Processing');output=migration_id('Output_1',scope);
outputPath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(output,'blk_')]);
ops={struct('op','replace_block','target',output,'type','simulink/Ports & Subsystems/Out Bus Element','params',struct('Port','1','PortName','Vehicle_State','Element',''))};
if ~strcmp(get_param(outputPath,'IsBusElementPort'),'on'),disp(model_edit(model,scope,jsonencode(ops),'incremental'));end
% The diagram was generated in this migration; arrange each functional scope.
subsystems=find_system(model,'LookUnderMasks','none','FollowLinks','off','BlockType','SubSystem');
for k=numel(subsystems):-1:1
    if strcmp(get_param(subsystems{k},'LinkStatus'),'none')
        sid=['blk_' extractAfter(Simulink.ID.getSID(subsystems{k}),':')];
        disp(model_edit(model,sid,'[]','full'));
    end
end
disp(model_edit(model,'root','[]','full'));
% SATK full layout can lose fanout edges on bus element ports; restore and check.
source=migration_id('CarSim_State_Processing');ops={};
if isempty(find_system(model,'SearchDepth',1,'Name','Vehicle_State_Bus'))
    for k=1:numel(consumers),target=migration_id(consumers{k});ops{k}=struct('op','connect','target',[source '.y1 -> ' target '.u1']);end
else
    target=migration_id('Vehicle_State_Bus');ops={struct('op','connect','target',[source '.y1 -> ' target '.u1'])};
end
disp(model_edit(model,'root',jsonencode(ops),'incremental'));
refine_main_view();
tidy_mpc_view();fix_local_spacing();annotate_model_chinese();
postload='addpath(fileparts(fileparts(get_param(''DRL_MPC'',''FileName'')))); setup;';
ops={struct('op','configure','target','config:DRL_MPC','params',struct('PostLoadFcn',postload))};
disp(model_edit(model,'root',jsonencode(ops),'incremental'));
check=model_check(model,'root');disp(check);assert(contains(check,'status: healthy'),check);
set_param(model,'SimulationCommand','update');save_system(model);
print(['-s' model],'-dpng',fullfile(root,'results','model_overview.png'));
fid=fopen(fullfile(root,'results','model_structure.txt'),'w');fprintf(fid,'%s\n',model_overview(model,'root','interfaces'));fclose(fid);
metadata=struct('model','DRL_MPC','compiled',true,'structure_healthy',true, ...
    'matlab_version',version,'matlab_pid',feature('getpid'),'fixed_step_s',.0005, ...
    'mpc_sample_s',.04,'rl_sample_s',.08,'native_stages',7);
settings=read_mpc_settings();training=read_training_config();metadata.mpc=settings;metadata.default_training_episodes=training.training.episodes;
fid=fopen(fullfile(root,'results','model_validation.json'),'w');fprintf(fid,'%s',jsonencode(metadata,'PrettyPrint',true));fclose(fid);
end
