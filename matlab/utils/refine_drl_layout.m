function refine_drl_layout()
model='DRL_MPC';scope=migration_id('DRL_Adaptive_Tuning');scopePath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(scope,'blk_')]);
if ~isempty(find_system(scopePath,'SearchDepth',1,'Name','Parameter_Update')),return;end
observation=migration_id('Observation_Builder',scope);mapping=migration_id('Action_Mapping',scope);agent=migration_id('RL_Agent',scope);selection=migration_id('Select_Fixed_or_RL',scope);
% Preserve SIDs and wiring when exposing functional subsystems.
ops={struct('op','configure','target',observation,'params',struct('Name','Road_Preview_V2')),struct('op','configure','target',mapping,'params',struct('Name','Discrete_Parameter_Map'))};
disp(model_edit(model,scope,jsonencode(ops),'incremental'));
obsPath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(observation,'blk_')]);
Simulink.BlockDiagram.createSubsystem(get_param(obsPath,'Handle'),'Name','Observation_Builder');
obs=get_param(Simulink.ID.getFullName(['DRL_MPC:' extractAfter(observation,'blk_')]),'Parent');obs=['blk_' extractAfter(Simulink.ID.getSID(obs),':')];
mapPath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(mapping,'blk_')]);
Simulink.BlockDiagram.createSubsystem(get_param(mapPath,'Handle'),'Name','Action_Mapping');
map=get_param(Simulink.ID.getFullName(['DRL_MPC:' extractAfter(mapping,'blk_')]),'Parent');map=['blk_' extractAfter(Simulink.ID.getSID(map),':')];
ops={struct('op','add_block','type','Goto','name','Road_Preview_Observation','ref','goto','params',struct('GotoTag','RoadPreviewV2','TagVisibility','scoped')), ...
 struct('op','add_block','type','From','name','Agent_Observation','ref','from','params',struct('GotoTag','RoadPreviewV2')), ...
 struct('op','add_block','type','GotoTagVisibility','name','Observation_Tag_Scope','params',struct('GotoTag','RoadPreviewV2')), ...
 struct('op','connect','target',[obs '.y1 -> #goto.u1']),struct('op','connect','target',['#from.y1 -> ' agent '.u1']), ...
 struct('op','add_block','type','SubSystem','name','Parameter_Update','ref','update')};
result=model_edit(model,scope,jsonencode(ops),'incremental');disp(result);
token=regexp(result,'update: (blk_\d+)','tokens','once');update=token{1};
ops={struct('op','add_block','type','Inport','name','Mapped_Parameters','ref','in'), ...
 struct('op','add_block','type','ZeroOrderHold','name','DRL_80ms_Parameter_Hold','ref','hold','params',struct('SampleTime','0.08')), ...
 struct('op','add_block','type','Outport','name','Applied_Parameters','ref','out'), ...
 struct('op','connect','target','#in.y1 -> #hold.u1'),struct('op','connect','target','#hold.y1 -> #out.u1')};
disp(model_edit(model,update,jsonencode(ops),'full'));
ops={struct('op','connect','target',[map '.y1 -> ' update '.u1']),struct('op','connect','target',[update '.y1 -> ' selection '.u1'])};
disp(model_edit(model,scope,jsonencode(ops),'incremental'));
disp(model_read(model,scope,'0'));disp(model_check(model,scope));save_system(model);
end
