function refine_main_view()
% Left-to-right view. Scoped tags carry the state bus and return signals.
model='DRL_MPC';consumers={'Reference_Trajectory_Preview','Tracking_Error_Calculation','LTV_MPC_Controller','DRL_Adaptive_Tuning','Reward_Termination','Monitoring_Evaluation'};
source=migration_id('CarSim_State_Processing');plant=migration_id('CarSim_Vehicle_Plant');act=migration_id('Steering_Actuator');controller=migration_id('LTV_MPC_Controller');delay=migration_id('Previous_Control_Result');monitor=migration_id('Monitoring_Evaluation');
if isempty(find_system(model,'SearchDepth',1,'Name','Vehicle_State_Bus'))
    ops={};
    tags={'VehicleStateSI','FrontWheelDeg','MpcResult'};
    gotoNames={'Vehicle_State_Bus','External_Steering_Command','MPC_Result'};
    for k=1:3
        ops{end+1}=struct('op','add_block','type','Goto','name',gotoNames{k},'ref',['g' num2str(k)],'params',struct('GotoTag',tags{k},'TagVisibility','scoped','ShowName','off'));
        ops{end+1}=struct('op','add_block','type','GotoTagVisibility','name',[gotoNames{k} '_Scope'],'params',struct('GotoTag',tags{k},'ShowName','off'));
    end
    for k=1:6
        ops{end+1}=struct('op','add_block','type','From','name',['State_For_' num2str(k)],'ref',['f' num2str(k)],'params',struct('GotoTag','VehicleStateSI','ShowName','off'));
        ops{end+1}=struct('op','connect','target',['#f' num2str(k) '.y1 -> ' migration_id(consumers{k}) '.u1']);
    end
    ops=[ops,{struct('op','add_block','type','From','name','Steering_For_Plant','ref','steer','params',struct('GotoTag','FrontWheelDeg','ShowName','off')), ...
        struct('op','add_block','type','From','name','Result_For_Delay','ref','d','params',struct('GotoTag','MpcResult','ShowName','off')), ...
        struct('op','add_block','type','From','name','Result_For_Monitor','ref','m','params',struct('GotoTag','MpcResult','ShowName','off')), ...
        struct('op','connect','target',[source '.y1 -> #g1.u1']),struct('op','connect','target',[act '.y1 -> #g2.u1']), ...
        struct('op','connect','target',[controller '.y1 -> #g3.u1']),struct('op','connect','target',['#steer.y1 -> ' plant '.u1']), ...
        struct('op','connect','target',['#d.y1 -> ' delay '.u1']),struct('op','connect','target',['#m.y1 -> ' monitor '.u2'])}];
    result=model_edit(model,'root',jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
end
ops={struct('op','connect','target',[source '.y1 -> ' migration_id('Vehicle_State_Bus') '.u1']), ...
    struct('op','connect','target',[act '.y1 -> ' migration_id('External_Steering_Command') '.u1']), ...
    struct('op','connect','target',[controller '.y1 -> ' migration_id('MPC_Result') '.u1']), ...
    struct('op','connect','target',[migration_id('Steering_For_Plant') '.y1 -> ' plant '.u1']), ...
    struct('op','connect','target',[migration_id('Result_For_Delay') '.y1 -> ' delay '.u1']), ...
    struct('op','connect','target',[migration_id('Result_For_Monitor') '.y1 -> ' monitor '.u2'])};
for k=1:6,ops{end+1}=struct('op','connect','target',[migration_id(['State_For_' num2str(k)]) '.y1 -> ' migration_id(consumers{k}) '.u1']);end
result=model_edit(model,'root',jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
[extraNames,extraPositions]=route_monitor_signals();
names=[consumers,{'CarSim_Vehicle_Plant','Steering_Actuator','MPC_Sampling','Previous_Control_Result','Episode_Complete', ...
    'Vehicle_State_Bus','External_Steering_Command','MPC_Result','Vehicle_State_Bus_Scope','External_Steering_Command_Scope','MPC_Result_Scope', ...
    'State_For_1','State_For_2','State_For_3','State_For_4','State_For_5','State_For_6','Steering_For_Plant','Result_For_Delay','Result_For_Monitor'}];
% Keep the plant, sampling, SI conversion and MPC on their existing main row.
% Give the reference, learning/reward and logging branches separate corridors.
positions=[710 60 950 150;1180 60 1400 150;1080 245 1320 345;710 510 950 670;1170 535 1410 715;1770 535 2010 735; ...
    160 245 360 345;1510 245 1750 345;395 275 435 315;710 800 765 840;1770 820 1840 860; ...
    700 280 810 310;1810 280 1940 310;1350 365 1460 395;160 960 380 980;430 960 650 980;700 960 920 980; ...
    570 85 680 115;1040 70 1140 100;930 245 1030 275;520 520 650 545;980 545 1110 570;1570 545 1710 569; ...
    0 280 115 310;520 805 650 835;1570 585 1710 609];
names=[names extraNames];positions=[positions;extraPositions];
ops=cell(1,numel(names)+1);
for k=1:numel(names)
    [id,path]=migration_id(names{k});params=struct('Position',mat2str(positions(k,:)),'FontSize','10');
    if strcmp(get_param(path,'BlockType'),'From'),params.FontSize='9';end
    ops{k}=struct('op','configure','target',id,'params',params);
end
id=migration_id('CarSim_State_Processing');ops{end}=struct('op','configure','target',id,'params',struct('Position','[470 245 670 345]','FontSize','10'));
result=model_edit(model,'root',jsonencode(ops),'incremental');disp(result);assert(~contains(result,'status: partial'),result);
check=model_check(model,'root');assert(contains(check,'status: healthy'),check);
% SATK's parameter whitelist excludes this display-only native parameter.
% Native fallback changes the preview icon only; all topology and positions use SATK.
for k=1:6
    [~,path]=migration_id(consumers{k});set_param(path,'ContentPreviewEnabled','off');
end
for name={'CarSim_Vehicle_Plant','CarSim_State_Processing','Steering_Actuator'}
    [~,path]=migration_id(name{1});set_param(path,'ContentPreviewEnabled','off');
end
Simulink.BlockDiagram.routeLine(find_system(model,'FindAll','on','SearchDepth',1,'Type','Line'));
check=model_check(model,'root');assert(contains(check,'status: healthy'),check);
end
