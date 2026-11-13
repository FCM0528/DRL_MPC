function build_model()
% Rebuild via SATK model_edit. Keep the original CarSim masked S-Function.
root=setup();model='DRL_MPC';file=fullfile(root,'models',[model '.slx']);
open_system(file);
assert(strcmp(get_param(model,'SimulationStatus'),'stopped'));
stamp=char(datetime('now','Format','yyyyMMdd_HHmmss'));
copyfile(file,fullfile(root,'backups',['before_build_' stamp '.slx']));
plant=Simulink.ID.getFullName('DRL_MPC:1');
% Existing SID 1 is intentionally preserved, including CarSim mask metadata.
if ~strcmp(get_param(plant,'Parent'),model)
    apply_current_configuration();disp(model_check(model,'root'));return
end
blocks=find_system(model,'SearchDepth',1,'Type','Block');ops={};
for k=2:numel(blocks)
    if ~strcmp(blocks{k},plant),ops{end+1}=struct('op','delete','target',id(blocks{k}));end
end
edit('root',ops);
Simulink.BlockDiagram.createSubsystem(get_param(plant,'Handle'),'Name','CarSim_Vehicle_Plant');
plant=Simulink.ID.getFullName('DRL_MPC:1');
plantScope=id(get_param(plant,'Parent'));
plantPath=get_param(plant,'Parent');
existingIn=find_system(plantPath,'SearchDepth',1,'BlockType','Inport');
existingOut=find_system(plantPath,'SearchDepth',1,'BlockType','Outport');
if isempty(existingIn),pin=add(plantScope,'Inport','Front_Wheel_deg',struct('PortDimensions','2'));else,pin=id(existingIn{1});end
if isempty(existingOut),pout=add(plantScope,'Outport','Raw_20_Channels',struct());else,pout=id(existingOut{1});end
wire(plantScope,pin,1,'blk_1',1);wire(plantScope,'blk_1',1,pout,1);
fields={'X_m','Y_m','Vx_mps','Vy_mps','Yaw_rad','YawRate_radps','Ax_mps2','Ay_mps2', ...
    'SteerL_rad','SteerR_rad','Beta_rad','RoadCurvature','AlphaL1_rad','AlphaR1_rad','AlphaL2_rad','AlphaR2_rad','FyL1_N','FyR1_N','FyL2_N','FyR2_N'};
[proc,ins,outs]=sub('CarSim_State_Processing',1,1);
dm=add(proc,'Demux','Channel_Order',struct('Outputs','20'));
bc=add(proc,'BusCreator','Vehicle_State_SI',struct('Inputs','20'));
wire(proc,ins{1},1,dm,1);
scale=ones(1,20);scale(3:4)=1/3.6;scale([5 6 9 10 11 13:16])=pi/180;scale(7:8)=9.80665;
for k=1:20
    g=add(proc,'Gain',fields{k},struct('Gain',num2str(scale(k),17)));
    wire(proc,dm,k,g,1);wire(proc,g,1,bc,k);
    edit(proc,{struct('op','configure','target',[g '.y1 -> ' bc '.u' num2str(k)],'params',struct('Name',fields{k}))});
end
wire(proc,bc,1,outs{1},1);
[ref,ri,ro]=sub('Reference_Trajectory_Preview',1,1);
sv=stateVector(ref,ri{1});f=sf(ref,'CarSim_LTARG_Preview','reference');wire(ref,sv,1,f,1);wire(ref,f,1,ro{1},1);
[err,ei,eo]=sub('Tracking_Error_Calculation',2,1);
sv=stateVector(err,ei{1});f=sf(err,'Tracking_Errors','errors');wire(err,sv,1,f,1);wire(err,ei{2},1,f,2);wire(err,f,1,eo{1},1);
[ctrl,ci,co]=sub('LTV_MPC_Controller',2,1);
sv=stateVector(ctrl,ci{1});f=sf(ctrl,'Incremental_LTV_QP','controller');wire(ctrl,sv,1,f,1);wire(ctrl,ci{2},1,f,2);wire(ctrl,f,1,co{1},1);
[drl,di,do]=sub('DRL_Adaptive_Tuning',2,1);
sv=stateVector(drl,di{1});f=sf(drl,'Observation_Builder','observation');wire(drl,sv,1,f,1);wire(drl,di{2},1,f,2);
obs=add(drl,'ToWorkspace','Observation_Log',struct('VariableName','observation_log','SaveFormat','Timeseries'));wire(drl,f,1,obs,1);
c=add(drl,'Constant','Neutral_Parameters',struct('Value','mpc_baseline_tuning'));wire(drl,c,1,do{1},1);
[act,ai,ao]=sub('Steering_Actuator',1,1);
sel=add(act,'Selector','First_Control_Move',struct('NumberOfDimensions','1','Indices','1','InputPortWidth','9'));
g=add(act,'Gain','Radians_to_Degrees',struct('Gain','180/pi'));
mx=add(act,'Mux','Paired_Front_Wheels',struct('Inputs','2'));
wire(act,ai{1},1,sel,1);wire(act,sel,1,g,1);wire(act,g,1,mx,1);wire(act,g,1,mx,2);wire(act,mx,1,ao{1},1);
[reward,wi,wo]=sub('Reward_Termination',2,2);sv=stateVector(reward,wi{1});f=sf(reward,'Tracking_V2_Reward','reward');wire(reward,sv,1,f,1);wire(reward,wi{2},1,f,2);wire(reward,f,1,wo{1},1);wire(reward,f,2,wo{2},1);
[mon,mi,~]=sub('Monitoring_Evaluation',5,0);
names={'state_log','control_log','reference_log','error_log','reward_log'};
for k=1:5
    src=mi{k};if k==1,src=stateVector(mon,src);end
    w=add(mon,'ToWorkspace',names{k},struct('VariableName',names{k},'SaveFormat','Timeseries'));wire(mon,src,1,w,1);
end
stop=add('root','Stop','Episode_Complete',struct());
delay=add('root','UnitDelay','Previous_Control_Result',struct('SampleTime','0.04','InitialCondition','mpc_initial_control_result'));
hold=add('root','ZeroOrderHold','MPC_Sampling',struct('SampleTime','0.04'));
wire('root',plantScope,1,hold,1);wire('root',hold,1,proc,1);
for target={ref,err,ctrl,drl,reward,mon},wire('root',proc,1,target{1},1);end
wire('root',ref,1,err,2);wire('root',ref,1,mon,3);wire('root',err,1,mon,4);
wire('root',drl,1,ctrl,2);wire('root',ctrl,1,act,1);wire('root',act,1,plantScope,1);
wire('root',ctrl,1,delay,1);wire('root',delay,1,drl,2);wire('root',delay,1,reward,2);
wire('root',ctrl,1,mon,2);wire('root',reward,1,mon,5);wire('root',reward,2,stop,1);
edit('root',{struct('op','configure','target','config:DRL_MPC','params',struct('SolverType','Fixed-step','Solver','FixedStepDiscrete','FixedStep','0.0005','StopTime','20','ReturnWorkspaceOutputs','on','InitFcn','mpc_mex(''reset'');'))});
save_system(model);fprintf('Saved modular model: %s\n',file);
add_rl_interface();add_steering_test_mode();
expand_controller_pipeline();refine_drl_layout();
    function result=edit(scope,operations)
        result=model_edit(file,scope,jsonencode(operations),'incremental');
        if contains(result,'status: partial')||contains(result,'status: error'),error('%s',result);end
    end
    function b=add(scope,type,name,params)
        out=edit(scope,{struct('op','add_block','type',type,'name',name,'ref','newblock','params',params)});
        tok=regexp(out,'newblock: (blk_\d+)','tokens','once');assert(~isempty(tok),out);b=tok{1};
    end
    function wire(scope,a,ap,b,bp)
        edit(scope,{struct('op','connect','target',sprintf('%s.y%d -> %s.u%d',a,ap,b,bp))});
    end
    function [b,ip,op]=sub(name,ni,no)
        b=add('root','SubSystem',name,struct());ip=cell(1,ni);op=cell(1,no);
        for z=1:ni,ip{z}=add(b,'Inport',['Input_' num2str(z)],struct('Port',num2str(z)));end
        for z=1:no,op{z}=add(b,'Outport',['Output_' num2str(z)],struct('Port',num2str(z)));end
    end
    function b=stateVector(scope,src)
        bs=add(scope,'BusSelector','Vehicle_State_Fields',struct('OutputSignals',strjoin(fields,',')));
        b=add(scope,'Mux','SI_State_Vector',struct('Inputs','20'));wire(scope,src,1,bs,1);
        for z=1:20,wire(scope,bs,z,b,z);end
    end
    function b=sf(scope,name,mode)
        b=add(scope,'simulink/User-Defined Functions/Level-2 MATLAB S-Function',name,struct('Parameters',['''' mode ''''],'FunctionName','migration_sfun'));
    end
end
function b=id(path)
sid=Simulink.ID.getSID(path);b=['blk_' extractAfter(sid,':')];
end

