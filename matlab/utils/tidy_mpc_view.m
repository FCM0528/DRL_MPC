function tidy_mpc_view()
% Collapse the twenty field wires into a real input adapter, then align stages.
model='DRL_MPC';[scope,path]=migration_id('LTV_MPC_Controller');
if isempty(find_system(path,'SearchDepth',1,'Name','State_Input_Adapter'))
    [~,selector]=migration_id('Vehicle_State_Fields',scope);[~,mux]=migration_id('SI_State_Vector',scope);
    Simulink.BlockDiagram.createSubsystem([get_param(selector,'Handle'),get_param(mux,'Handle')],'Name','State_Input_Adapter');
    disp(model_read(model,scope,'0'));disp(model_check(model,scope));
end
names={'State_Input_Adapter','Vehicle_Prediction_Model','Time_Varying_Linearization','Reference_Prediction','Cost_Function','Constraints','QP_Solver','Steering_Output','Input_1','Input_2','Output_1'};
positions=[110 180 280 260;330 180 500 260;585 180 755 260;840 180 1010 260;1095 180 1265 260;1350 180 1520 260;1605 180 1775 260;1860 180 2030 260;30 210 30 210;240 310 270 330;2090 210 2120 230];
ops=cell(1,numel(names));
for k=1:numel(names)
    id=migration_id(names{k},scope);ops{k}=struct('op','configure','target',id,'params',struct('Position',mat2str(positions(k,:)),'FontName','宋体','FontSize','10'));
end
result=model_edit(model,scope,jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
for k=1:8
    [~,block]=migration_id(names{k},scope);set_param(block,'ContentPreviewEnabled','off');
end
Simulink.BlockDiagram.routeLine(find_system(path,'FindAll','on','SearchDepth',1,'Type','Line'));
check=model_check(model,scope);assert(contains(check,'status: healthy'),check);
end
