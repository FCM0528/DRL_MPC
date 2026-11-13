function expand_controller_pipeline()
% Seven executable native stages, connected in dependency order.
% Run only after native replay passes and a new MEX has been built.
model='DRL_MPC';scope=migration_id('LTV_MPC_Controller');scopePath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(scope,'blk_')]);
if ~isempty(find_system(scopePath,'SearchDepth',1,'Name','Vehicle_Prediction_Model')),return;end
legacy=migration_id('Incremental_LTV_QP',scope);state=migration_id('SI_State_Vector',scope);tuning=migration_id('Input_2',scope);output=migration_id('Output_1',scope);
disp(model_read(model,scope,'0'));
names={'Vehicle_Prediction_Model','Time_Varying_Linearization','Reference_Prediction','Cost_Function','Constraints','QP_Solver','Steering_Output'};
modes={'operatingpoint','linearization','reference_stage','cost','constraints','qp','steering'};
ids=cell(1,7);
for k=1:7
    out=model_edit(model,scope,jsonencode({struct('op','add_block','type','SubSystem','name',names{k},'ref','stage')}),'incremental');
    token=regexp(out,'stage: (blk_\d+)','tokens','once');assert(~isempty(token),out);ids{k}=token{1};
    ops={struct('op','add_block','type','Inport','name','Input_1','ref','in1'), ...
         struct('op','add_block','type','Outport','name','Output_1','ref','out'), ...
         struct('op','add_block','type','simulink/User-Defined Functions/Level-2 MATLAB S-Function','name',[names{k} '_Core'],'ref','core','params',struct('Parameters',['''' modes{k} ''''],'FunctionName','migration_sfun')), ...
         struct('op','connect','target','#in1.y1 -> #core.u1'),struct('op','connect','target','#core.y1 -> #out.u1')};
    if k==1
        ops=[ops,{struct('op','add_block','type','Inport','name','Tuned_Parameters','ref','in2','params',struct('Port','2')),struct('op','connect','target','#in2.y1 -> #core.u2')}];
    end
    result=model_edit(model,ids{k},jsonencode(ops),'full');assert(~contains(result,'status: partial'),result);
end
ops={struct('op','delete','target',legacy),struct('op','connect','target',[state '.y1 -> ' ids{1} '.u1']),struct('op','connect','target',[tuning '.y1 -> ' ids{1} '.u2'])};
for k=1:6,ops{end+1}=struct('op','connect','target',[ids{k} '.y1 -> ' ids{k+1} '.u1']);end
ops{end+1}=struct('op','connect','target',[ids{7} '.y1 -> ' output '.u1']);
disp(model_edit(model,scope,jsonencode(ops),'incremental'));
disp(model_read(model,scope,'1'));disp(model_check(model,scope));save_system(model);
end
