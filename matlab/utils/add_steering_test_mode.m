function add_steering_test_mode()
scope=migration_id('Steering_Actuator');scopePath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(scope,'blk_')]);
if ~isempty(find_system(scopePath,'SearchDepth',1,'Name','Steering_Interface_Test')),return;end
mux=migration_id('Paired_Front_Wheels',scope);output=migration_id('Output_1',scope);
ops={struct('op','add_block','type','Constant','name','Test_Wheel_Angles_deg','ref','angles','params',struct('Value','[steering_test_deg;steering_test_deg]'));
struct('op','add_block','type','Constant','name','Interface_Test_Enable','ref','enable','params',struct('Value','steering_interface_test'));
struct('op','add_block','type','Switch','name','Steering_Interface_Test','ref','switch','params',struct('Threshold','0.5'));
struct('op','connect','target','#angles.y1 -> #switch.u1');
struct('op','connect','target','#enable.y1 -> #switch.u2');
struct('op','connect','target',[mux '.y1 -> #switch.u3']);
struct('op','connect','target',['#switch.y1 -> ' output '.u1'])};
disp(model_edit('DRL_MPC',scope,jsonencode(ops),'incremental'));save_system('DRL_MPC');
end


