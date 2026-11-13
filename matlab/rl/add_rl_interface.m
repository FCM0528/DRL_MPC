function add_rl_interface()
root=setup();file=fullfile(root,'models','DRL_MPC.slx');
scope=migration_id('DRL_Adaptive_Tuning');
scopePath=Simulink.ID.getFullName(['DRL_MPC:' extractAfter(scope,'blk_')]);
if ~isempty(find_system(scopePath,'SearchDepth',1,'Name','RL_Agent')),return;end
observation=migration_id('Observation_Builder',scope);neutral=migration_id('Neutral_Parameters',scope);output=migration_id('Output_1',scope);reward=migration_id('Reward_Termination');
agent=create_agent('ddpg',42);assignin('base','agent',agent);
ops={struct('op','add_block','type','Inport','name','Decision_Reward','ref','rew','params',struct('Port','3'));
struct('op','add_block','type','Inport','name','Episode_Done','ref','done','params',struct('Port','4'));
struct('op','add_block','type','rllib/RL Agent','name','RL_Agent','ref','agent','params',struct('Agent','agent'));
struct('op','add_block','type','simulink/User-Defined Functions/Level-2 MATLAB S-Function','name','Action_Mapping','ref','map','params',struct('Parameters','''action''','FunctionName','migration_sfun'));
struct('op','add_block','type','Constant','name','Control_Mode','ref','mode','params',struct('Value','controller_mode'));
struct('op','add_block','type','Switch','name','Select_Fixed_or_RL','ref','switch','params',struct('Threshold','0.5'));
struct('op','connect','target',[observation '.y1 -> #agent.u1']);
struct('op','connect','target','#rew.y1 -> #agent.u2');
struct('op','connect','target','#done.y1 -> #agent.u3');
struct('op','connect','target','#agent.y1 -> #map.u1');
struct('op','connect','target','#map.y1 -> #switch.u1');
struct('op','connect','target','#mode.y1 -> #switch.u2');
struct('op','connect','target',[neutral '.y1 -> #switch.u3']);
struct('op','connect','target',['#switch.y1 -> ' output '.u1'])};
result=model_edit(file,scope,jsonencode(ops),'incremental');disp(result);
assert(~contains(result,'status: partial')&&~contains(result,'status: error'),result);
ops={struct('op','connect','target',[reward '.y1 -> ' scope '.u3']),struct('op','connect','target',[reward '.y2 -> ' scope '.u4'])};
disp(model_edit(file,'root',jsonencode(ops),'incremental'));
save_system('DRL_MPC');
end

