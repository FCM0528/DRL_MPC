function annotate_model_chinese()
% Chinese documentation uses the installed SimSun family (宋体).
model='DRL_MPC';settings=read_mpc_settings();c=read_training_config();font='宋体';
assert(any(strcmp(listfonts,font)),'The requested SimSun font is not installed.');
old=find_system(model,'FindAll','on','LookUnderMasks','all','FollowLinks','off','Type','annotation');
for k=1:numel(old)
    a=get_param(old(k),'Object');
    if startsWith(string(a.Tag),'DRLMPC_CN_'),delete(a);end
end
roots={'CarSim_Vehicle_Plant','CarSim_State_Processing','Reference_Trajectory_Preview','Tracking_Error_Calculation','LTV_MPC_Controller','DRL_Adaptive_Tuning','Steering_Actuator','Monitoring_Evaluation','Reward_Termination','Previous_Control_Result'};
titles={'CarSim 车辆动力学','车辆状态与单位转换','参考轨迹读取与预瞄','轨迹跟踪误差计算','增量式时变模型预测控制','强化学习自适应调参','前轮转向执行接口','监测、记录与性能评价','奖励函数与回合终止','上一控制周期结果'};
ops={};
for k=1:numel(roots)
    [id,path]=migration_id(roots{k});p=get_param(path,'Position');
    ops{end+1}=struct('op','configure','target',id,'params',struct('FontName',font,'FontSize','10'));
    titleY=p(2)-25;
    note(model,titles{k},[p(1) titleY p(1)+max(240,p(3)-p(1)) titleY+20],12,['title_' num2str(k)]);
end
result=model_edit(model,'root',jsonencode(ops),'incremental');assert(~contains(result,'status: partial'),result);
note(model,sprintf('当前工况：DLC SYX，70 km/h\n固定 MPC：Np=%d，Nc=%d\nMPC 周期 0.04 s；策略周期 0.08 s\nDDPG / TD3 默认训练 %d 回合\n车辆动力学与目标轨迹均来自 CarSim',settings.np,settings.nc,c.training.episodes),[160 425 545 575],12,'configuration');
note(model,sprintf('仅从 CarSim 完整目标表读取参考\n插值计算位置、航向与曲率'),[710 188 950 228],11,'reference');
note(model,sprintf('Np=%d，Nc=%d，Ts=0.04 s\n内部保留七阶段预测与 QP 求解',settings.np,settings.nc),[1080 370 1320 405],11,'mpc');
note(model,sprintf('14 维观测，4 维参数动作\n每次决策对应两个 MPC 周期\n默认 %d 回合；零动作对应固定基准',c.training.episodes),[710 567 1000 617],11,'rl');
% Per-subsystem guides sit above their actual content, preserving block paths.
descriptions={
 'CarSim_Vehicle_Plant','车辆动力学：两路前轮角度输入（度），20 路原始状态输出。';
 'CarSim_State_Processing','状态处理：速度、角度、角速度与加速度转换为 SI 单位，并组织为车辆状态总线。';
 'Reference_Trajectory_Preview','参考预瞄：使用 CarSim 完整 LTARG_TABLE；不生成独立双移线。';
 'Tracking_Error_Calculation','误差计算：全局横向误差与车身航向误差，保持原坐标和符号约定。';
 'LTV_MPC_Controller',sprintf('LTV-MPC：Np=%d，Nc=%d；按连线顺序执行下方七个真实计算阶段。',settings.np,settings.nc);
 'DRL_Adaptive_Tuning',sprintf('自适应调参：观测 → Agent → 动作映射 → 参数保持；默认训练 %d 回合。',c.training.episodes);
 'Steering_Actuator','转向接口：取 MPC 首步转角，弧度转度，同时施加左右前轮。';
 'Monitoring_Evaluation','监测评价：记录状态、参考、误差、控制参数和奖励，输出 MAT、CSV 与图形。';
 'Reward_Termination','奖励与终止：保留 tracking_v2 奖励；质心 X 达到 150 m 或安全条件触发时结束回合。'};
for k=1:size(descriptions,1)
    [~,path]=migration_id(descriptions{k,1});guide(path,descriptions{k,2},['guide_' num2str(k)]);
end
controller=migration_id('LTV_MPC_Controller');
stages={'Vehicle_Prediction_Model','Time_Varying_Linearization','Reference_Prediction','Cost_Function','Constraints','QP_Solver','Steering_Output'};
texts={'车辆预测模型：初始化实测状态、车辆参数和上一控制量。','逐步线性化：滚动名义轨迹，使用 Tustin 离散化并递推灵敏度。','参考预测：按名义预测位置查询同一 CarSim 目标轨迹。','代价函数：跟踪误差、转角、转角增量与轮胎侧偏等完整代价。','约束构造：保持原转角幅值和速率约束逻辑。','QP 求解：保留 ADMM、热启动和收敛状态判定。','转角输出：约束投影、首步控制输出及历史控制量更新。'};
for k=1:numel(stages)
    [~,path]=migration_id(stages{k},controller);guide(path,texts{k},['stage_' num2str(k)]);
end
[~,controllerPath]=migration_id('LTV_MPC_Controller');
stageTitles={'车辆预测模型','时变线性化','参考预测','代价函数','约束构造','QP 求解','转角输出'};
for k=1:7
    [~,path]=migration_id(stages{k},controller);p=get_param(path,'Position');
    note(controllerPath,stageTitles{k},[p(1) p(2)-30 p(3) p(2)-10],12,['stage_title_' num2str(k)]);
end
if ~isempty(find_system(controllerPath,'SearchDepth',1,'Name','State_Input_Adapter'))
    [~,path]=migration_id('State_Input_Adapter',controller);p=get_param(path,'Position');
    note(controllerPath,'状态输入整理',[p(1) p(2)-30 p(3) p(2)-10],12,'state_adapter_title');
    guide(path,'状态输入整理：从车辆总线选择 20 路 SI 状态，按约定顺序组成预测模型输入。','state_adapter');
end
Simulink.BlockDiagram.routeLine(find_system(model,'FindAll','on','SearchDepth',1,'Type','Line'));
    function guide(scope,text,tag)
        b=find_system(scope,'FindAll','on','SearchDepth',1,'Type','Block');b=b(b~=get_param(scope,'Handle'));
        if isempty(b),return;end
        p=cell2mat(get_param(b,'Position'));left=min(p(:,1));top=min(p(:,2));
        note(scope,text,[left top-90 left+max(700,max(p(:,3))-left) top-35],12,tag);
    end
    function note(scope,text,position,size,tag)
        a=Simulink.Annotation(scope,text);a.Tag=['DRLMPC_CN_' tag];a.Interpreter='off';
        a.FontName=font;a.FontSize=size;a.Position=position;
    end
end
