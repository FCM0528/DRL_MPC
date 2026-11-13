function write_current_configuration_report()
root=setup();settings=read_mpc_settings();training=read_training_config();
metrics=readtable(fullfile(root,'results',settings.fixed_label,'metrics.csv'));
annotations=find_system('DRL_MPC','FindAll','on','LookUnderMasks','all','FollowLinks','off','Type','annotation');
count=0;
for k=1:numel(annotations)
    a=get_param(annotations(k),'Object');
    if startsWith(string(a.Tag),'DRLMPC_CN_'),assert(strcmp(a.FontName,'宋体'));count=count+1;end
end
assert(count>20);assert(training.training.episodes==300 && settings.np==40 && settings.nc==12);
assert(isequal(mpc_mex('map',zeros(4,1)),settings.baseline));
check=model_check('DRL_MPC','root');assert(contains(check,'status: healthy'),check);
new_agents_trained=true;
for algorithm={'ddpg','td3'}
    file=fullfile(root,'results',sprintf('training_%s_%s_ep300_seed%d',algorithm{1},settings.key,training.training.seed),'final_agent.mat');
    if ~isfile(file),new_agents_trained=false;continue;end
    data=load(file,'model_settings','stats');
    new_agents_trained=new_agents_trained && isfield(data,'model_settings') && isequal(data.model_settings,settings) && numel(data.stats.EpisodeIndex)==300;
end
metadata=struct('mpc',settings,'default_training_episodes',training.training.episodes, ...
    'annotation_font','宋体','chinese_annotations',count,'compiled',true, ...
    'structure_healthy',true,'new_agents_trained',new_agents_trained,'matlab_pid',feature('getpid'));
fid=fopen(fullfile(root,'results','current_configuration.json'),'w');fprintf(fid,'%s',jsonencode(metadata,'PrettyPrint',true));fclose(fid);
fid=fopen(fullfile(root,'results','CURRENT_CONFIGURATION.md'),'w','n','UTF-8');
fprintf(fid,'# 当前 MATLAB / Simulink 配置\n\n');
fprintf(fid,'- 固定 MPC：Np=%d，Nc=%d；MPC 周期 0.04 s。\n',settings.np,settings.nc);
fprintf(fid,'- DDPG / TD3 默认训练：%d 回合；策略周期 0.08 s。\n',training.training.episodes);
fprintf(fid,'- RL 时域范围：%d–%d；零动作对应 Np=%d。\n',settings.min_np,settings.max_np,settings.np);
fprintf(fid,'- 模型内 %d 条中文注释均为宋体；七阶段前的 20 路状态连线已封装为可展开输入整理子系统。\n',count);
fprintf(fid,'- 模型结构检查与编译通过；新基准已实际完成 CarSim 闭环。\n\n');
fprintf(fid,'原始仿真全程（至约 150 m）横向 RMSE：%.6f m；最大绝对横向误差：%.6f m；QP 失败：%d；实际末点 X：%.4f m。\n\n',metrics.lateral_rmse_m,metrics.max_abs_lateral_m,metrics.qp_failures,metrics.end_x_m);
fprintf(fid,'当前距离图统一显示 0–120 m，图中误差统计使用 comparison_%s/metrics_120m.csv；完整原始数据和全程指标仍保留。诊断结论见 drl_diagnosis/DIAGNOSIS.md。\n\n',settings.key);
if new_agents_trained
    fprintf(fid,'DDPG / TD3 已分别实际完成 300 回合训练（种子 42）。同工况关闭探索噪声的对比见 comparison_%s/metrics.csv、comparison_cn.png 和 training_comparison.png。\n\n',settings.key);
else
    fprintf(fid,'当前参数的 300 回合训练尚未全部完成。\n\n');
end
fprintf(fid,'原 47/7、500 回合 Agent 和对照结果保留为历史记录，不作为 40/12 的结果。\n\n');
fprintf(fid,'当前训练目录：training_ddpg_%s_ep300_seed42 / training_td3_%s_ep300_seed42。compare_controllers 仅接受匹配当前参数的新 Agent。\n',settings.key,settings.key);
fclose(fid);
end
