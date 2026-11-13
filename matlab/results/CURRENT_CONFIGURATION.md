# 当前 MATLAB / Simulink 配置

- 固定 MPC：Np=40，Nc=12；MPC 周期 0.04 s。
- DDPG / TD3 默认训练：300 回合；策略周期 0.08 s。
- RL 时域范围：12–50；零动作对应 Np=40。
- 模型内 40 条中文注释均为宋体；七阶段前的 20 路状态连线已封装为可展开输入整理子系统。
- 模型结构检查与编译通过；新基准已实际完成 CarSim 闭环。

原始仿真全程（至约 150 m）横向 RMSE：0.082592 m；最大绝对横向误差：0.259639 m；QP 失败：0；实际末点 X：150.5820 m。

当前距离图统一显示 0–120 m，图中误差统计使用 comparison_np40_nc12/metrics_120m.csv；完整原始数据和全程指标仍保留。诊断结论见 drl_diagnosis/DIAGNOSIS.md。

DDPG / TD3 已分别实际完成 300 回合训练（种子 42）。同工况关闭探索噪声的对比见 comparison_np40_nc12/metrics.csv、comparison_cn.png 和 training_comparison.png。

原 47/7、500 回合 Agent 和对照结果保留为历史记录，不作为 40/12 的结果。

当前训练目录：training_ddpg_np40_nc12_ep300_seed42 / training_td3_np40_nc12_ep300_seed42。compare_controllers 仅接受匹配当前参数的新 Agent。
