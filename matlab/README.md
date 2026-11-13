# MATLAB / Simulink / CarSim 迁移工程

本目录独立于原 C++ / Python 工程。原 `src/`、`python/`、`configs/`、`docs/`、`tests/` 和根 README 均未修改。

当前固定基准为 **Np=40、Nc=12**，DDPG/TD3 默认训练 **300 回合**；RL 的 Np 范围为 **12–50**，零动作对应 Np=40。两种算法均已实际完成 300 回合训练（种子 42），当前配置与验证见 `results/CURRENT_CONFIGURATION.md`，本轮对比见 `results/comparison_np40_nc12/COMPARISON.md`。`results/VALIDATION.md`、`results/comparison/` 和原 `training_*_seed42/` 保留的是 **47/7、500 回合的历史结果**，与本轮结果分开存放。

## 运行

MATLAB R2025a 中：

```matlab
cd('C:\Users\fengchengmin\Desktop\JQRKZ\DRL-MPC\matlab')
setup
open_system(fullfile(pwd,'models','DRL_MPC.slx'))
run_fixed_mpc
test_steering_interface
test_native_parity

% 默认每种 300 回合，种子 42；每个策略动作持续两个 MPC 周期。
train_ddpg
train_td3
compare_controllers

% 或执行完整一轮：DDPG / TD3 各 300 回合（种子 42），随后评价并导出中文对比图。
% 此入口从新 Agent 开始训练；已有结果需先备份再重跑。
run_training_campaign

% 少量回合与多种子入口。6 回合约 588 个交互，覆盖 550 步学习预热。
train_controller('ddpg',6,42,'smoke')
train_controller('td3',6,42,'smoke')
train_ddpg(300,[42 43 44])
```

需要 Simulink、Reinforcement Learning Toolbox、Deep Learning Toolbox、有效 CarSim 求解器许可。检测到可用 GPU 时训练网络使用 GPU；车辆与 MPC 保持 CPU 计算。仿真采用 Normal 模式、禁用 Fast Restart 和并行环境；同一 MATLAB 会话只运行一个本工程模型。

当前参数的完整结果保存在 `results/comparison_np40_nc12/`：`comparison_cn.png` 为同工况闭环对比，`training_comparison.png` 为实际训练曲线，均附 PDF 和可编辑 FIG；`COMPARISON.md` 记录实际回合、交互数与数值指标。`results/training_campaign_status.json` 记录本轮执行状态。布局调整前的模型位于 `backups/before_spacing_training300_*/`，连线核对见 `results/layout_spacing_audit.json`。

距离图统一显示 **0–120 m**，对应数值指标为 `metrics_120m.csv`；时间曲线显示至车辆到达 120 m 的时刻，训练曲线保留 300 回合。原始仿真数据与全程指标仍保留。DRL 原因检查及 18 组实测消融见 `results/drl_diagnosis/DIAGNOSIS.md`，扩范围的固定参数探针不属于新训练的 DRL 结果。

## 原算法与 MEX

选择 C++ MEX 方案。`controllers/native/` 内保留原 MPC、车辆、参考轨迹、RL 观测及奖励的源码副本。`mpc_mex.cpp` 只负责 MATLAB 数据接口，不构造 `CarSimPlant`，不调用动力学积分 API；车辆动力学唯一由原 CarSim `vs_sf` S-Function 推进。

保留原增量控制、名义轨迹逐步线性化、Tustin 离散化、灵敏度递推、完整代价、约束、ADMM QP、热启动、QP 不收敛时的约束投影与预测异常时的前次转角保持。预测模型包含六个物理状态和两个轮胎松弛状态。纵向加速度沿用原算法的相邻 `Vx` 差分；Bus 中 `Ax` 的加速度计读数用于监测。

`utils/generate_pipeline.py` 从原副本提取 `PipelineMpc`，将相同运算公开为七个顺序阶段。生成代码位于 `mpc_pipeline.h/.cpp`，原副本保持不变。197 个实际记录状态的独立原程序对照已覆盖首步转角、QP 状态、预测状态、观测、参考和奖励。详细最大差异保存在 `results/native_parity.csv`；在线参数变化另有回放测试。

编译脚本使用本机 MSVC 14.44 与 R2025a 的头文件和导入库。MATLAB 的 `mex.getCompilerConfigurations` 未配置编译器，因此通过本机编译器直接生成 MEX。重新编译前停止仿真并 `clear mpc_mex`：

```powershell
cmd /c matlab\build_native.cmd
cmd /c matlab\build_parity.cmd
cmd /c matlab\build_pipeline_test.cmd
```

## CarSim 工况和单位

`carsim/case/source_simfile.sim`、`source_Run_all.par` 是工况快照；运行使用同目录的私有 `simfile.sim`、`Run_all.par`。`manifest.json` 记录源路径、摘要和副本调整。切换工况后先重新生成 CarSim Simfile，再运行 `utils/stage_case.py`，重新执行接口验收与 `setup`。该脚本要求本次已验证的两路输入、20 路输出及直线道路的 LTARG 全表格式；不支持的工况应先扩展适配器，不能直接替换参考轨迹。

本次源工况为 DLC SYX，70 km/h，0–20 s。初次审计发现第二路导入为 `IMP_STEER_L2`，之后用户已在源工况修正为 `IMP_STEER_R1`。当前源与副本均为 `IMP_STEER_L1`、`IMP_STEER_R1`，Replace 模式。完整 LTARG 表有 241 行，逐行一致性由 `source_audit.json` 记录。

原 Parsfile 积分步长为 **0.001 s**，Simfile 外部接口步长为 **0.0005 s**，均予以保留。MPC 为 **0.04 s**，DRL 为 **0.08 s**。副本沿用原 C++ 的站位终止余量，将 `SSTOP` 设为 160 m；评价仍在质心全局 X 首次达到 150 m 后的策略采样时刻终止，因此实际末点可能略超过 150 m。用户源工况 `SSTOP=150` 保持不变。

求解器回显实际核对的输出单位：

| 通道序号 | 内容 | 原单位 → 内部单位 |
| --- | --- | --- |
| 1–2 | Xcg_TM、Ycg_TM | m → m |
| 3–4 | Vx、Vy | km/h → m/s |
| 5–6 | Yaw、AVz | deg、deg/s → rad、rad/s |
| 7–8 | Ax、Ay | g → m/s²，g=9.80665 |
| 9–11 | Steer_L1、Steer_R1、Beta | deg → rad |
| 12 | Rho_Road | 1/m；不作为 DLC 目标曲率 |
| 13–16 | AlphL_L1/R1/L2/R2 | deg → rad |
| 17–20 | Fy_L1/R1/L2/R2 | N → N |

MEX 沿用原模型约定：轴平均滞后侧偏角取 CarSim 信号的负值，轴侧向力为左右轮之和。转向输出 rad 转 deg，分别施加左右前轮。零指令反馈允许原车辆前束及柔顺引起的小偏差；权限验收使用零、正、负指令和两侧实际反馈共同判定。

## 模型结构

主模型保留 CarSim Vehicle Plant、CarSim State Processing、Reference Trajectory & Preview、Tracking Error Calculation、LTV-MPC Controller、DRL Adaptive Tuning、Steering Actuator、Monitoring & Evaluation，并独立封装 Reward & Termination。

车辆状态以有名称、已转换单位的 Bus 传递。外部参考模块提供完整 LTARG 插值结果与未来 50 步预瞄；MPC 内部仍按其名义预测轨迹逐步查询同一参考表。DRL 使用真实 RL Agent 库块；动作只改变 Np、Qy、Qψ、RΔ，不直接控制转角。

`build_model.m` 使用 SATK `model_edit` 构建、连接及布局，并在修改前备份。已有模块化模型再次调用时保留当前版本并检查，不静默重建。若需从头恢复，请先将当前模型另行备份，再从 `backups/initial_*/DRL_MPC.slx` 恢复；初始备份记录 SHA256。历史 4 秒成功运行没有独立可核验的模型快照或日志，本工程不将其视作本次验收证据。

## 强化学习

14 维 `road_preview_v2` 观测、四维动作离散映射和 `tracking_v2` 奖励直接复用原 C++。零动作严格对应 `[40,9200,2800,13500]`。网络宽度 256、两层；DDPG 单 critic、TD3 双 critic；学习率 3e-4、τ=0.005、γ=0.995967610541、回放容量 100000、batch 256、预热 550、每步一次梯度更新、探索标准差 0.15。TD3 延迟 2，目标噪声标准差 0.2、截断 ±0.5。

MATLAB 环境使用 `rlSimulinkEnv`，每回合重新初始化原 CarSim S-Function 和控制器热启动。周期评估设置为每 6 回合一次，约对应 588 个策略交互；这与原 Python 的精确 550 步评估调度存在差异。保存每 10 回合的检查点、最终 Agent、完整训练统计、回报 CSV/PNG/FIG，并保留实际参数更新量。

Smoke Test 证明环境交互和梯度更新已发生，不代表策略收敛。正式结果和固定 MPC 的差异必须以同工况的实际回放评价；训练 300 回合也不等于证明优于固定 MPC。

## 数据与验证

`results/<controller>/raw.mat` 保存 MATLAB SimulationOutput；`trace.csv` 保存状态、误差、MPC 参数和 QP 状态；`metrics.csv` 保存 RMSE、峰值、失败次数与终点；`tracking.png/.fig` 保存曲线。三组比较位于 `results/comparison/`。`results/steering_authority.csv`、`interface_units.txt`、`source_audit.json` 和数值回放文件提供接口与来源证据。

完整 Agent、回放缓冲及 MAT 原始数据保留在本地；本目录的 `.gitignore` 排除这些大文件和编译缓存，CSV、图形和验收摘要可随源码版本化。

MATLAB MCP 初始 PID 为 38724，已用于模型读取、修复和实际接口仿真。该进程后续退出，MCP 无法重新附着；随后使用实际 MATLAB COM 会话继续模型构建和联合仿真。`utils/matlab_com.ps1` 是备用执行入口，输出追加至 `results/execution.log`。工具调用成功、Agent 创建成功和模型打开成功均不作为功能验收。

当前验证范围是本机 CarSim 2024.1、R2025a、该车辆及该 DLC SYX 工况。变路面、不同车辆、其他坐标道路和其他软件版本需重新验收。TQC 未实现。


