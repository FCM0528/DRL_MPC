# 算法来源与 Simulink 接口

| 功能 | 原文件与入口 | 迁移实现 |
| --- | --- | --- |
| 轮胎表、载荷插值、饱和力与峰值 | `src/vehicle.cpp`：`TireTable` | 保留同名源码副本；不使用另行拟合的轮胎模型 |
| 三自由度及轮胎松弛状态 | `discretizeErrorModel` | 原实现；六个物理状态加两个松弛状态 |
| Tustin 离散化 | `discretizeErrorModel` | 原实现；逐预测阶段执行 |
| 名义轨迹和灵敏度递推 | `LateralMpc::precompute` | 原实现与等价的 `PipelineMpc` 阶段 |
| 完整误差/前馈/转角/增量/增量变化/轮胎侧偏代价 | `LateralMpc::solve` | `PipelineMpc::stageCost`，由原代码机械提取 |
| 幅值、速率约束 | `LateralMpc::buildBounds` | `PipelineMpc::stageConstraints` |
| ADMM、Cholesky、收敛判据 | `LateralMpc::solve` | `PipelineMpc::stageQp` |
| 转角投影与热启动 | `LateralMpc::solve`、`adoptPreviousSolution` | `stageSteering`、`adoptPreviousSolution` |
| LTARG 全表、自然三次样条、平坦外延 | `src/reference.cpp` | 原实现；不调用独立双移线轨迹发生器 |
| 前馈曲率空间平滑 | `DoubleLaneChange::controlCurvature` | 原实现；跟踪目标仍为原始样条 |
| 车辆参数、轮胎表、路面附着 | `readCarSimMpcModel` | 读取当前私有求解器回显；缺失参数报错，不使用默认车辆替代 |
| 转角约束 | `readCarSimSteeringModel` | 从回显的当前转向机构推导 |
| 14 维道路预瞄观测 | `makeRoadPreviewObservation` | MEX 直接调用原函数 |
| Np 与离散 Q/R 映射 | `advanceRlTuning`、`applyRlAction` | MEX 直接调用原函数 |
| tracking_v2 单步奖励 | `scalarizedRlReward` | MEX 直接调用原函数 |
| 两个 MPC 步的奖励聚合 | `rlDecisionReward` | `migration_sfun` 中按两个 0.04 s 步聚合，并在 0.08 s 决策点输出 |
| CarSim 实际动力学推进 | 原 `CarSimPlant` | 由现有 `vs_sf` 替代；不构造原 `CarSimPlant` |
| RL Agent 与训练 | Python SB3 网络和训练配置 | MATLAB RL Toolbox DDPG/TD3，配置取 `matlab/configs/rl_training.ini` |

`PipelineMpc` 的七个阶段通过 MEX 内部上下文共享名义轨迹、Hessian、梯度、约束和 QP 结果。阶段间 Simulink 信号显式规定调用依赖；每个阶段均执行对应计算。预测模型阶段初始化测量状态、模型参数和暖启动控制器；线性化阶段使用该车辆模型滚动预测。阶段上下文每回合重置，不能并行驱动同一个 MEX 上下文。

单步奖励、观测和预测状态的数值一致性由独立链接原 `src/` 的可执行程序验证；独立程序不调用 CarSim 动力学求解器。在线调参回放覆盖当前 Np 下限、中性值、上限及 22（本次为 12/40/50/22）和全部权重候选，七阶段版本与原程序各列的差异为零。MEX 对照另外允许 CSV 十进制往返和独立编译造成的浮点残差，以 `results/*parity*` 为准。

以下差异需在解释实验时保留：

- 本次共同评价终点由原默认配置的 120 m 改为任务要求的 150 m，仅修改迁移副本。
- 私有 `SSTOP=160` 是原 C++ 站位余量机制；Simulink 按质心 X=150 结束评价。
- MATLAB 周期评估每 6 回合一次，原 Python 按 550 次决策调度，二者并非相同计时机制。
- MATLAB 和 PyTorch 的随机初始化、优化器数值细节及 GPU 运算不能保证训练轨迹逐步相同；这不属于 MPC 数值一致性结论。
- Simulink/CarSim S-Function 的硬运行错误会停止训练并保留已有检查点；正常的 QP 失败与受控提前终止进入原奖励惩罚。
- 本次配置中的硬转向速率限制关闭。转向增量及其变化的代价仍按原配置保留。

