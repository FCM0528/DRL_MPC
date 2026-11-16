# MPC 双误差可行性与动作消融（2026-11-16）

本次变更文件：`src/config.cpp`、`src/mpc.cpp`、`src/mpc.h`、`src/rl.cpp`、`src/rl.h`、`src/rl_bridge_main.cpp`、`src/visualization.cpp`、`src/visualization.h`、`python/envs/carsim_mpc_env.py`、`python/evaluate.py`、`python/joint_feasibility.py`、`python/joint_selection.py`、`python/parity.py`、`python/scenario.py`、`python/train.py`、`python/train_four.py`、`python/tests/test_joint_action_dims.py`、`python/tests/test_joint_feasibility.py`、`python/tests/test_scenario.py`、`tests/verify.cpp`、三份 `configs/carsim_python_rl_joint_actions{2,3,4}.ini` 及本文档。扫描输出仅在新建 `runs/joint_feasibility_20261116_104436/` 等独立目录。

## 已确认的代码问题及修复

1. `src/rl.cpp` 的 `joint_tracking` 曾忽略 `w_lateral_growth`、`w_np_change`、`w_weight_change` 等已配置项。现由 `jointRewardTerms` 分项计算，再由标量奖励取总和；旧奖励分支保留。`bridge_trace.csv` 记录横向、车身航向、航迹角、峰值、增长、转向、参数切换、安全、其他及总损失；`decision_reward.csv` 单独列出 QP 和终止惩罚。真实 CarSim 零动作轨迹的逐步与决策级分解残差均为 0。
2. 原桥接 `requested_np/q_y/q_psi/r_delta` 实际是归一化动作。现明确列为 `action_*_normalized`，并记录未受每决策变化限制的 `target_*`；原 `np/q_y/q_psi/r_delta` 是实际执行参数。用 `[0,1,1,1]` 实测：Qy 目标 20240、执行 12420；Qpsi 目标 11200、执行 3024，符合 1.08 的逐决策比率限制。
3. `final_model.zip` 只作为诊断样本写入验证 CSV，不进入合格检查点或 Pareto 集。三算法批量训练若任一无合格模型，套件状态为 `no_joint_qualified_model`，不会用 final_model 冒充最优模型。
4. 训练、验证、测试源 Simfile 与 Parsfile 均按完整文件 SHA-256 核对；任一文件内容相同即拒绝划分。运行前将各工况复制到独立目录，并记录源哈希。现有实验仍只包含单工况，尚无独立验证/测试结论。
5. 固定 MPC 的 `simulation.csv` 新增九项预测 QP 代价贡献：Qy、Qcourse、横向误差率、横摆角速度误差、前后轮胎侧偏、Rsteer、Rdelta、Rdelta_change。代价在求解并投影实际控制序列后，按优化时的线性化预测模型计算，不等于实测车辆损失。

## 真实 CarSim 搜索

运行目录：`runs/joint_feasibility_20261116_104436/`。源 Simfile SHA-256 为 `9b2f81682b4e350e69d2c820923265faabcfc5f3a9157756d00c884716f3ab6e`；Parsfile SHA-256 为 `bb878204652e1351e0fa30e7c6c66654ff61539475e4d06846acbacfc198be7e`；固定配置 SHA-256 为 `7853cb9c2799366faa2eba6c73ca70862d802173901572b4b5c4055245ce8101`。原始固定 MPC 为 Np=40、Nc=13、Qy=11500、Qpsi=2800、Rdelta=10800，横向 RMSE 0.087240 m、车身航向 RMSE 1.014074°。

粗扫 **45/45** 完赛、零 QP 失败且满足峰值侧向加速度 ≤0.85 g、峰值侧偏角 ≤5°；其中 **9 组同时改善**。局部细扫再完成 26 组，均符合上述门槛且双指标改善。完整结果与 20 个非支配候选见运行目录中的 `feasibility_metrics.csv`、`feasible_candidates.csv`、`pareto_candidates.csv`、`weight_contribution.csv` 和 `pareto_front.png`。

| 设置 | 横向 RMSE | 航向 RMSE | Iy | Iheading | min(Iy,Iheading) |
|---|---:|---:|---:|---:|---:|
| 原固定 MPC：40/11500/2800 | 0.087240 m | 1.014074° | 0 | 0 | 0 |
| 粗扫最佳平衡：36/18400/11200 | 0.078913 m | 0.920231° | 9.55% | 9.25% | 9.25% |
| 局部最佳平衡：36/18400/8960 | 0.078953 m | 0.919092° | 9.50% | 9.37% | 9.37% |

以上是同一工况上的参数搜索表现，不是独立测试或 DRL 改善。

## 弱作用维度

原固定 MPC 的预测 QP 代价均值：Qy=4311.68、Qcourse=58.98、Rsteer=619.02、Rdelta=10.81、Rdelta_change=349.34。Qcourse 相当于 Qy 的 1.37%；Rdelta 相当于 Rdelta_change 的 3.1%。在 Np=36/Qy=18400 下，Qpsi 由 2800 增至 11200 时转角指令 RMS 差为 0.00162 rad，说明有影响，但相对较小。`track_course=true` 优化的是航迹角而非车身航向角，可能限制其对航向 RMSE 的收益；尚不能单独认定这是因果原因。

同一冻结工况下，Rdelta 扩展到 1080/10800/54000，横向 RMSE 为 0.087178/0.087240/0.087259 m，航向 RMSE 为 1.019078/1.014074/1.018377°；均完赛、零 QP 失败。Rdelta 代价变化明显，输出却很弱，符合 Rdelta_change 和 Rsteer 项相对更强的解释。最大实际转角指令约 0.096 rad，远低于 CarSim 转角界限约 0.532 rad，因此绝对转角饱和不是当前弱响应的主要原因。轮胎约束、前馈和预测状态敏感度尚待独立消融。

局部最佳静态参数与一个道路需求 Np 调度器（32/36）比较：调度器 38 步使用 32、119 步使用 36，横向/航向 RMSE 为 0.0789534 m/0.9191002°，静态为 0.0789528 m/0.9190925°。这个调度器没有带来可观额外收益；不能据此推断任何 DRL 策略无收益。

## 下一步动作实验

`configs/carsim_python_rl_joint_actions2.ini`、`joint_actions3.ini`、`joint_actions4.ini` 分别提供 Np+Qy、Np+Qy+Qpsi、Np+Qy+Qpsi+Rdelta 的 2/3/4 维策略动作。三份配置使用相同奖励、20 维观测和扩展权重范围；固定 MPC 中性参数仍为原基线。2/3 维策略在 Python 端补零送入四动作桥接，旧四维模型不兼容。三份配置各自通过 20 控制步真实 CarSim 零动作 parity。当前证据支持先做 2 维，再做 3 维，最后用 4 维检验 Rdelta 的增益；Qpsi 和 Rdelta 均未从代码中删除。训练必须同时比较原固定 MPC、优化后静态 MPC 和独立测试工况。

```powershell
uv run --no-project --with-requirements python/requirements.txt python python/joint_feasibility.py --fine
uv run --no-project --with-requirements python/requirements.txt python python/train_four.py --config configs/carsim_python_rl_joint_actions2.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --validation-simfile <不同内容的验证.sim> --test-simfile <不同内容的测试.sim> --out runs/pilot_actions2
```

将 `actions2` 分别替换为 `actions3`、`actions4`，并使用新的独立输出目录。正式多种子训练尚未启动；不得把本次静态参数搜索结果解释为算法排名。
