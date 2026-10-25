# CarSim 闭环中的 MPC 与 DDPG 调参

车辆、轮胎、道路附着系数、目标速度和参考轨迹均在 CarSim 工况中设置；C++ 只设置控制器与训练参数。每次修改 CarSim 工况后，应先在 CarSim 重新生成 Simfile / Parsfile，再运行 C++。

## MPC

在 `configs/default.ini` 调整：

- `[mpc]` 的 `np`、`nc`、`dt_s`：预测步数、控制步数和采样周期，满足 `1 <= Nc <= Np`。采样周期必须是 CarSim 求解器步长的整数倍。
- `[weights]` 的 `q_y`、`q_psi`、`q_vy`、`q_r`、`r_steer`、`r_delta`：横向误差、航向误差、侧向速度、横摆误差、转向指令和转角增量代价。
- `[constraints]` 的 `max_steer_deg` 和 `max_steer_rate_deg_s`：C++ 指令的幅值和变化率限制；需与 CarSim 前轮路轮转角 IMPORT 的实际单位核对。

MPC 预测模型的质量、转动惯量、轴距分量、轮胎侧偏刚度和路面摩擦系数来自本次 CarSim 求解器回显。速度每步从 CarSim 状态读取。`effective_config.ini` 记录本次使用的参数；`carsim_native/LastRun_echo.par` 和 `Run_all.par` 保存源数据。

## DDPG

`[rl]` 配置独立的 Np 范围、中性 Np、六个 Q/R 绝对数值范围、终端 Q 系数、训练轮次、速度与附着系数范围。默认 200 回合，30–120 km/h，μ=0.3–0.9。策略使用曲率、车速、横向状态和附着系数等观测，自适应选择 Np 及六个 Q/R 数值。修改固定 `[mpc] np` 或 `[weights]` 不会改变已训练 V15 策略在相同观测下的动作；`[rl_reward]` 定义误差、控制量、变化率、受限控制和失败等代价。

训练结果中检查 `training_rewards.csv`、`checkpoint_validation.csv` 和 `rl_report.html`。使用 `--rl-mode compare --rl-model <策略文件>` 在同一 CarSim 工况比较固定与自适应控制。一次训练不能证明策略在未训练道路或车辆上泛化，应通过另行设置的 CarSim 工况验证。
