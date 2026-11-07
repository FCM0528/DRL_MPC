# CarSim、C++ MPC 与 Python 的连接

`CarSimPlant` 通过 VS Solver DLL 读取车辆状态并写入前轮转角。C++ `LateralMpc` 以转角增量求解 MPC，实际车辆运动、道路与驾驶工况由 CarSim 计算。默认固定 MPC 在三自由度车辆状态 `[vy,vx,psi,r,Y,X]` 上加入前后轴的滞后侧偏角，通过 CarSim 的 `AlphL_*` 和 `Fy_*` 初始化轮胎预测。配置项 `front_relaxation_m` 和 `rear_relaxation_m` 是零侧偏附近的等效松弛长度。两个值都为 0 时使用瞬时侧偏模型。

`MPC_RL_Bridge` 每个进程负责一个回合。Python 可先发送 `PARAMS` 获取中性动作对应的 Np、权重、控制周期、决策步数、折扣因子和奖励尺度。发送 `RESET` 后，桥接程序返回 17 维观测。Python 随后发送 `STEP a_np a_qy a_qpsi a_rdelta`，四个动作均在 `[-1,1]`。C++ 按配置量化 Np 并为 Qy、Qψ 和 RΔ 选择数组元素。Qvy、Qr 和 Rsteer 取 `[weights]` 中的设置。每次 `STEP` 执行 `decision_steps` 个 MPC 控制周期，返回新观测、C++ 计算的奖励、终止标记及诊断字段。`CLOSE` 结束进程。新的 Gymnasium 回合启动新的桥接进程，以隔离 CarSim DLL 状态。

默认配置为 `configs/carsim_python_rl.ini`，默认工况读取继承配置中的 `[carsim] simfile`，也可以显式指定。训练和评价启动时冻结 Simfile 和 Parsfile；桥接程序在每回合目录中复制快照并按目标车速设置停止时间，不修改源工况。车速不再固定为 60 km/h；模型只在训练工况得到验证，切换工况后应重新验证。当前参考轨迹解析、车辆和道路格式仍受 C++ 接口支持范围限制。

默认配置将 Np 量化到 10 至 50 的偶数，并限制每次决策最多改变 2 步。三个权重各自按归一化动作选择最接近的数组索引，零动作选择中间项。MPC 在整个决策区间使用同一组已执行参数，观测、奖励和 `bridge_trace.csv` 均以这些参数为准。旧配置没有候选数组时，权重继续使用对数映射及其 `smooth_tuning` 设置。旧策略应与训练时保存的配置配对。

构建、训练和运行命令见项目根目录 [README.md](../README.md)。CarSim Browser 与求解器许可必须在运行桥接程序的同一 Windows 会话中可用。
