# CarSim、C++ MPC 与 Python 的连接

`CarSimPlant` 通过 VS Solver DLL 读取车辆状态并写入前轮转角。C++ `LateralMpc` 使用六状态三自由度预测模型 `[vy,vx,psi,r,Y,X]`，以转角增量求解 MPC；实际车辆运动、道路与驾驶工况始终由 CarSim 计算。车辆与轮胎参数来自本次求解器回显。

`MPC_RL_Bridge` 每个进程负责一个回合。Python 可先发送 `PARAMS` 获取中性动作对应的 Np、权重、控制周期、决策步数、折扣因子和奖励尺度；发送 `RESET` 后，桥接程序返回 17 维观测。Python 随后发送 `STEP a_np a_qy a_qpsi a_rdelta`，四个动作均在 `[-1,1]`。C++ 依据配置把 Np 映射为整数、把 Qy/Qψ/RΔ 映射为范围内的正数；Qvy、Qr 和 Rsteer 保持 `[weights]` 中的设置。每次 `STEP` 执行 `decision_steps` 个 MPC 控制周期，返回新观测、由 C++ 计算的奖励、终止标记及诊断字段。`CLOSE` 结束进程。新的 Gymnasium 回合启动新的桥接进程，以隔离 CarSim DLL 状态。

默认配置为 `configs/carsim_python_rl.ini`，默认工况读取继承配置中的 `[carsim] simfile`，也可以显式指定。训练和评价启动时冻结 Simfile 和 Parsfile；桥接程序在每回合目录中复制快照并按目标车速设置停止时间，不修改源工况。车速不再固定为 60 km/h；模型只在训练工况得到验证，切换工况后应重新验证。当前参考轨迹解析、车辆和道路格式仍受 C++ 接口支持范围限制。

构建、训练和运行命令见项目根目录 [README.md](../README.md)。CarSim Browser 与求解器许可必须在运行桥接程序的同一 Windows 会话中可用。
