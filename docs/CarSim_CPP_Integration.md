# CarSim、C++ MPC 与 Python 的连接

`CarSimPlant` 通过 VS Solver DLL 读取车辆状态并写入前轮转角。C++ `LateralMpc` 以转角增量求解 MPC，实际车辆运动、道路与驾驶工况由 CarSim 计算。默认固定 MPC 在三自由度车辆状态 `[vy,vx,psi,r,Y,X]` 上加入前后轴的滞后侧偏角，通过 CarSim 的 `AlphL_*` 和 `Fy_*` 初始化轮胎预测。配置项 `front_relaxation_m` 和 `rear_relaxation_m` 是零侧偏附近的等效松弛长度。两个值都为 0 时使用瞬时侧偏模型。

当前 Run Control 输入为 `IMP_STEER_L1`、`IMP_STEER_R1`，零基索引为 0、1，均采用 `REPLACE` 和 `deg`。C++ 输出的是等效前轮角，转换为度后写入这两个通道。初始化时始终核对输入名称、索引、模式和单位，包括手动配置转角上限的情况。额外输入保持回显中声明的初值；需要随时间变化的其他输入仍须另行实现控制逻辑。

工况暂存会保留原始 `EXPORT` 与 `PORTS_EXP`，支持原来的两路轮角导出和当前二十路导出，后续增减或重排导出也不改变反馈绑定。反馈通过 `vs_get_var_ptr` 按关键词读取 **SI 值**：位置 m、速度 m/s、角度 rad、角速度 rad/s、力 N。实际轮角读取 `STEER_L1`、`STEER_R1`，不再将导出数组前两项解释成轮角。Run Control 导出数组的显示单位不同，例如 `Vx` 为 km/h、`Yaw` 为 deg、`Ax` 为 g，不能直接作为 MPC 状态使用。`VehicleState.ax` 仍取相邻车体纵向速度之差除以实际积分时长；它与 CarSim 加速度计通道 `Ax` 的物理含义也有区别。

当前二十路导出顺序为：`Xcg_TM, Ycg_TM, Vx, Vy, Yaw, AVz, Ax, Ay, Steer_L1, Steer_R1, Beta, Rho_Road, AlphL_L1, AlphL_R1, AlphL_L2, AlphL_R2, Fy_L1, Fy_R1, Fy_L2, Fy_R2`。轨迹预瞄仍读取同一工况的 `LTARG_TABLE`；当前位置的 `Rho_Road` 不足以替代未来轨迹。

其中 `AlphL_*` 与 `Fy_*` 已用于 MPC 轮胎松弛状态和近端轴力残差；`Beta` 现优先用于策略航迹角观测与稳定性奖励，未导出时回退到 `atan2(vy,vx)`。Ax/Ay、Beta、Rho_Road、前后轴侧向力和实际轮角均以 SI 值写入固定与桥接日志。四组指标新增实测侧向加速度与侧偏峰值、轮角执行差、含旋转坐标项的速度/加速度一致性和轮胎力平衡残差；缺失的可选传感量留空，不当作零值。当前工况 `Rho_Road=0`，而双移线目标曲率非零，两者没有混用于预瞄。

`MPC_RL_Bridge` 每个进程负责一个回合。Python 可先发送 `PARAMS` 获取中性动作对应的 Np、权重、控制周期、决策步数、折扣因子和奖励尺度。发送 `RESET` 后，桥接按配置返回当前20维 `road_preview_joint` 或兼容的14维 `road_preview_v2`、12维 `road_preview`。Python按 `policy_action_names` 将低维策略动作补齐为 `STEP a_np a_qy a_qpsi a_rdelta`，未学习维度严格填零。C++按配置量化Np，并使用legacy五档或continuous_weights映射权重。Qvy、Qr和Rsteer维持配置值。每次STEP执行 `decision_steps` 个控制周期，返回观测、奖励、终止标记及诊断字段。CLOSE结束进程；新回合重启桥接以隔离DLL状态。

默认配置为 `configs/carsim_python_rl.ini`，默认工况读取继承配置中的 `[carsim] simfile`，也可以显式指定。训练和评价启动时冻结 Simfile 和 Parsfile；桥接程序在每回合目录中复制快照并按目标车速设置停止时间，不修改源工况。车速不再固定为 60 km/h；模型只在训练工况得到验证，切换工况后应重新验证。当前参考轨迹解析、车辆和道路格式仍受 C++ 接口支持范围限制。

固定 MPC 和桥接程序使用 `configs/default.ini` 中的同一终点 X = 120 m。两者在质心首次达到或越过该位置时结束，并按控制后的状态样本计算跟踪误差。对照评估核对两边 `effective_config.ini` 中的终点。

旧 `carsim_python_rl.ini` 保留Np范围28–52、中性40及五档映射。当前强基线profile以Np36、Qy18400、Qpsi8960、Rdelta10800为中性点，Nc13；权重连续映射并限速1.08，full4另限制Np每决策变化2。桥接CSV的 `action_*_normalized`、`target_*` 和实际参数列含义不同，分别对应请求坐标、映射后的目标和限速后的执行值。旧模型必须使用其冻结配置；同维度策略不能据此跨动作语义或 `track_course` 目标加载。

四组比较会同时冻结 CarSim 工况、固定 MPC 配置和 DRL 配置的完整继承链，避免回放期间修改源配置造成对比失真。Windows 下建议使用 `runs` 等较短输出根目录；CarSim 原生文件及训练套件嵌套目录过深时会触发路径长度限制。

构建、训练和运行命令见项目根目录 [README.md](../README.md)。CarSim Browser 与求解器许可必须在运行桥接程序的同一 Windows 会话中可用。
