# CarSim 2024.1 与 C++ MPC 闭环集成

本项目的 `CarSimPlant` 通过 CarSim VS Solver DLL 的 VS API 驱动车辆。MPC 预测模型采用《基于RL-MPC的智能车辆轨迹跟踪横向控制策略》式 (3)–(6) 的三自由度动力学（纵向速度、侧向速度、横摆角速度）和六状态 `[vy,vx,psi,r,Y,X]`，在 CarSim 当前状态处线性化并按前向欧拉离散。论文的全局位置记号与 CarSim 的 X 前向、Y 向左坐标对应时需要调整顺序。质量、惯量、轴距、轮胎侧偏刚度和附着参数来自本次 CarSim Solver 回显；当前接口没有纵向轮胎滑移与刚度输出，因此纵向力在工作点由相邻 CarSim 纵向速度求得的加速度表示，不填写未知轮胎参数。实际车辆状态、车辆运动由 CarSim 计算。固定 MPC、DDPG 训练和 DDPG 策略评估均可接入 CarSim。

## 1. 在 CarSim 中生成可控工况

当前提供的 `D:\Program Files (x86)\CarSim2024_Date\simfile.sim` 对应 **C-Class Hatchback**。最近生成的 Simfile 写有 `PORTS_IMP 1,2`、`PORTS_EXP 1,15`。其 `Run_all.par` 中导入通道顺序是 `IMP_STEER_L1`、`IMP_STEER_R1`，均为 `Replace`，因此 C++ 应配置左右索引 0 和 1。CarSim 工况文件随重新运行可能覆盖，运行前请再次核对。

在 CarSim Browser 的 Run Control / I/O Channels: Import 中维护**前轮路轮转角** `Replace` IMPORT，并记下从 0 开始的通道顺序。不要把方向盘角或转向力矩通道当作前轮路轮转角；它们需要不同的映射和单位。左右轮同时由 C++ 指令控制时，两通道均需激活。保留所需的纵向驾驶员；重新运行工况后核对 Simfile 和生成的 `Run_all.par`。

当前 C++ 程序会从 CarSim 生成的 `Run_all.par` 中读取 `LTARG_TABLE SPLINE_FLAT`，以表中采样点构造三次样条，供 MPC 预测、误差计算和图表共用。当前适配仅接受**全局 X 方向的直线道路（Straight East）**，此时 CarSim 的道路纵向站位和横向偏移可对应全局 X/Y；换成弯道或其他朝向时会报错，需提供道路坐标到全局 X/Y 的转换。插值在采样点处与 CarSim 表一致，采样点之间不保证与 CarSim 内部样条逐点完全相同。CarSim 纵向驾驶员控制实际车速。

## 一次仿真的顺序

1. **CarSim：车辆与道路。** 在 Run Control 选择车辆和道路；目前是 C-Class Hatchback 与 Straight East 路。配置车辆参数、轮胎、附着和初始速度。
2. **CarSim：速度与参考轨迹。** 在 Procedure / Speed Controller 中设置目标速度及仿真结束条件，在 Steering by Closed-loop Driver Model 的 `LTARG_TABLE` 中设置期望横向路径。每次修改后检查新生成的 `Run_all.par` 里的 `SPEED_TARGET_CONSTANT`，并以求解器输出的 `Vx` 核对实际运行速度；这两个值不能互相代替。
3. **CarSim：外部控制接口。** 保留左右前轮 `IMP_STEER_L1`、`IMP_STEER_R1` 的 Replace 导入，顺序为 0、1。编辑完成后在 CarSim 运行一次，生成新的 `simfile.sim` 与 `Run_all.par`。
4. **C++：控制器。** 设置 MPC 的 `Np`、`Nc`、`dt_s`、Q/R 权重、转角和转角速率约束。CarSim 模式下，MPC 从本次求解器回显文件读取整车总质量、总质心、偏航惯量、道路摩擦系数和四轮侧向力表，按静态轴荷与小侧偏角计算预测模型参数；缺少这些数据或四轮侧向力表不同会停止并报错。C++ 不修改 CarSim 的速度设定；预测模型每周期使用求解器返回的当前 `Vx`。
5. **C++：启动闭环。** 运行下方命令。每个控制周期，C++ 读取 CarSim 状态和 CarSim 表定义的参考轨迹，计算左右前轮转角；CarSim 推进车辆后，C++ 记录状态并生成 `report.html`、`simulation.csv` 与 CarSim 原生输出文件。

只修改 C++ 控制参数时无需重新运行 CarSim Browser；修改 CarSim 的车辆、道路、速度或参考轨迹后，必须让 CarSim 重新生成 Simfile 和参数文件，再启动 C++。

## 本机文件与职责

| 文件 | 用途 | 是否复制到 C++ 项目 |
| --- | --- | --- |
| `D:\Program Files (x86)\CarSim2024_Date\simfile.sim` | 当前工况入口，记录输入文件、求解器路径和 I/O 通道数 | 同一台机器用绝对路径读取即可 |
| `D:\Program Files (x86)\CarSim2024_Date\Results\Run_6cb07365-85a3-4a3b-95f9-c5c844687601\Run_all.par` | 本次生成的车辆、道路、工况、驾驶员和参考轨迹参数 | 保持在 Simfile 指向的位置 |
| `D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll` | 64 位 CarSim VS Solver | 用绝对路径加载，保留原安装目录 |
| CarSim 数据库内的车辆、道路、轮胎和动画等资源 | 工况编辑、运行和可视化所依赖的数据 | 保留 CarSim 数据库；单独复制两三个文件不保证可运行 |

`SDK_External_Control.cpar` 是工况归档，用于在 CarSim 数据库导入示例，不是 C++ 每一步仿真直接读取的文件。

## 2. 配置项目

`configs/default.ini` 已配置 `simulation.plant = carsim`。核对下列路径和接口配置，固定 MPC 运行时使用 `--rl-mode fixed`：

```ini
[carsim]
simfile = D:\Program Files (x86)\CarSim2024_Date\simfile.sim
solver_dll = D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll
sync_browser_result = true
steering_left_index = 0
steering_right_index = 1
steering_gain = 57.29577951308232
x_keyword = XCG_TM
y_keyword = YCG_TM
yaw_keyword = YAW
vx_keyword = VX
vy_keyword = VY
yaw_rate_keyword = AVZ
```

`steering_gain` 把 MPC 的前轮转角弧度换算成 CarSim IMPORT 单位。示例值对应度。若存在第二个前轮转角 IMPORT，将 `steering_right_index` 设为其索引；这两个通道目前会收到相同指令。状态关键词须在当前车辆模型中可用；不同 CarSim 数据集可能需要调整。MPC 的质量、轴距分量、偏航惯量、轮胎表和路面摩擦系数随本次 CarSim Solver 回显更新；预测模型是小侧偏角三自由度近似，包含纵向、侧向和横摆耦合；纵向驱动力仍由 CarSim 驾驶员控制，不等同于完整 CarSim 车辆动力学。

`sync_browser_result = true`（默认值）会在每次 C++ 闭环结束并关闭求解器后，把该次结果复制到源 Simfile 的 `FILEBASE` 所指向的 CarSim Run Control 结果目录。这样该 Run Control 的 **Video / Video + Plot / Plot** 可以在点击 **Refresh** 后读取本次结果。同步前会把原有 Browser `LastRun` 文件备份到该次独立结果目录下的 `carsim_native/browser_previous_cache/`；每次独立结果始终保留。若 CarSim 的源 `Run_all.par` 在运行期间发生变化，同步会拒绝执行并在报告中注明，避免新旧工况错配。若只需独立回放，可设 `--set carsim.sync_browser_result=false`。

也可以通过命令行临时指定（PowerShell）：

```powershell
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode fixed --plant carsim --set 'carsim.simfile=D:\Program Files (x86)\CarSim2024_Date\simfile.sim' --set 'carsim.solver_dll=D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll' --set carsim.steering_right_index=1 --no-open --out results/carsim_trial
```

若需要在**同一 CarSim 工况下训练自适应 Np、Q 和 R 策略**，先运行：

```powershell
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --plant carsim --set 'carsim.simfile=D:\Program Files (x86)\CarSim2024_Date\simfile.sim' --set 'carsim.solver_dll=D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll' --set carsim.steering_right_index=1 --no-open --out results/carsim_training
```

默认训练 200 回合，目标速度在 30–120 km/h、路面附着系数 μ 在 0.3–0.9 的 6×5 分层网格内抽样，并增加低速过渡区的训练覆盖。每回合运行真实 CarSim Solver，在私有 `training_solver/Run_all.par` 中设置 `SPEED_TARGET_CONSTANT`、`MU_ROAD_CONSTANT`，必要时延长 `TSTOP`；原工况不修改。求解器回显会核对设定速度和 μ，实际车速以求解器输出为准。`training_rewards.csv` 保存逐回合速度、μ、回报、RMSE 和完成情况；每 10 回合在原工况及十三个速度/μ 组合验证，`checkpoint_validation.csv` 保存各工况指标。CarSim 轨迹的当前位置与前方曲率会输入策略，但当前训练只使用这一条 CarSim 参考轨迹。

DDPG Actor 每次输出七个有界动作：Np、`Q_y/Q_psi/Q_vy/Q_r` 四个绝对数值、`R_steer/R_delta` 两个绝对数值。Np 范围与中性值及六个 Q/R 数值范围在 `[rl]` 中单独配置，与固定 MPC 的 `[mpc] np` 和 `[weights]` 解耦。奖励对横向误差、航向误差、侧向速度、横摆速率误差、转角与转角变化赋予主要代价，并对较长 Np、动作跳变、权重偏离独立范围中心、受限控制、求解失败和未完成工况计入代价。精确权重及归一化参数在本次 `effective_config.ini` 和 `rl_report.html` 中核对。Q/R 的变化本身不会获得奖励。

训练完成后，取终端输出目录中的 `ddpg_policy.txt` 进行**同一 CarSim 工况下的固定 MPC 与 DDPG 自适应 MPC 对比**：

```powershell
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode compare --rl-model 'results\carsim_training\<本次运行子目录>\ddpg_policy.txt' --plant carsim --set 'carsim.simfile=D:\Program Files (x86)\CarSim2024_Date\simfile.sim' --set 'carsim.solver_dll=D:\Program Files (x86)\CarSim2024.1_Prog\Programs\solvers\carsim_64.dll' --set carsim.steering_right_index=1 --no-open --out results/carsim_trial
```

该命令在每次独立结果目录下生成 `fixed/`、`adaptive/`、`comparison/`、`comparison_metrics.csv` 和 `comparison_report.html`。两次运行使用相同的 CarSim 工况快照；报告给出 Np、误差、转向、速度曲线、预测模型融合权重，以及六个 Q/R 绝对值曲线。`adaptive/simulation.csv` 包含每个控制周期实际使用的六个 MPC 权重和模型融合权重。固定 MPC 原生结果保存在 `fixed/carsim_native/`，DDPG 原生结果保存在 `adaptive/carsim_native/`。若启用 Browser 同步，最近的 GUI Video/Plot 对应 DDPG 运行。策略文件作为 `ddpg_policy_used.txt` 单独存档。V14 倍率策略不能直接加载；当训练时基准权重与绝对动作边界满足等价条件，可使用 `tools/convert_v14_policy.py` 迁移 Actor 网络，不满足条件则需重训。

上述命令需要正常可用的 CarSim Solver 许可证。`mpc.dt_s` 必须是 VS API 返回的求解步长的整数倍；当前 Simfile 的 `EXT_MODEL_STEP` 是 0.0005 s，而生成的 `Run_all.par` 写有 `tstep 0.001` s，不能将这两个字段混为一谈。程序以 VS API 返回值检查步长；项目默认 `dt_s` 为 0.05 s。

CarSim 模式下，`--out results/carsim_trial` 指定**结果根目录**。每次运行都会在其下建立带时间和 MPC 时域参数的独立子目录，同一秒内再次运行会追加序号；终端会打印该次结果和报告的完整路径。省略 `--out` 时也使用独立子目录。文件夹名称不再使用 C++ 模型的名义速度或轮胎力倍率冒充 CarSim 工况值。

## 3. 查看结果

`report.html` 和 `simulation.csv` 保留横向轨迹、航向角、误差、CarSim 求解器输出的车速与横摆角速度、MPC 转角指令及求解耗时。速度图、表、CSV 和终端输出统一使用 km/h；MPC 内部仍使用 m/s。车辆、道路、工况和轨迹标题来自本次 `carsim_native/Run_all.par`；速度设定的 km/h 单位与整车质量、质心、惯量、轮胎表和道路摩擦参数由本次 `carsim_native/LastRun_echo.par` 确认。CarSim 模式下，轮胎侧偏角、轴侧向力和实际前轮轮角没有映射，因此报告不将 C++ 的轮胎估计写成 CarSim 实测值，图中的转角是**指令**。`effective_config.ini` 中的质量、惯量、轴距分量、轮胎 CSV 和摩擦系数对应本次生成的 MPC 预测模型。修改工况后需在 CarSim 重新生成 Simfile 和 `Run_all.par`，再运行 C++。

启动时，C++ 会在该次结果目录的 `carsim_native/` 中生成私有的 `simfile.sim` 和求解器用 `Run_all.par`：将相对 `INPUT` 改为绝对路径，并让 `FILEBASE`、`ECHO` 等原生输出先写入该目录。CarSim 的 `LastRun.vs`、`LastRun.vsb`、日志和 Echo 文件永久保留在该次独立目录；Browser 的 `LastRun` 是供 GUI 按钮读取的最近一次副本。另保留**未经删改**的 `Run_visualizer.par`，供原生可视化读取。当前适配器通过 `vs_get_var_ptr` 获取车辆状态，没有消费 EXPORT 数组；仅求解器用的私有参数副本会去掉原工况中的 EXPORT 定义，以规避当前工况 `X_Drv_1` 在直接 SDK 运行时无法作为 EXPORT 的错误。CarSim 的动画和绘图配置仍保存在 `Run_visualizer.par`，原始 CarSim 工况参数文件不变。

要回放这次 **C++ 闭环**的 CarSim 画面和绘图，可以在对应 CarSim Run Control 点击 **Refresh**，再点击 **Video / Video + Plot / Plot**；前提是报告写明 Browser 缓存同步成功。也可以在资源管理器中双击该次独立目录的 `carsim_native/open_visualizer.cmd`。它调用安装目录中的 `Programs/VsVisualizer/VsVisualizer.exe` 并加载同目录的 `visualizer.par`；这个入口按 CarSim Browser 生成的格式，把完整 `Run_visualizer.par` 与 `LastRun.vs` 一起交给 VS Visualizer。`LastRun.vsb` 必须与 `.vs` 保持同目录。需要仿真过程中的实时动画，可在 Run Control 配置 live animators；本项目目前验证的是每次闭环结束后的原生回放。
