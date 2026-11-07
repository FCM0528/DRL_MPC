# CarSim + C++ MPC + Python 强化学习

CarSim 提供车辆动力学、道路和驾驶工况。C++ 从 CarSim 读取状态、计算增量式 LTV-MPC 前轮转角并写回求解器。Python 使用 Stable-Baselines3 / SB3-Contrib 训练 DDPG、TD3 或 TQC，在每次决策时经 `MPC_RL_Bridge` 向 C++ 发送 Np、Qy、Qψ、RΔ 四个归一化调参动作。C++ 将动作映射到配置的时域与权重范围，再执行控制。

默认读取所选配置中 `[carsim] simfile` 指向的当前 CarSim 工况；修改 CarSim 工况后请先生成或更新该 Simfile。也可用 `--simfile` 显式指定。运行 CarSim 求解器需要在同一 Windows 会话中打开 CarSim Browser 并具备可用许可。

默认固定 MPC 和 Python DRL 训练使用 `distance_m = reference_end`，仿真在 CarSim 质心首次达到当前 `LTARG_TABLE` 末端时结束。固定 MPC 可用 `--distance` 显式指定结束 X。

默认固定 MPC 使用质心运动方向跟踪路径切线，并用 CarSim 实测侧偏和侧向力预测前后轴的轮胎力建立过程。横向误差按同一全局 X 处的 `Y - Yref` 计算。Python 配置单独指定已有策略的时域、轮胎预测方式、权重和转向约束。对照实验请核对 CarSim 快照及 `effective_config.ini`。

直接运行 `MPC_DoubleLaneChange.exe` 或指定 `--config configs/default.ini` 即可使用固定 MPC。控制器、CarSim 与 Visualizer 共用当前工况 Parsfile 中的 `LTARG_TABLE`。默认固定 MPC 关闭 Browser 缓存回写，结果目录包含对应的 Visualizer 入口。

固定 MPC 的转向角坐标始终是前轮角。默认直接前轮通道从本次 CarSim 的 `RACK_KIN` 表取左右前轮共用的对称角度范围。若 CarSim 工况改为单个 `IMP_STEER_SW REPLACE`，设置 `carsim.steering_input=steering_wheel` 和 `steering_right_index=-1`，并将 `steering_left_index` 指向该输入。对支持的内部齿条转向系统，程序用实际的 `RACK_TRAVEL` 和左右轮运动学反算方向盘角。速率默认由转角增量代价调节，已知的前轮执行器限值可在 `[constraints]` 显式填写。历史高速实验记录见 [高速工况核验](docs/High_Speed_Validation.md)。

控制器始终使用非线性轮胎表的工作点力与局部斜率；历史配置中的 `nonlinear_tire=true` 仍可读取，`false` 会报错。历史线性轮胎模型得到的训练或评价结果不应直接视为本模型结果。

## 参数配置与调整

| 文件 | 修改内容 | 主要字段 |
| --- | --- | --- |
| `configs/default.ini` | 固定 MPC 与 CarSim 基础设置 | `[mpc] np/nc/dt_s`、`[weights]`、`[constraints]`、`[solver]`、`[carsim] simfile` |
| `configs/carsim_python_rl.ini` | Python 策略调用 C++ 桥接时的控制与奖励设置；继承 `default.ini` | `[rl]` 的 Np 和 Q/R 动作范围、`decision_steps`、`discount_gamma`，以及 `[rl_reward]` |
| `configs/rl_training.ini` | Python 神经网络训练设置，不由 C++ 读取 | `[training]` 通用参数；`[ddpg]`、`[td3]`、`[tqc]` 各自的 Actor/Critic 学习率、网络、目标更新与探索设置 |

在 `default.ini` 的 `[weights]` 调整固定 MPC 权重。`track_course = true` 时，角度代价比较 `ψ + atan2(vy, vx)` 与路径切线，图中的航向误差仍按车身朝向计算。结果目录提供横向误差 MAE、RMSE、最大绝对值和参考曲率。策略的离散候选值位于 `carsim_python_rl.ini` 的 `[rl]`。Np 取 10 至 50 的偶数，三个自适应权重的中性值取各数组的中间项。车辆参数与轮胎数据来自 CarSim 求解器回显。训练和评价使用同一份配置。

例如只调 TQC 训练：编辑 `configs/rl_training.ini` 后运行 `python python/train.py --algorithm tqc`；临时覆盖可用 `--episodes 100 --seed 43 --eval-frequency 500 --tqc-profile practical --tqc-drop 2`。指定另一个训练配置用 `--train-config <文件>`，指定另一组控制器参数用 `--config <文件>`。三算法顺序训练 `train_four.py` 同样支持 `--train-config`。每次训练会把配置文件和源工况复制进结果目录，避免运行中修改原文件影响该次实验。

DDPG、TD3、TQC 的逐项训练参数表和单位见 [强化学习训练参数](docs/RL_Training_Parameters.md)。采样周期来自 `carsim_python_rl.ini` 所继承的 `[mpc] dt_s`，不是 Python 优化器参数。

固定 MPC 的 `r_delta_change` 软惩罚相邻两步的转角增量差。报告将 MPC 转角指令与 CarSim 输出的左右前轮实际角度分别绘制。横向误差、横摆角速度、实际前轮角均报告绝对值峰值和平均值。车轮峰值取左右前轮中的较大值，均值汇总两轮所有样本。

`reference_curvature_sigma_m` 只作用于转向前馈的空间曲率估计。位置跟踪、误差统计和原生几何曲率均来自同一张 CarSim `LTARG_TABLE`。报告分别提供两种曲率口径。

## 构建

在项目根目录运行，按本机 Visual Studio 版本调整 CMake 路径和生成器：

```powershell
$cmake = 'D:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmake -S . -B out/python-rl-build -G 'Visual Studio 18 2026' -A x64
& $cmake --build out/python-rl-build --config Release --target MPC_RL_Bridge MPC_DoubleLaneChange mpc_tests
& "$PWD\out\python-rl-build\Release\MPC_RL_Bridge.exe" --self-test-np
```

## Python 环境与桥接检查

```powershell
conda activate drl_mpc
pip install -r python/requirements.txt
python python/test_bridge.py --exe out/python-rl-build/Release/MPC_RL_Bridge.exe --config configs/carsim_python_rl.ini --simfile 'D:\Program Files (x86)\CarSim2024_Date\simfile.sim'
```

`--config` 可指定其他 MPC 配置；`--simfile` 可覆盖配置中的工况路径。训练启动时将当前 Simfile 和 Parsfile 保存到本次训练目录的 `source_case`，各回合使用该快照；在 CarSim 改动工况后重新启动训练即可读取新工况。桥接程序每回合启动独立进程，不修改源工况。四维动作依次为 Np、Qy、Qψ、RΔ；Qvy、Qr、Rsteer 保持 `[weights]` 中的设置。观测归一化和奖励只由 C++ 计算。

评价会冻结一次工况，确保中性 MPC 和策略使用同一份源文件；每次评价请选择新的输出目录。训练所得策略不自动适用于新工况，改变车速、车辆或道路后需重新验证。若源工况没有唯一的恒定目标车速，目录名称使用 `vVariable`。桥接仍要求 C++ 支持该 CarSim 的参考轨迹和道路格式，并且 `simulation.distance_m` 应与当前场景相符。

## 训练、运行与评价

### TQC（参考 Kuznetsov 等，2020）

TQC 使用 SAC 随机策略和自动熵系数。多个评论家各输出 25 个分位数，目标网络的输出合并排序后，每个评论家去掉最高的 `d` 个分位数；演员更新使用全部评论家分位数的平均值。`--tqc-drop` 默认 2，可针对本工况比较 0、2、3；不能直接套用论文在 MuJoCo 上的最优值。

`--tqc-profile practical` 使用 `rl_training.ini` 中明确填写的评论家数量和隐藏层（当前为 3 个评论家、每个两层 256 单元）；`--tqc-profile paper` 临时采用论文实验中的 5 个评论家、每个三层 512 单元，计算开销明显更大。分位数数量、学习率、批量和目标网络更新频率均可在训练配置中分别调整；折扣系数从 C++ 配置读取。完整设置保存在训练 `manifest.json`；奖励函数更改后，旧 Critic 和回放池不适合直接续训。

```powershell
python python/train.py --algorithm tqc --episodes 500 --seed 42 --eval-frequency 1000 --tqc-profile practical --tqc-drop 2
```

论文：[Kuznetsov et al., ICML 2020](https://proceedings.mlr.press/v119/kuznetsov20a.html)。设计公式与训练设置见 [TQC 设计说明](docs/TQC_Design.md)。项目的 CarSim 工况与论文的 MuJoCo 基准不同，需看独立工况的控制误差和安全指标，不能仅据训练回报判断策略有效。

三种算法各训练 500 回合，并将中性 MPC、DDPG、TD3、TQC 画在同一组对比图上：

```powershell
python python/train_four.py --episodes 500 --seed 42
```

该命令按顺序训练，避免多个 CarSim 求解器互相干扰。三种算法和四组评价使用同一份启动时冻结的工况与控制配置；最终打开命令输出的 `four_controller_comparison.html`，每种算法的训练日志位于对应 `train_*/training.log`。如已有三个模型，可运行 `python python/compare_four.py --help` 查看独立对照命令。

训练默认完成 500 个回合（失败或截断也计为结束）；`--episodes` 可调整本次运行的回合数。验证与检查点仍按 `--eval-frequency` 指定的决策步数触发，当前 60 km/h 工况默认每 550 步（约 10 回合）；奖励图横坐标为已完成的训练回合。续训时回合从本次运行重新计数，模型加载不会自动恢复经验回放池。

```powershell
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
python python/parity.py --blocks 10 --out "runs/neutral_parity_$stamp"
python python/train.py --algorithm td3 --episodes 500 --seed 42
$latest = Get-ChildItem runs/python_rl -Directory -Filter 'td3_*' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$model = Join-Path $latest.FullName 'best/best_model.zip'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
python python/run_policy.py --algorithm td3 --model $model --out "runs/td3_policy_$stamp"
python python/evaluate.py --algorithm td3 --model $model --out "runs/td3_evaluation_$stamp"
```

将 `td3` 改为 `ddpg` 或 `tqc` 即可选用其他算法。`train.py` 会先运行固定动作对照：用同一 CarSim 工况比较固定 C++ MPC 与桥接中性动作的逐步状态、转角、权重和奖励；检查不通过则不启动训练。诊断保存在本次训练目录的 `neutral_parity/`。每次 Python 决策对应 `dt_s × decision_steps` 秒，训练模型使用 C++ 配置中的每决策折扣因子。训练保存模型、检查点、监控数据和同工况验证报告。运行命令中的 `applied_actions.csv` 记录 Python 发出的四个动作与 C++ 实际执行的 Np；`solver/episode_000001/bridge_trace.csv` 记录控制步的 Q/R 权重、转角和误差；`summary.json` 记录回合终止状态。

Python `.zip` 模型由 Python 加载并通过桥接协议发送动作，不能交给旧 C++ `.txt` 策略加载器。C++ 的旧强化学习训练入口已停用。基础 MPC 仍可用 `MPC_DoubleLaneChange.exe --config configs/default.ini --rl-mode fixed` 运行。

训练目录名包含算法、CarSim 目标车速、目标回合数、种子与时间（例如 td3_v60kmh_500episodes_seed42_时间），manifest.json 保存工况参数文件路径和 SHA-256 指纹；`文件说明.txt` 解释模型、检查点和图表。主报告是 `evaluation/training_and_control_report.html`：结果图参考 Peng 等（2025）的 Fig. 5 与 Fig. 10，分别显示车辆状态六联图，以及轨迹与控制量对比图。Fig. 5 在同一 CarSim 工况下对比中性 MPC 与策略；Fig. 10 扩展为八个子图，后四幅完整展示 Np、Qy、Qψ、RΔ 四个调参量。只有另外运行多个恒速工况才能得到论文式多速度叠加图。可通过 `--set mpc.np=25 --set mpc.nc=10` 临时指定时域；`rl.max_np_change_per_decision=0` 可关闭策略的时域变化限制。其他逐周期诊断量可在 CSV 中查看。当前奖励是非正的负损失，零损失时为 0；控制误差下降时回报向 0 增大。桥接不裁剪回报，旧正回报与当前回报不能直接比较。更新源代码后需重新构建 `MPC_RL_Bridge` 和 `MPC_DoubleLaneChange`。

控制算法见 [docs/Control_Algorithm_Implementation.md](docs/Control_Algorithm_Implementation.md)，CarSim 接口见 [docs/CarSim_CPP_Integration.md](docs/CarSim_CPP_Integration.md)。
