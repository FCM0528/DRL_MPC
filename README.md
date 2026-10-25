# CarSim + C++ 横向控制

CarSim 负责车辆动力学、纵向驾驶、道路、工况和参考轨迹；C++ 通过 VS Solver 接口读取状态、计算 MPC 前轮转角并写回 CarSim。DDPG 可调节预测时域 Np 及 Q/R 权重。MPC 预测模型采用用户提供的《基于RL-MPC的智能车辆轨迹跟踪横向控制策略》式 (3)–(6) 的三自由度形式：纵向、侧向、横摆三个动力学自由度，状态为 `[vy,vx,psi,r,Y,X]`。质量、惯量、轴距、轮胎侧偏刚度和附着系数从本次 CarSim 求解器回显获取；未知的纵向轮胎滑移与刚度没有虚构，由相邻 CarSim 状态得到的纵向加速度作为当前工况的外部作用量。预测模型只是 C++ 控制器内部近似，实际车辆运动始终由 CarSim 求解。独立车辆模型仿真模式已移除。

## 准备 CarSim 工况

1. 在 CarSim Run Control 配置车辆、道路、目标速度、路面附着系数与 `LTARG_TABLE` 参考轨迹。
2. 将左右前轮路轮转角设置为 `Replace` IMPORT，核对顺序和单位。当前默认工况左右索引为 0、1，单位为度。
3. 在 CarSim 运行工况一次，使 `simfile.sim` 和其引用的 `Run_all.par` 更新。当前轨迹适配器要求 `Straight East` 道路和 `LTARG_TABLE SPLINE_FLAT`。
4. 检查 [default.ini](configs/default.ini) 的 `[carsim]` 路径、IMPORT 索引及 `steering_gain` 与这次工况一致。

同机运行时不必复制 CarSim 安装目录。程序会在每次结果子目录内归档所用工况副本及 VS Solver 原生输出。改变 CarSim 工况后，请重新生成 Simfile 和 Parsfile；仅改变 C++ 控制权重时可直接重跑程序。

## 运行

在项目目录构建 Visual Studio x64 Debug 后，用 PowerShell 执行：

```powershell
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode fixed --no-open --out results\carsim_trial
```

训练自适应 Np/Q/R 策略（默认 200 回合，速度 30–120 km/h、附着系数 0.3–0.9）：

```powershell
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --no-open --out results\carsim_training
```

五个随机种子（101–105）分别训练 200 回合后，汇总性能均值、样本标准差与学习曲线：

```powershell
.\tools\run_five_seeds.ps1
.\tools\evaluate_five_seeds.ps1
& 'C:\Users\fengchengmin\.local\bin\python3.14.exe' tools\aggregate_five_seeds.py results\absolute_v15_five_seeds
```

多种子脚本读取项目中已固定的 CarSim 80 km/h 来源快照；第二步在额外速度和附着组合上评估，结果见 `five_seed_report.html`。V15 使用 CarSim 实测车速选择 MPC 预测模型：低于 32 km/h 为实测侧偏补偿的运动学模型，高于 40 km/h 为六状态三自由度动力学模型，32–40 km/h 平滑融合。DDPG 在所有速度输出独立的 Np 与六个 Q/R 绝对数值；固定 MPC 的 `[mpc] np` 和 `[weights]` 不改变同一策略在相同观测下的输出。训练的配对奖励参考控制器也使用 RL 自有的中性 Np/Q/R。动作的数值范围、奖励和收敛诊断见 [DDPG 说明](docs/DDPG_Adaptive_Horizon.md)。五种子报告汇总独立策略的性能，不平均网络权重。V14 倍率策略不能直接加载；满足范围等价条件时可按下文迁移网络权重。

已有的五份 V14 策略已经使用训练时权重换算为 V15 绝对动作策略，保存在 `results/absolute_v15_migrated_seeds/seed_101..105`。迁移仅改动作定义并原样复制网络权重，不等于重新训练；迁移过程和来源见各目录的 `conversion_manifest.txt`。用 seed 103 与当前 `default.ini` 的固定 MPC 比较：

```powershell
$policy = 'results\absolute_v15_migrated_seeds\seed_103\ddpg_policy.txt'
.\out\build\x64-Debug\MPC_DoubleLaneChange.exe --rl-mode compare --rl-model $policy --out results\carsim_compare_v15
```

历史 V14 的五种子结果仍在 `results/reward_v14_five_seeds_20261025/five_seed_report.html`。迁移网络在相同观测和原固定权重下的动作数学上等价；修改固定 MPC 后仍需重新评估闭环指标。新训练的 V15 策略在训练和独立评估完成前没有可报告的改善率或收敛结论。若另有 V14 策略，可运行 `python tools/convert_v14_policy.py <旧策略文件> --config configs/default.ini --out <新文件路径>`；脚本只接受训练配置与当前绝对范围严格对应的策略。

`--out` 指定结果**根目录**；每次运行自动创建唯一子目录，不覆盖旧结果。报告、CSV、有效配置、策略和 CarSim 原生文件保存在本次目录中。若 `sync_browser_result=true`，C++ 闭环结束后会把原生结果同步到对应 CarSim Run Control 的最近结果缓存；在 CarSim 点击 `Refresh` 后可使用 `Video / Video + Plot`。原缓存会备份在本次独立结果目录中。

## 参数与数据来源

- 在 CarSim 设置车辆、速度、道路、附着与参考轨迹。在 C++ 的 [default.ini](configs/default.ini) 设置 MPC 的 Np/Nc、Q/R、转角约束及 DDPG 奖励参数。
- MPC 预测模型的质量、惯量、轴距分量、轮胎侧偏刚度与路面摩擦系数来自本次 `LastRun_echo.par`，运行中的速度来自 CarSim 状态。缺失必要参数会报错。
- 速度图、CSV 与终端摘要统一使用 km/h；MPC 内部使用 m/s。报告中的转角是 C++ 发给 CarSim 的指令，并非测得的实际轮角。
- 当前参考轨迹仅支持全局 X 正方向直路；CarSim 表格的中间点经 C++ 三次样条插值，可能与 CarSim 内部插值有细微差别。

详细接口和运行顺序见 [CarSim_CPP_Integration.md](docs/CarSim_CPP_Integration.md)。
