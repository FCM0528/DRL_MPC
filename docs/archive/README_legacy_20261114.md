# 历史入口快照：2026-11-14流程（非当前执行规范）

# CarSim · LTV-MPC · DRL

CarSim 计算车辆运动。C++ MPC 每个控制周期求解前轮转角。DDPG、TD3 和 TQC 通过 Python 桥接，在同一条轨迹上随车辆状态和道路预瞄调整 Np、Qy、Qψ、RΔ。四组报告对比固定 MPC 和三种 DRL 策略的闭环效果。

**执行顺序**　CarSim 生成 Simfile → 冻结工况 → 时域粗扫 → 最佳 Np 组细扫 → 入选时域扫权重 → 生成固定与 DRL 配置 → DDPG、TD3、TQC 顺序训练 → 四组闭环回放并出报告。

2026-11-14 主项目检查已适配当前 **2 路输入 / 20 路输出**。结合 182 组扫描与历史套件实测复用，当前默认固定 MPC 为 **Np=40、Nc=13、Qy=11500、Qψ=2800、RΔ=10800**。默认动作映射已对齐既有三组 500 回合模型，并在当前 70 km/h、μ=0.8 工况验证。`Beta` 用于观测与奖励，Ax/Ay、轮胎力和实际轮角用于物理一致性诊断。结果与复用方法见 [主项目扫描记录](Main_Project_Audit_20261114.md)。

## 1. 准备环境

- 在 Windows 上准备 CarSim、可用的求解器许可、Visual Studio C++ 工具链、CMake 和 Python 环境。
- 在 CarSim Browser 中选好工况并生成 Simfile。配置文件中的 [carsim] simfile 指向这份文件，[carsim] solver_dll 指向对应版本的 64 位求解器。运行时保持 Browser 打开。
- 当前接口要求可读取的 CarSim LTARG_TABLE 参考轨迹、车辆状态和转向导入通道。支持的输入格式与转向配置见 [CarSim 接口](CarSim_CPP_Integration.md)。

以下命令在项目根目录的 PowerShell 中运行。CMake 需要在 PATH 中，也可从 Visual Studio Developer PowerShell 运行。生成器按本机 Visual Studio 版本选择。

~~~powershell
cmake -S . -B out/python-rl-build -G 'Visual Studio 18 2026' -A x64
cmake --build out/python-rl-build --config Release --target MPC_DoubleLaneChange MPC_RL_Bridge

conda activate drl_mpc
python -m pip install -r python/requirements.txt
~~~

## 2. 扫当前 CarSim 工况

每次换工况，先在 CarSim Browser 重新生成 Simfile，再从这一步开始。默认读取 `configs/default.ini` 指向的 Simfile。车速、附着、车辆和参考轨迹都来自该工况。默认终点是参考轨迹末端，也可在此步用 `--distance` 指定全局 X 终点。

~~~powershell
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$sweep = "runs/sweep_$tag"
python python/sweep.py --out $sweep
if ($LASTEXITCODE -ne 0) { throw '扫参未成功，请先查看本轮日志' }
~~~

脚本内部依次执行以下步骤。

1. **冻结工况。** 将 Simfile 和它指向的 Parsfile 保存到 `$sweep/source_case/`，每组仿真从这份快照读取工况。
2. **固定权重，按 Np 分组粗扫。** 默认范围是 **Np=20–60、Nc=5–16**，包含两端。将连续 Np 每 5 个分一组，例如 20–24、25–29，分别用 Np=22、27 作代表。每个代表值配合范围内所有满足 Nc < Np 的 Nc 跑一遍。起始配置中的 Np 或 Nc 若超出默认区间，区间会连续扩展以覆盖该值。
3. **只细扫最佳 Np 组。** 用粗扫中全程横向 RMSE 最低的合格结果选组。在该组内逐整数扫 Np，仍与所有合法 Nc 配对，并复用代表点的结果。若只指定了一个组宽以内的范围，直接细扫。
4. **固定入选时域，再扫权重。** 在细扫胜出的 Np/Nc 上联合扫描 Qy、Qψ、RΔ。三个权重各用原值的 0.8、1.0、1.25 倍，复用原始权重组合的结果，所以补跑 26 组。
5. **输出同工况的两份配置。** 各阶段以跑完全程且 QP 零失败作为入选条件，先按全程横向 RMSE 排名，再比较横向误差峰值。`$sweep/configs/sweep_fixed.ini` 保存最终入选参数，`$sweep/configs/sweep_rl.ini` 的零动作对应同一组参数。`sweep_metrics.csv` 保存各阶段结果，`sweep_selection.json` 记录粗扫代表点、入选 Np 组、最终参数和工况指纹。

按当前默认配置，粗扫 108 次，细扫最多再跑 48 次，加上 26 次权重扫参，合计最多 **182 次仿真**。最终结果是本轮实际采样中的最优值。

### 默认权重扫描值

权重基准取自起始配置的 `[weights]`。以下是使用 `configs/default.ini` 时实际扫描的离散候选值。

| 权重 | 0.8 倍 | 1.0 倍（原值） | 1.25 倍 |
| --- | ---: | ---: | ---: |
| Qy（`q_y`） | 9200 | 11500 | 14375 |
| Qψ（`q_psi`） | 2240 | 2800 | 3500 |
| RΔ（`r_delta`） | 8640 | 10800 | 13500 |

三个权重独立选值，形成 27 种组合。全为原值的组合复用时域扫参结果，所以实际补跑 26 次。用 `--weight-scales` 可指定其他倍率，脚本始终保留 1.0 倍。

| 选项 | 用途 |
| --- | --- |
| --simfile 'D:\CarSimData\simfile.sim' | 显式指定当前工况文件 |
| --distance 120 | 在全局 X = 120 m 结束 |
| --np-range 30 50 --nc-range 7 14 | 指定连续区间，对 Np 分组粗扫，再细扫入选组 |
| --np 20 30 40 50 --nc 5 8 12 16 | 直接扫描指定的离散候选值 |
| --weight-scales 0.7 1 1.4 | 对 Qy、Qψ、RΔ 分别使用这些倍率 |
| --config configs/default.ini | 选择起始 MPC 配置 |

## 3. 使用扫参结果训练三组

扫参成功后，在同一个 PowerShell 中沿用 `$sweep`。如果重开终端，先把 `$sweep` 设为本轮扫参目录。

~~~powershell
python python/train_four.py --config "$sweep/configs/sweep_rl.ini" --fixed-config "$sweep/configs/sweep_fixed.ini"
~~~

套件会保存两份配置和工况快照。每种算法先做中性动作一致性检查，再按 **DDPG → TD3 → TQC** 的顺序训练，默认每种 **500 回合**、随机种子 **42**。训练期间保存检查点，每组结束后保存 `final_model.zip`。可用 `--episodes` 和 `--seed` 覆盖默认值。

训练经验全部通过 CarSim 在线闭环交互采集，三种算法使用各自的空经验池；不使用扫参结果或固定 MPC 转角作为监督标签。道路预瞄保留 CarSim 参考轨迹和附着信息。初始采样预热为 550 个决策，续训加载权重后也重新采样预热；数据来源、采样数量和更新次数记录在训练 manifest 中。

需要逐段比较时，在训练命令末尾加 `--segments` 和本工况的 X 切点，例如 `--segments 50 90`。切点用于报告统计，训练奖励由 DRL 配置决定。

### 实时查看训练日志

保持训练窗口运行，另开一个 PowerShell，在项目根目录执行以下命令。它会找到最近创建的训练套件和当前算法的日志，持续显示新增内容。

~~~powershell
$run = Get-ChildItem .\runs -Directory -Filter 'four_*' |
    Sort-Object CreationTime -Descending | Select-Object -First 1

$log = Get-ChildItem "$($run.FullName)\train_*\training.log" |
    Sort-Object CreationTime -Descending | Select-Object -First 1

$log.FullName
Get-Content -LiteralPath $log.FullName -Encoding UTF8 -Tail 30 -Wait
~~~

- `episodes` 是当前算法已结束的回合数，默认目标为 500。
- `ep_rew_mean` 是近期平均训练回报，`total_timesteps` 是累计策略决策步数。
- 切到下一个算法时，在**看日志的窗口**按 `Ctrl+C`，再运行上述命令即可跟随新日志。若训练指定了 `--out`，把命令里的 `.\runs` 换成对应目录。

## 4. 查看四组对比报告

2026-11-14 已完成新一轮 DDPG、TD3、TQC 各 500 回合在线交互训练（seed=42，从随机权重和空经验池开始）。每种策略采集 39,500 个决策。当前工况四组均完赛、QP 零失败；固定 MPC / DDPG / TD3 / TQC 横向 RMSE 分别为 0.08724 / 0.07783 / 0.07511 / 0.07532 m。[本轮四组报告](../../runs/four_v70kmh_500ep_seed42_20261114_112814/comparison/four_controller_comparison.html) ｜ [训练曲线](../../runs/four_v70kmh_500ep_seed42_20261114_112814/training_summary.png) ｜ [采样核验记录](../../runs/four_v70kmh_500ep_seed42_20261114_112814/training_data_verification.json)。

三组训练完成后，`train_four.py` 会自动用扫参选出的固定 MPC 和三个最终模型做四次确定性闭环回放。四组使用同一工况和结束位置。终端打印套件目录和报告路径。

本机已有三组 500 回合模型，当前接口下的复用报告位于 [已训练模型四组实测报告](../../runs/reuse500_final_20261114/four_controller_comparison.html)。报告同时展示实测侧向加速度、侧偏角、轮角执行差和前后轴轮胎力。回放前会核对模型保存的训练配置与当前控制、观测和动作设置，避免将同为四维的不同动作映射混用。

先看四组是否跑完全程及 QP 是否失败，再比较横向 RMSE 和峰值。使用 `--segments` 时，逐段查看是否到达段终点、样本数、RMSE 和绝对峰值，便于判断某段的改善是否伴随其他路段退化。

| 文件 | 内容 |
| --- | --- |
| comparison/four_controller_comparison.html | 四组误差、终点与 QP 指标，轨迹、车辆状态和调参曲线 |
| comparison/four_controller_metrics.csv | 四组全程指标 |
| comparison/segment_metrics.csv | 使用 --segments 时输出的分段指标 |
| training_summary.png | 三种算法的训练回报曲线 |
| suite_manifest.json | 工况、配置与最终模型位置 |
| train_ddpg/、train_td3/、train_tqc/ | 各组训练日志、检查点和 final_model.zip |

## 5. 单独运行与配置

~~~powershell
# 固定 MPC，使用扫参选出的参数
& .\out\python-rl-build\Release\MPC_DoubleLaneChange.exe --config "$sweep/configs/sweep_fixed.ini"

# 只训练一种算法，将 td3 换成 ddpg 或 tqc 即可
python python/train.py --algorithm td3 --config "$sweep/configs/sweep_rl.ini" --fixed-config "$sweep/configs/sweep_fixed.ini"

# 用已有的三个模型重做四组对比
python python/compare_four.py --ddpg D:\models\ddpg.zip --td3 D:\models\td3.zip --tqc D:\models\tqc.zip --config "$sweep/configs/sweep_rl.ini" --fixed-config "$sweep/configs/sweep_fixed.ini"
~~~

已有模型应与训练时的控制配置配对。单模型回放入口为 python/run_policy.py，单算法评价入口为 python/evaluate.py。

| 位置 | 负责内容 |
| --- | --- |
| configs/default.ini | 当前工况路径、MPC 时域与权重、转向约束、求解器 |
| configs/carsim_python_rl.ini | 道路预瞄观测、动作范围、策略决策周期、奖励 |
| configs/rl_training.ini | 回合数、种子、学习率、网络和探索设置 |
| configs/carsim_curvature_schedule_40ms.ini | 曲率预瞄切换 Np 的示例，参数来自历史低附着工况 |
| src/ | CarSim 接口、车辆模型、LTV-MPC、桥接与 C++ 报告 |
| python/ | 工况快照、扫参、训练、模型回放和对照报告 |

进一步说明见 [控制算法](../../docs/MPC_MODEL_AND_BASELINES.md)、[训练参数](../../docs/DRL_DESIGN_AND_REWARD.md) 和 [TQC 设计](../../docs/DRL_DESIGN_AND_REWARD.md)。
