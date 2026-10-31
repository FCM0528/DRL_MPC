# CarSim + C++ 横向控制

CarSim 负责车辆动力学、纵向驾驶、道路、工况和参考轨迹；C++ 通过 VS Solver 接口读取状态、计算 MPC 前轮转角并写回 CarSim。控制器使用增量式 LTV-MPC：唯一预测模型为六状态三自由度动力学 `[vy,vx,psi,r,Y,X]`，优化变量为前轮转角增量；沿上一控制序列的预测轨迹逐步线性化，并用 Tustin 方法离散化。DDPG 或可选 TD3 调节预测时域 Np 及 Q/R 权重。质量、惯量、轴距、轮胎侧偏刚度和附着系数从本次 CarSim 求解器回显获取；相邻 CarSim 状态给出当前纵向加速度。预测模型只是 C++ 控制器内部近似，实际车辆运动始终由 CarSim 求解。算法与公式见 [控制算法说明](docs/Control_Algorithm_Implementation.md)。

## 当前道路的速度训练

### Np 和 Q/R 敏感性扫描

[查看 2026-10-30 扫描报告](policies/audit/sensitivity_20261030/report.html)。该报告使用保存的当前道路和车辆快照，在 60、70、73.6667、80 km/h 扫描固定 MPC 的 Np、Qy、Qψ 和 RΔ，并补充 Np/Qy/RΔ 的组合试验；包含 RMSE、峰值误差、航向误差、求解和保护步数。控制器使用 `configs\carsim_focus_fast_guarded.ini` 的 MPC 参数，扫描不是强化学习训练。

如需使用同一快照重新扫描，选择一个较短且不存在的输出目录，以下命令分别运行单变量和组合扫描（每个速度会生成私有工况副本）：

```powershell
$single = 'policies\audit\scan_new'
$joint = 'policies\audit\joint_new'
foreach ($v in @('60','70','73.6667','80')) {
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sensitivity_scan.ps1 -SpeedKmh $v -OutputRoot $single
  powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\sensitivity_scan.ps1 -SpeedKmh $v -OutputRoot $joint -Interactions
}
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_sensitivity_report.ps1 -OutputRoot $single -InteractionRoots $joint
```

扫描脚本默认引用上一次训练保存的 `training_source_Run_all.par` 和 Simfile 快照；更换 CarSim 工况后，需指定相应快照的 `-SourcePar` 和 `-SourceSimfile`，不能将旧报告当成新工况结果。

### 60 和 80 km/h 专项策略

#### 固定使用 60 和 80 km/h 策略

`policies\fixed_speed\v60` 和 `policies\fixed_speed\v80` 分别存放该速度下固定选用的 DDPG、TD3 策略。每份策略包含训练配置、选中回合和 SHA-256 校验记录；对照命令只读取这些固定文件，不会自动切换到后续最新训练结果。状态为 `inherited_actor` 表示选中了热启动 Actor；`neutral_fallback` 表示没有选出优于基准的策略，比较报告会警示。

2026-10-30 归档结果：60 km/h 的 DDPG 为继承 Actor（本轮 200 回合未超越），TD3 为第 100 回合；80 km/h 的 DDPG 为第 190 回合，TD3 为继承 Actor。按各自的固定道路快照复跑，横向 RMSE（固定 MPC / DDPG / TD3）分别为 60 km/h：0.0616 / 0.0533 / 0.0530 m，80 km/h：0.2417 / 0.2206 / 0.2123 m。这些是各自训练时的工况，不是独立泛化验证，也不能单凭一组结果证明训练已稳定收敛。

[查看四组训练曲线与检查点核查](policies/fixed_speed/training_curves.html)。该报告读取四组原始逐回合和检查点 CSV，分别绘制每步训练奖励及无探索验证奖励；原始训练没有记录 Actor/Critic 损失，因此报告不含损失曲线。若需要重新生成报告：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\build_fixed_training_report.ps1
```

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_fixed_speed.ps1 -Speed 60
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_fixed_speed.ps1 -Speed 80
```

命令运行固定策略目录内同一道路的私有 60/80 km/h 工况快照，并打开固定 MPC、DDPG、TD3 三组对照报告；它不会改写 CarSim 当前工况。固定速度策略仅对对应车速和当前道路进行了训练与测试，切换其他工况后必须重新验证。各策略的训练和性能情况应查看 `pinned_policy.csv` 与对应训练报告。

如需继续试验训练，可运行以下命令；产生的新训练结果保存在 `policies\iterations`，**不会自动替换上述固定策略**：

```powershell
foreach ($speed in @(60, 80)) {
  foreach ($algorithm in @('ddpg', 'td3')) {
    powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 `
      -Algorithm $algorithm -Config "configs\carsim_fixed${speed}_np_focus.ini" `
      -Simfile "policies\fixed_speed\v${speed}\source\simfile.sim" `
      -Episodes 200 -FocusTargetSpeeds -NoPromotion
  }
}
```

#### 固定 60 km/h 的 DDPG 训练

`configs\carsim_fixed60_np_focus.ini` 固定训练速度为 60 km/h，保留 Np、Qy、Qψ、RΔ 四个动作，并将 Np 探索噪声设为其他动作的 1.5 倍；每 0.2 s 决策一次。奖励包含横向误差、航向误差和附着保护等项，检查点还要求：航向 RMSE 不超过基准的 105% + 0.02°，附着保护与 QP 失败步数均不多于基准。固定车速没有独立速度网格，训练结果不能证明邻近速度性能。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 `
  -Algorithm ddpg -Config configs\carsim_fixed60_np_focus.ini -Episodes 200 `
  -Simfile policies\audit\focus60_80_source\v60\simfile.sim -FocusTargetSpeeds -NoPromotion

powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_fixed60_ddpg.ps1
```

第二条命令默认使用固定归档的 60 km/h DDPG 策略，与固定 MPC 对照并打开报告。默认使用上述私有 60 km/h 快照；切换工况时可用 `-Simfile <路径>`，但结果需重新验证。想从特定旧 Actor 初始化训练，可给第一条命令加入 `-WarmStart <策略路径>`，Critic 与经验池会重新初始化。

2026-10-30 的 200 回合实验选中的是继承的初始 Actor（检查点 -1），本轮没有找到综合指标优于它的后续检查点。该策略同工况复跑的横向 RMSE 为 0.0533 m，固定 MPC 为 0.0616 m；航向 RMSE 分别为 0.9525°、1.0004°，两者附着保护与 QP 失败均为 0 步。由于只验证了一个确定性的 60 km/h 工况，不能由此证明其他工况有效或训练已稳定收敛。

`configs\carsim_focus_fast_guarded.ini` 是新的 0.2 s 决策实验：`dt_s=0.05`、`decision_steps=4`，动作仍为 Np、Qy、Qψ、RΔ。折扣系数设为 `0.98^(0.2)`，保持约一秒的折扣为 0.98；奖励增加横向和峰值误差的权重，并保留航向、横摆、转角与平顺性项。更改决策周期或奖励后不得直接复用旧 Critic/经验池，只可 `-WarmStart` 同算法的 Actor。

该策略在 55–85 km/h 训练，在每次策略决策时按实测车速将 85–95 km/h 的策略动作平滑混合到中性动作，决策时车速达到 95 km/h 后使用中性 MPC 参数；两次决策间保持上次动作，切换可能延迟至多 0.2 s。这是为防止将未覆盖高速的学习动作全量应用于高速而设计，并不保证在所有车辆、道路和速度下安全；85–95 km/h 过渡区域也需单独评估。

当前控制器会在自适应参数变化、切换 MPC 对象时继承上一周期的预测控制序列；训练端和运行端均使用该逻辑。修复前的策略不是在相同控制器下训练的，自动选择会拒绝旧版本；如需利用旧策略，应通过 `-WarmStart` 只继承同算法 Actor，并重新验证。训练目录通过 `controller_revision.txt` 记录该版本。

该实验使用 65°/s 前轮指令速率上限，并从 80 至 95 km/h 线性降至 5°/s。65°/s 是当前 CarSim 回显中的驾驶员方向盘限速 1200°/s，结合齿条行程 44.12 mm/rev 和零位附近齿条到轮角斜率约 0.441 deg/mm 的估算；前轮 IMPORT 绕过驾驶员方向盘控制，二者不等同。实验工况下完全取消高速调度会显著发散，因此必须保留高速保护；该实验并不保证其他道路与车辆安全。全局 `default.ini` 未改，其他配置继续使用原限值。

该速率并非全程不约束：当前道路固定 MPC 的 60、80 km/h 仿真中，分别有 10/219、9/164 个指令间隔接近 65°/s。与旧限值相比，两种工况固定 MPC 的横向 RMSE 分别由 0.0587、0.2171 m 变为 0.0616、0.2417 m；结果不能说明速率放宽本身改善跟踪。

专项训练及三组对照命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_focus_fast_guarded.ini -Episodes 80 -Curriculum -FocusTargetSpeeds
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3 -Config configs\carsim_focus_fast_guarded.ini -Episodes 80 -Curriculum -FocusTargetSpeeds
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 -Config configs\carsim_focus_fast_guarded.ini
```

三组对照自动选择匹配 0.2 s 决策周期和实验转向约束的最新完成策略；如选出的是中性回退，报告会提示。该策略的训练速度范围仍为 55–85 km/h，不能将固定 MPC 的高速测试当成强化学习策略的高速验证。

2026-10-30 的 80 回合训练：DDPG 正式选优为中性回退；TD3 选中的是继承的 Actor（检查点 -1），不是本轮更新后的新检查点。该 TD3 在 60、80 km/h 分别得到 0.0566、0.2123 m 横向 RMSE，相应固定 MPC 为 0.0616、0.2417 m；但在独立的 57.7、73.7 km/h 工况有退化，报告会显示提示，不能称为全速度可靠或训练稳定收敛。100、120 km/h 的本轮选中策略在该道路快照与固定 MPC 一致，因高速保护将动作归中。

如需诊断 DDPG 第 20 回合候选，可显式指定该检查点；它在 60、80 km/h 的横向 RMSE 有下降，但在独立 73.7 km/h 工况从 0.146 m 增至 0.199 m，未通过正式选优：

```powershell
$ddpg = 'policies\iterations\ddpg\cycle_20261030-203227-216\ddpg_train_carsim_NpNominal25_Nc8_20261030-203227\checkpoint_0020_policy.txt'
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_focus_fast_guarded.ini -DdpgPolicy $ddpg
```

`configs\carsim_focus_60_80.ini` 使用 Np、Qy、Qψ、RΔ 四个动作，关闭 60 km/h 中性动作混合；训练采样范围为 55–85 km/h，速度观测按固定的 30 m/s 缩放。该专项策略不能视为已覆盖 30–120 km/h。训练保留 CarSim 当前道路附着表，并从当前求解器回显读取车辆质量、横摆惯量、轴距和轮胎表。Q/R、时域、转向约束和求解器参数为 C++ 控制器配置，并非 CarSim 驾驶员参数。

`configs\carsim_focus_60_80_tracking.ini` 是该专项的另一轮实验，提高了横向误差项的奖励权重。由于奖励改变，须重新初始化 Critic 和经验池；可通过 `-WarmStart` 仅继承同算法 Actor。它与上述配置的控制器动作和执行设置相同，但训练奖励不同。

在 CarSim 更新当前工况、启用前轮转向 IMPORT 后，运行下列命令进行三组对照并打开报告；脚本自动选取与该专项配置和速度范围兼容的最新已完成 DDPG、TD3 策略。修改 CarSim 目标速度到 60 或 80 km/h 并更新后，可分别运行同一条命令：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_focus_60_80.ini
```

上述自动选择的最新策略如果是中性回退，报告会提示，此时曲线可能重合。2026-10-30 专项第一轮有两份历史检查点，在旧版控制器的同源 60 和 80 km/h 工况横向 RMSE 均低于固定 MPC，但部分邻近速度退化。新版控制器修复了自适应参数切换时预测序列被重置的问题，旧检查点仅可作为跨版本诊断或 Actor 热启动，其原有数值不能代表新版结果；下列诊断会明确提示模型不匹配：

```powershell
$ddpg = 'policies\iterations\ddpg\cycle_20261030-193544-607\ddpg_train_carsim_NpNominal25_Nc8_20261030-193544\checkpoint_0160_policy.txt'
$td3 = 'policies\iterations\td3\cycle_20261030-193544-824\td3_train_carsim_NpNominal25_Nc8_20261030-193544\checkpoint_0050_policy.txt'
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_focus_60_80.ini -DdpgPolicy $ddpg -Td3Policy $td3 -AllowModelMismatch
```

继续训练时，以下命令分别续训对应算法的兼容状态；从头训练可加 `-Fresh`，只继承历史 Actor 可指定 `-WarmStart <策略文件>`。训练及评估结果分别保存在 `policies\iterations` 和 `policies\evaluations\iterations`，须检查选中检查点和独立验证结果。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_focus_60_80.ini -Episodes 160 -Curriculum -FocusTargetSpeeds
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3 -Config configs\carsim_focus_60_80.ini -Episodes 160 -Curriculum -FocusTargetSpeeds
```

### 历史多速度实验

`default.ini` 的 `[mpc]` 配置速度相关转向变化率上限：80 km/h 及以下使用 `[constraints] max_steer_rate_deg_s`，从 80 到 110 km/h 线性下降到 5°/s，超过 110 km/h 保持该上限；限值始终不超过 `[constraints]` 的物理上限。该限制同时作用于 MPC 优化、受附着限制时的保护转向和最终后备转向。`speed_rate_enabled=false` 可用于复现旧行为。参数均可在 `default.ini` 中修改。

实际速率上限按车身纵向速度实时计算，公式为 `rate(v) = rate_max + clamp((|v|-v_start)/(v_end-v_start), 0, 1) * (min(rate_max, rate_end)-rate_max)`，速度单位须一致。`[solver]` 的 ADMM 参数由 rho=20000、迭代上限 2500 调整为 40000、10000，以改善新约束下的收敛。终点从 120 调整到 110 km/h 后，当前道路 100 km/h 的固定 MPC 横向 RMSE 从 1.483 m 降至 0.680 m；120 km/h 为 0.863 m。100 km/h 测试仍有 52 步进入附着保护控制。不同附着道路的回归显示部分 100 km/h 工况略有退化，不表示所有工况都已达标。

控制器约束改变后，旧强化学习策略不能直接视为已适配：比较和评估脚本会检查训练配置并拒绝模型不匹配。可用旧策略 `-WarmStart` 仅继承 Actor，在新约束下重新训练；Critic 和经验池重新初始化，之后须进行独立验证。

当前训练配置为 `configs/carsim_compact_training.ini`：DDPG/TD3 调节 Np、Qy、Qψ 和 RΔ，其余 Q/R 固定为对应上下界的几何中心；当前中心与 `default.ini` 中的 Q/R 一致。策略决策间隔为 20 个 MPC 采样周期，训练范围为 30–120 km/h。使用 `-FocusTargetSpeeds` 时，60、80、95、98、100、102、104、105、116、120 km/h 的选优要求 RMSE 不超过同工况的中性 MPC；若无学习检查点满足要求，回退中性策略。修改 Q/R 范围须重新训练并验证。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_compact_training.ini -Episodes 300 -Seed 42 -Fresh -Curriculum -FocusTargetSpeeds
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3 -Config configs\carsim_compact_training.ini -Episodes 300 -Seed 42 -Fresh -Curriculum -FocusTargetSpeeds
```

可选 `configs\carsim_compact_relative.ini` 将每个决策块的配对奖励差除以 `max(0.2 × reward_scale, |同工况基准块回报|)`，再限制在 ±8，以降低不同速度和轨迹阶段的奖励尺度差异。它只改变训练奖励，并不保证收敛；需要重新训练和独立验证，不能直接复用不同奖励的 Critic 与经验池。`-WarmStart <策略文件>` 可以只继承 Actor。

`configs\carsim_compact_guarded.ini` 额外对 QP 未收敛或预测失败后进入保护的事件计数并施加失败惩罚；检查点选优要求这些事件不多于同工况基准。普通保护控制未尝试 QP 时不计作求解失败。该选项改变训练奖励，需重新训练 Critic 与经验池。

`configs\carsim_compact_blend.ini` 是低速混合实验：62 km/h 及以下使用中性动作，62–80 km/h 平滑过渡，80 km/h 及以上使用完整策略，因此 60 km/h 与中性 MPC 相同，并不代表强化学习有提升。2026-10-30 的两组 160 回合训练在部分独立速度下退化，尚未通过全速度验证。脚本会按模型配置选择最新完成的兼容策略，并提示已记录的独立验证失败；更换工况后仍需重新评估。

使用 **CarSim 当前工况**运行基础 MPC、DDPG-MPC 和 TD3-MPC 三组对照，并自动打开图形报告：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_compact_blend.ini
```

先在 CarSim 中启用前轮转向 IMPORT、更新 Simfile，并保持 Browser 或 License Manager 开启。以上命令读取当前工况；更改速度或道路后重新运行即可。脚本自动选择与该配置匹配的最新已完成策略。

查看该实验在当前道路快照的三组对照和独立速度评估：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 -Config configs\carsim_compact_blend.ini -Simfile policies\audit\current100_source\simfile.sim -NoOpen
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\evaluate_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_compact_blend.ini -Simfile policies\audit\current100_source\simfile.sim -NoOpen
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\evaluate_general_policy.ps1 -Algorithm td3 -Config configs\carsim_compact_blend.ini -Simfile policies\audit\current100_source\simfile.sim -NoOpen
```

训练结束后，运行下列命令自动选择最新完成的精简动作策略并生成三组曲线报告。选中检查点可能是中性回退，仍须查看训练报告和独立评估。CarSim 需先启用前轮转向 IMPORT 并更新 Simfile；若当前是驾驶员模式，可显式用 `-Simfile <私有联仿副本>`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1
```

在调整高速约束与求解器之前，2026-10-29 的 Np 单动作训练选中了 DDPG 第 40 回合和 TD3 第 200 回合。该结果不能直接应用于当前新参数。此前以旧 Actor 热启动的 60 回合训练均未选出通过门槛的学习检查点；这是历史记录，不能据此判断后续训练的选优结果。不能宣称 30–120 km/h 全范围达标或已经稳定收敛。

以下是此前 Np 与 Q/R 同时训练的记录和命令，供历史对照。Np 单动作策略与全动作策略格式不同，需要用训练时的配置评估。

`configs/carsim_current_road_training.ini` 继承非线性实验模型和 `default.ini`，保留当前 CarSim 道路的完整附着表，只在私有工况副本中抽样 30–120 km/h 的车速。策略的中性 Np 当前设为 25，与本轮固定基准一致；修改固定 MPC 的 Np 后，如需改变策略动作中心，也须同步修改该训练配置并重新训练。

首次从头训练并进行独立速度评估：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_current_road_training.ini -Episodes 200 -Fresh -Curriculum -NoPromotion
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3 -Config configs\carsim_current_road_training.ini -Episodes 200 -Fresh -Curriculum -NoPromotion
```

训练完成后，结果和检查点保存在 `policies/iterations/<算法>/`，独立速度验证保存在 `policies/evaluations/iterations/<算法>/`。加 `-FocusTargetSpeeds` 时，当前道路每 8 回合覆盖 8 个速度层，其中 95–105 km/h 单独分层；`-Fresh` 不加载旧模型，后续在相同道路和模型下续训时可省略。训练选择没有通过门槛时，程序会保留中性策略，不代表强化学习已优于基础 MPC。当前道路训练不证明其他附着道路也能达到相同效果；下方旧版本训练与注册表说明仅作历史参考，不应直接将旧策略用于新模型。

当前道路三组对照可从最近完成的训练目录选择策略（运行前查看训练报告，确认选中的是学习检查点还是中性回退）：

```powershell
$ddpg = Get-ChildItem policies\iterations\ddpg -Recurse -Filter ddpg_policy.txt | Where-Object { (Test-Path (Join-Path $_.DirectoryName 'rl_report.html')) -and -not (Test-Path (Join-Path $_.DirectoryName 'invalid_training.txt')) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
$td3 = Get-ChildItem policies\iterations\td3 -Recurse -Filter td3_policy.txt | Where-Object { (Test-Path (Join-Path $_.DirectoryName 'rl_report.html')) -and -not (Test-Path (Join-Path $_.DirectoryName 'invalid_training.txt')) } | Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 -Config configs\carsim_np_only.ini -DdpgPolicy $ddpg -Td3Policy $td3
```

2026-10-29 本轮结果为 DDPG 第 160 回合和 TD3 中性回退，60 km/h 下 DDPG 误差高于基础 MPC；未证明 TD3 优于 DDPG 或固定 MPC，也未证明高速工况达标。

## 准备 CarSim 工况

1. 在 CarSim Run Control 配置车辆、道路、目标速度、路面附着系数与 `LTARG_TABLE` 参考轨迹。
2. 将左右前轮路轮转角设置为 `Replace` IMPORT，核对顺序和单位。当前默认工况左右索引为 0、1，单位为度。
3. 在 CarSim 运行工况一次，使 `simfile.sim` 和其引用的 `Run_all.par` 更新。当前轨迹适配器要求 `Straight East` 道路和 `LTARG_TABLE SPLINE_FLAT`。
4. 检查 [default.ini](configs/default.ini) 的 `[carsim]` 路径、IMPORT 索引及 `steering_gain` 与这次工况一致。

同机运行时不必复制 CarSim 安装目录。程序会在每次结果子目录内归档所用工况副本及 VS Solver 原生输出。改变 CarSim 工况后，请重新生成 Simfile 和 Parsfile；仅改变 C++ 控制权重时可直接重跑程序。

当 CarSim 使用 `OPT_STOP=1` 按道路站距停止时，程序会在私有求解副本中将 `SSTOP` 至少延长到 C++ 目标质心距离加 10 m，让控制器自行停止；原 CarSim 工况不变。这避免车辆原点站距先到 180 m，而质心尚未到 180 m 时被误判为仿真失败。非数值站距停止配置会明确报错，需先在 CarSim 中设置数值。

支持单一道路的常量附着 `MU_ROAD_CONSTANT`，以及随站距变化的 `MU_ROAD_TABLE STEP`。后者按车辆质心 X 对应的道路站距读取当前 μ，每个控制周期更新 MPC 参数和强化学习观测；这依赖 Straight East 道路及道路与驾驶员参考路径一致。μ 是地图估计，预测时域内保持当前值，不能表示阶跃边界处四个轮胎的不同接触值。横向附着表、二维摩擦图、多道路及其他插值方式尚不支持，会明确报错。

新结果目录按实际算法和运行模式命名，例如 `td3_train_carsim_NpNominal15_Nc7_时间`、`ddpg_eval_general_carsim_Nc7_时间`、`mpc_ddpg_td3_compare_carsim_Nc7_时间`。训练目录中的 `NpNominal` 是自适应控制的标称预测时域，并不表示运行时 Np 固定；详细参数和实际范围见结果文件。直接训练 TD3 时默认策略文件为 `td3_policy.txt`，也可显式指定文件名。旧结果目录保持原名，历史路径仍可使用。

## 运行

在项目根目录用 PowerShell 构建 Release 版（下面的 CMake 路径为当前机器的 Visual Studio 安装路径）：

```powershell
$cmake = 'D:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
& $cmake -S . -B out\build\x64-Release -G 'Visual Studio 18 2026' -A x64
& $cmake --build out\build\x64-Release --config Release --parallel 4
```

先在 CarSim Browser 运行工况并保持 Browser 或 License Manager 开启。随后在同一 Windows 会话执行固定 MPC：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode fixed --set carsim.sync_browser_result=false --no-open --out results\carsim_trial
```

### 历史注册表说明

历史注册表使用旧模型；当前默认配置的约束和求解器已调整，因此旧注册表不应直接用于当前模型。当前道路的新策略请使用文首的三组对照命令。运行时保持 CarSim Browser 或 License Manager 开启。

脚本依次运行基础 MPC、DDPG-MPC 和 TD3-MPC，各运行一次，并核对三次使用的 CarSim 源工况。结果目录内的 `comparison_report.html` 在同一张图中叠加三组轨迹、误差、转角、横摆角速度、速度、Np 和六项 Q/R 权重，并给出三组指标表；原始结果和 `comparison_metrics.csv` 也保存在目录内。报告默认自动打开，如不需自动打开，在命令末尾加 `-NoOpen`。基础 MPC 使用当前配置的 `[mpc]` 和 `[weights]`，与训练选优时的 RL 中性参考不一定相同。

脚本默认读取 `policies\active_registry.csv` 中的变附着策略，训练范围为 μ=0.3–0.9、30–120 km/h。修改 CarSim 目标速度或附着系数并重新运行工况、更新 Simfile 后，仍使用上面的同一条命令。当前在 μ=0.4、30–70 km/h 时会自动采用经过独立验证的 DDPG 区域专家，其他工况采用通用 DDPG；TD3 使用变附着策略。报告会显示本次选用的策略及训练范围。如果通用策略更新导致区域验证依据变化，专家路由会暂停，直至重新验证。

历史策略、模型索引、训练状态和评估记录集中在 [policies](policies/README.md)。需要在之前的训练基础上继续迭代时，在 CarSim Browser 开启的同一 Windows 会话中依次执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3
```

训练脚本默认增加 300 回合，使用当前道路配置并进行独立速度评估，不自动更新历史注册表。后续续训可以恢复 Critic、优化器和经验池；不同工况的表现和训练是否收敛，需要查看无探索检查点及独立评估。

只有选择非保留源道路的配置时，变附着训练才会按比例缩放源 STEP 表。当前道路配置保留原表；不要同时使用 `-VaryRoadTransition`。不同道路的评估结果不能直接用于原道路的自动晋升，需在新工况重新建立基准。

多附着配置的普通课程采样每 30 回合覆盖 6 个速度层与 5 个初始附着层；`-FocusTargetSpeeds` 使用 8 个速度层。`-Curriculum` 对 DDPG 和 TD3 均按速度由低到高排序。固定验证工况及无探索检查点用于判断改进；课程顺序和平滑曲线不能保证收敛。

历史多附着训练可以使用非保留道路配置并追加 `-FocusTargetSpeeds -Curriculum`；`-VaryRoadTransition` 会随机调整 STEP 表切换位置。以下是历史多附着命令，当前道路训练请使用文首命令。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm ddpg -Config configs\carsim_nonlinear_experimental.ini -FocusTargetSpeeds -Curriculum -VaryRoadTransition
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 -Algorithm td3 -Config configs\carsim_nonlinear_experimental.ini -FocusTargetSpeeds -Curriculum -VaryRoadTransition
```

训练后可用 `tools/select_target_checkpoint.ps1 -RunDirectory <训练目录>` 按固定训练验证工况、60/80 km/h 误差和全网格平均 RMSE 筛选候选，再用 `tools/evaluate_target_scenarios.ps1 -DdpgPolicy <策略> -Td3Policy <策略> -OutputRoot policies\tgt\<短目录名>` 生成两种速度、三种附着道路的对比报告。检查点选择和独立评估使用不同道路；多次查看独立评估后再次调参会逐渐消耗其独立性，应保留新的测试道路。使用短结果路径可避免 Windows 路径过长。

单独检查当前通用策略的跨工况表现，可运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\evaluate_general_policy.ps1 -Algorithm ddpg
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\evaluate_general_policy.ps1 -Algorithm td3
```

每次在私有 CarSim 副本中评估 9 个速度、4 个附着系数，共 36 组，分别运行基础 MPC、RL 中性 MPC 和所选策略。报告展示逐工况 RMSE、最大误差、完成状态与理想侧向需求；评估需要较长时间，不修改 CarSim 原工况。

### 非线性轮胎预测模型（实验配置）

对当前变附着道路的 80 km/h 同条件复现显示，原模型的 RMSE 为 0.606 m，CarSim 内置驾驶员约为 0.226 m；不能仅将原控制器误差归因于附着极限。实验模型使用轮胎表的局部力和受限斜率，以及基于实际状态的受限模式切换。此前 Np=30 时，该道路的 60/80 km/h 固定 MPC RMSE 分别为 0.086/0.184 m；这些结果不证明所有道路和速度均安全，宽速度低附着测试仍有大误差。

`configs/carsim_nonlinear_experimental.ini` 继承同目录的 `default.ini`，仅覆盖实验模型的轮胎和附着保护选项。修改 `default.ini` 的 `[mpc]`、`[weights]` 等参数会被实验配置继承；实验文件中明确写出的选项以实验文件为准，命令行参数再覆盖配置文件。若要复现上述 Np=30 测试，请将 `default.ini` 的 `[mpc] np` 设为 30，或在命令中添加 `--np 30`。每次运行结果目录中的 `effective_config.ini` 记录最终配置。

`[mpc] np` 和 `[weights]` 用于固定 MPC；DDPG/TD3 根据策略输出自适应 Np 和 Q/R，其范围由 `[rl]` 设置，已有策略不会因修改固定权重而重新训练。`nc`、采样周期和约束等公共配置仍会影响自适应控制器。

不改变现有默认模型或注册策略。可显式运行实验版固定 MPC：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --config configs\carsim_nonlinear_experimental.ini --rl-mode fixed --set carsim.sync_browser_result=false --no-open --out results\nonlinear_mpc_trial
```

三控制器对比脚本可用 `-Config configs\carsim_nonlinear_experimental.ini` 选择实验模型；旧 DDPG/TD3 策略是在旧模型上训练的，跨工况测试出现明显退化，不应视为已适配新模型。脚本默认检查策略目录中的 `effective_config.ini`，阻止不同模型直接混用；仅作诊断时可显式加 `-AllowModelMismatch`，并关闭旧区域专家路由。继续训练时可用训练脚本的相同 `-Config`，建议 `-WarmStart <历史Actor>` 继承权重并重新初始化 Critic 与经验池；新模型策略应单独验证，未通过门槛不替换默认注册表。新增 MPC 模型和保护参数纳入训练状态兼容检查。

以下是修正站距停止后重新训练、按目标速度检查筛出的两份实验候选，尚未通过原有严格选优；可以在同一实验模型上生成三组对照：

```powershell
$ddpg = 'policies\iterations\ddpg\cycle_20261029-182031-989\ddpg_train_carsim_NpNominal15_Nc7_20261029-182032\checkpoint_0060_policy.txt'
$td3 = 'policies\iterations\td3\cycle_20261029-182033-362\td3_train_carsim_NpNominal15_Nc7_20261029-182033\checkpoint_0020_policy.txt'
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 -Config configs\carsim_nonlinear_experimental.ini -DdpgPolicy $ddpg -Td3Policy $td3
```

2026-10-27 的 μ=0.4 单附着训练仍保存在 `policies\archive` 和原结果目录中。多工况独立评估除了记录完成状态和误差，还给出 `v²|κ|max/(μg)` 的理想侧向需求估算：当要求的侧向加速度超过可用附着时，固定车速下仅靠横向转向控制不能保证准确跟踪。报告的绿色只表示比基础 MPC 相对更好，不能代表绝对误差达标。

训练 DDPG 自适应 Np/Q/R 策略（默认 200 回合，速度 30–120 km/h、附着系数 0.3–0.9）：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_v16_training_host
```

若需要固定 μ 的对照实验，可将附着系数固定为 μ=0.8，仅对速度分层抽样。`train_mu_min` 与 `train_mu_max` 相等时启用固定附着训练；每 6 回合覆盖一次速度分层，选优只比较同一 μ 下的工况。原来的变附着训练入口和已保存的 DDPG 策略保持有效：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --set rl.algorithm=ddpg --set rl.train_mu_min=0.8 --set rl.train_mu_max=0.8 --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_ddpg_fixed_mu08
```

该命令已实跑 200 回合：`results\incremental_ltv_ddpg_fixed_mu08\ddpg_adaptive_carsim_Np15_Nc7_20261027-195004\`。12 个固定验证工况中没有检查点同时通过奖励和误差门槛，最终保存的是中性 MPC；固定 μ **没有单独解决收敛问题**。逐回合回报跨速度变化很大，应查看 `checkpoint_summary.csv` 与 `training_cycle_summary.csv`，不要把平滑后的曲线当作收敛证据。

需要低附着策略时，单独训练并使用另一个输出目录。加载这种策略时，必须传入相同的附着边界；单次 `eval/compare` 的 CarSim 工况 μ 也必须相同，否则程序报错。该策略仅在对应 μ 上训练，不能把其他 μ 的性能当作已验证：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --set rl.algorithm=ddpg --set rl.train_speed_max_kmh=70 --set rl.train_mu_min=0.4 --set rl.train_mu_max=0.4 --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_ddpg_fixed_mu04
```

μ=0.4、30–70 km/h 的单独实跑在 `results\incremental_ltv_ddpg_fixed_mu04\ddpg_adaptive_carsim_Np15_Nc7_20261027-200347\`，选中第 140 回合。12 个选优工况的平均 RMSE 从约 0.22 m 降到 0.18 m；在 11 个未参与选优的速度点上从 0.2023 m 降到 0.1761 m，降低 12.9%，9/11 个点改善。后续当前策略检查点又回落，所以这仍不是稳定收敛。用于该策略的独立评估命令：

```powershell
$policy = 'results\incremental_ltv_ddpg_fixed_mu04\ddpg_adaptive_carsim_Np15_Nc7_20261027-200347\ddpg_policy.txt'
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode eval_grid --rl-model $policy --set rl.train_speed_max_kmh=70 --set rl.train_mu_min=0.4 --set rl.train_mu_max=0.4 --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_ddpg_fixed_mu04_heldout
```

已完成的严格独立明细在 `results\incremental_ltv_ddpg_fixed_mu04_heldout_strict\ddpg_adaptive_carsim_Np13_Nc7_20261027-203617\heldout_validation.csv`。结果子目录里的 `Np13` 是命令默认固定控制器的命名字段；CSV 中的配对基线实际使用 RL 中性 Np=15。

若继续检验动作维度是否过多，可用可选精简动作模式。它只训练 Np、Qy、Qψ、RΔ；Qvy、Qr、Rsteer 固定在中性值。默认仍为完整七维动作；精简动作策略使用独立 V18 文件标识，加载和评估时须带 `--set rl.action_profile=compact`：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 200 --set rl.algorithm=ddpg --set rl.action_profile=compact --set rl.train_speed_max_kmh=70 --set rl.train_mu_min=0.4 --set rl.train_mu_max=0.4 --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_ddpg_fixed_mu04_compact
```

该精简动作对照已完成 200 回合，但仅选中第 10 回合；11 个独立速度点平均 RMSE 为 0.1991 m，相比中性 MPC 的 0.2023 m 仅降低 1.6%，弱于完整动作策略的 0.1761 m。后段检查点仍回落，因此精简动作**不作为推荐策略**。结果在 `results\incremental_ltv_ddpg_fixed_mu04_compact\ddpg_adaptive_carsim_Np15_Nc7_20261027-204625\` 和 `results\incremental_ltv_ddpg_fixed_mu04_compact_heldout\ddpg_adaptive_carsim_Np13_Nc7_20261027-210131\heldout_validation.csv`。

参考所给 TD3 论文的双 Critic、目标动作平滑及延迟更新时，使用可选 TD3 模式。它与 DDPG 使用相同的增量式 LTV-MPC、动作范围和奖励配置，策略文件格式独立：

```powershell
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode train --rl-episodes 300 --set rl.algorithm=td3 --set rl.model_file=td3_policy.txt --set carsim.sync_browser_result=false --no-open --out results\incremental_ltv_td3_training
```

已完成的 TD3 对照训练在 `results\incremental_ltv_td3_reward_v4\ddpg_adaptive_carsim_Np15_Nc7_20261026-231140\`，第 200 回合策略保存在 `td3_policy.txt`。14 个选优工况平均横向 RMSE 为 0.412476 m，中性 MPC 为 0.470791 m；24 个独立工况分别为 0.320238 m 与 0.330883 m，但部分高速工况退步，当前策略分数仍波动，**不能称为稳定收敛**。此前 DDPG 第 100 回合策略仍保存在 `results\incremental_ltv_v16_training_host\ddpg_adaptive_carsim_Np15_Nc7_20261026-211905\ddpg_policy.txt`。`--out` 是结果根目录，重新训练会另建子目录。

五个随机种子（101–105）分别训练 200 回合后，汇总性能均值、样本标准差与学习曲线：

```powershell
.\tools\run_five_seeds.ps1
.\tools\evaluate_five_seeds.ps1
& 'C:\Users\fengchengmin\.local\bin\python3.14.exe' tools\aggregate_five_seeds.py results\incremental_ltv_v16_five_seeds
```

多种子脚本读取项目中已固定的 CarSim 80 km/h 来源快照；第二步在额外速度和附着组合上评估，结果见 `five_seed_report.html`。DDPG 在所有速度输出独立的 Np 与六个 Q/R 绝对数值；其中 Rsteer 惩罚由增量重构的转角与曲率前馈之差，Rdelta 惩罚转角增量。训练的配对奖励参考控制器使用 RL 自有的中性 Np/Q/R。动作范围、奖励和检查点选择见 [DDPG 说明](docs/DDPG_Adaptive_Horizon.md)。五种子报告汇总独立策略的性能，不平均网络权重。

当前完整动作 DDPG 策略格式为 `DDPG_MPC_V16_INCREMENTAL_LTV`，可选 TD3 策略格式为 `TD3_MPC_V17_INCREMENTAL_LTV`；精简动作另用 V18 格式，三者不能交叉加载。旧 V14/V15 策略对应不同的预测模型和优化问题，也不能直接加载。使用本次 TD3 策略与固定 MPC 对比：

```powershell
$policy = 'results\incremental_ltv_td3_reward_v4\ddpg_adaptive_carsim_Np15_Nc7_20261026-231140\td3_policy.txt'
.\out\build\x64-Release\Release\MPC_DoubleLaneChange.exe --rl-mode compare --rl-model $policy --set rl.algorithm=td3 --set carsim.sync_browser_result=false --no-open --out results\carsim_compare_td3_v17
```

历史 V14/V15 报告只代表当时的控制器。查看本次 TD3 未参与选优的 24 工况明细：`results\incremental_ltv_td3_reward_v4_heldout_neutral\ddpg_adaptive_carsim_Np13_Nc7_20261026-233250\heldout_validation.csv`；虽然结果子目录名称保留了命令行固定 Np=13，CSV 中的固定基线已按训练口径设为 RL 中性 Np=15。奖励和损失公式、分层周期曲线及收敛限制见 [DDPG/TD3 说明](docs/DDPG_Adaptive_Horizon.md)。`tools/convert_v14_policy.py` 是历史 V14→V15 转换工具，不能生成当前策略。

`--out` 指定结果**根目录**；每次运行自动创建唯一子目录，不覆盖旧结果。报告、CSV、有效配置、策略和 CarSim 原生文件保存在本次目录中。若 `sync_browser_result=true`，C++ 闭环结束后会把原生结果同步到对应 CarSim Run Control 的最近结果缓存；在 CarSim 点击 `Refresh` 后可使用 `Video / Video + Plot`。原缓存会备份在本次独立结果目录中。

## 参数与数据来源

- 在 CarSim 设置车辆、速度、道路、附着与参考轨迹。在 C++ 的 [default.ini](configs/default.ini) 设置 MPC 的 Np/Nc、Q/R、转角约束及 DDPG 奖励参数。
- MPC 预测模型的质量、惯量、轴距分量、轮胎侧偏刚度与路面摩擦系数来自本次 `LastRun_echo.par`，运行中的速度来自 CarSim 状态。缺失必要参数会报错。
- 速度图、CSV 与终端摘要统一使用 km/h；MPC 内部使用 m/s。报告中的转角是 C++ 发给 CarSim 的指令，并非测得的实际轮角。
- 当前参考轨迹仅支持全局 X 正方向直路；CarSim 表格的中间点经 C++ 三次样条插值，可能与 CarSim 内部插值有细微差别。

详细接口和运行顺序见 [CarSim_CPP_Integration.md](docs/CarSim_CPP_Integration.md)。

## 固定速度收敛自查与训练图

### 使用当前 CarSim 工况和已保存的固定速度策略

在 CarSim 更新当前工况并运行一次以刷新 `simfile.sim` 后，在项目根目录运行（将 `$speed` 改为当前目标速度 60 或 80）：

```powershell
$speed = 60
$ddpg = "policies\fixed_speed\v$speed\ddpg\ddpg_policy.txt"
$td3 = "policies\fixed_speed\v$speed\td3\td3_policy.txt"
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config "configs\carsim_fixed${speed}_np_focus.ini" `
  -DdpgPolicy $ddpg -Td3Policy $td3 `
  -TrainSpeedMinKmh $speed -TrainSpeedMaxKmh $speed
```

此命令不传 `-Simfile`，读取 `configs/default.ini` 中 `carsim.simfile` 指向的当前工况，而不是 `policies\fixed_speed` 中的历史工况快照。所选策略按各自固定速度训练，当前道路或车辆变化时需要重新验证；运行结果的三组图会自动打开。若 CarSim 的 Simfile 路径不同，可在命令末尾添加 `-Simfile '实际路径\simfile.sim'`。

2026-10-31 后续优化试验的 [60 km/h 训练与诊断报告](policies/audit/optimization_60_20261031/report.html) 对比固定分母基线、较小 Actor 步长和附着需求上限从 3 扩至 4。两项均为可选设置；默认保持 `actor_learning_rate=0.0001`、`grip_demand_clip=3`。上限为 4 时使用 V21 策略格式，加载时必须使用对应配置；从旧 V18 Actor 只能显式热启动，不能直接当作 V21 模型加载。两个参数对应配置见 `configs/carsim_fixed60_audit_slow.ini`、`carsim_fixed60_audit_grip.ini` 与 `carsim_fixed60_audit_slowgrip.ini`。训练默认创建新迭代目录，不覆盖固定策略。

[80 km/h 训练与诊断报告](policies/audit/optimization_80_20261031/report.html) 包含固定分母、小步长和小步长加上限 4 的独立对照。60 km/h 小步长还用种子 984、985、986 各训练 100 回合；种子间仍有退化，不能称为稳定收敛。80 km/h 上限 4 仍约有 15% 决策触及边界，不能认为这种输入对所有速度已足够。原固定策略没有自动替换。

查看本轮 60 km/h 小步长候选的三组图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_fixed60_audit_slow.ini -Simfile policies\fixed_speed\v60\source\simfile.sim `
  -DdpgPolicy policies\iterations\ddpg\cycle_20261031-105254-570\ddpg_train_carsim_NpNominal25_Nc8_20261031-105254\ddpg_policy.txt `
  -Td3Policy policies\iterations\td3\cycle_20261031-105254-774\td3_train_carsim_NpNominal25_Nc8_20261031-105254\td3_policy.txt `
  -TrainSpeedMinKmh 60 -TrainSpeedMaxKmh 60 -OutputRoot policies\comparisons\optimization60
```

80 km/h V21 候选三组图：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_three_controllers.ps1 `
  -Config configs\carsim_fixed80_audit_slowgrip.ini -Simfile policies\fixed_speed\v80\source\simfile.sim `
  -DdpgPolicy policies\iterations\ddpg\cycle_20261031-111918-605\ddpg_train_carsim_NpNominal25_Nc8_20261031-111918\ddpg_policy.txt `
  -Td3Policy policies\iterations\td3\cycle_20261031-111918-734\td3_train_carsim_NpNominal25_Nc8_20261031-111918\td3_policy.txt `
  -TrainSpeedMinKmh 80 -TrainSpeedMaxKmh 80 -OutputRoot policies\comparisons\optimization80
```

2026-10-31 的 60 km/h 奖励和归一化诊断：[训练诊断报告](policies/audit/diagnostic_60_20261031/report.html)。四个配置分别为 `carsim_fixed60_audit_paired.ini`（配对相对奖励）、`carsim_fixed60_audit_fixedscale.ini`（配对奖励使用固定分母）、`carsim_fixed60_audit_direct.ini`（固定尺度块奖励）和 `carsim_fixed60_audit_calibrated.ini`（保留配对奖励，调整航向及峰值权重）。每组目录的 `diagnostic_steps.csv` 是逐控制步奖励分项，`diagnostic_decisions.csv` 是每次决策的归一化状态、实际动作、块奖励与截断标记。诊断分项是未乘 reward_scale、整体损失截断之前的值；失败额外惩罚见训练与决策记录。

八组各完成 100 回合（种子 984）。调整权重的 DDPG 在第 70 回合选中策略，独立复测横向 RMSE 为 0.051908 m，固定 MPC 和原固定 DDPG 分别为 0.061624 m 和 0.053304 m；但该组 90、100 回合检查点退化，因此仍不能称为收敛。诊断显示预测附着需求状态约 9.09% 的决策触及上限，结合当前道路附着与其他状态反算，裁剪前最大约 3.83（该推算只适用于本次工况）；TD3 配对奖励截断主要位于 100–130 m。固定分母 DDPG 的后五个检查点均优于基础 MPC，最大相对均值偏差约 5.29%，尚未通过 5% 门槛；原固定策略未被替换。候选 DDPG、固定 MPC 和原固定 TD3 的三组图在 `policies\audit\diagnostic_60_candidate_compare`。

例如复现 60 km/h DDPG 的权重调整实验，结果会保存在新的迭代目录，不会替换固定策略：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 `
  -Config configs\carsim_fixed60_audit_calibrated.ini -Algorithm ddpg -Episodes 100 -Seed 984 `
  -WarmStart policies\fixed_speed\v60\ddpg\ddpg_policy.txt `
  -Simfile policies\fixed_speed\v60\source\simfile.sim -FocusTargetSpeeds -NoPromotion
```

查看本次固定 60、80 km/h 的对照训练、逐回合回报、无探索检查点误差、Critic 损失和 Actor 代理目标：[收敛自查报告](policies/audit/rl_convergence_20261031/report.html)。报告中各实验均从对应的已固定 Actor 开始，Critic 和经验池重新初始化；不同实验是独立分支，不能把回合数拼成一条连续训练曲线。原固定策略没有被自动替换。自查配置可选整数 Np 对齐、直接块奖励及增强观测；增强观测采用 V20 策略格式，不可直接按旧 V18 配置加载。

2026-10-31 自查的四个 120 回合对照均保留热启动 Actor；四个 60 回合增强观测实验中，仅 60 km/h TD3 选中第 20 回合，后续检查点退化。该检查点复测横向 RMSE 为 0.054036 m，固定 MPC 为 0.061624 m，原固定 TD3 为 0.052966 m；航向 RMSE 分别为 0.937298°、1.000423°、0.972579°。因此它有取舍，不能宣称全面优于原固定策略或已稳定收敛。原固定策略的 60 和 80 km/h 三组复测报告分别保存在 `policies\audit\rl_convergence_pinned60_regression`、`policies\audit\rl_convergence_pinned80_regression`。

重新运行指定固定速度的三组对照（需要对应 CarSim 快照及固定策略）：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_fixed_speed.ps1 -Speed 60
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\compare_fixed_speed.ps1 -Speed 80
```

例如独立运行 60 km/h 的 DDPG 增强观测实验，产生新目录且不推广替换固定策略：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\train_general_policy.ps1 `
  -Config configs\carsim_fixed60_audit_augmented.ini -Algorithm ddpg -Episodes 60 -Seed 984 `
  -WarmStart policies\fixed_speed\v60\ddpg\ddpg_policy.txt `
  -Simfile policies\fixed_speed\v60\source\simfile.sim -FocusTargetSpeeds -NoPromotion
```

80 km/h 和 TD3 分别将配置、快照、热启动文件及 `-Algorithm` 改成对应速度和算法；训练总预算由使用者根据此前已运行的回合数控制。相同工况检查点用于选优，不属于独立泛化验证；回报滚动均值和网络损失下降均不能单独证明闭环收敛。
