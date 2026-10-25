# DDPG 自适应 MPC：预测时域与 Q/R 权重

## 作用与边界

CarSim 模式下，车辆运动、速度和参考轨迹来自指定工况与 VS Solver；C++ 每个周期读取状态，用本次 CarSim 回显参数构造线性 MPC 预测模型，计算前轮转角并回写 CarSim。DDPG 训练和评估使用同一 CarSim 接口。独立车辆仿真模式已移除。

低速预测从 CarSim 测量 `β=atan2(vy,vx)`，在当前短预测时域保持该侧偏角；用 CarSim 回显的前后轴质心距离 `a+b` 计算 `r_kin=vx/(a+b)·tan(δ)`，并保持当前实测横摆角速度相对于上一转角的残差。全局位姿按 `Ẋ=vx·cosψ−vy·sinψ`、`Ẏ=vx·sinψ+vy·cosψ` 推进。这是基于 Peng 等（2025）式 (2) 的实测侧偏补偿运动学预测，并非论文中的 BiLSTM 侧偏预测器。模型融合权重为 `w=t²(3−2t)`、`t=clamp((v_kmh−32)/8,0,1)`；MPC 的离散预测矩阵和仿射项用该权重融合，CarSim 车辆模型本身始终由 Solver 负责。

`[mpc] np` 和 `[weights]` 仅定义固定 MPC。DDPG 每 `decision_steps` 个控制周期输出七个动作：整数 `Np`，以及 Q 的 `q_y/q_psi/q_vy/q_r` 和 R 的 `r_steer/r_delta` 六个绝对数值。`[rl] min_np/max_np/nominal_np` 与六对 `*_min/*_max` 定义独立动作空间；`[rl] terminal_scale` 定义自适应控制器自己的终端 Q 系数。`Nc`、采样周期、转角与转角速率约束仍由共享 MPC 配置控制。更改固定 MPC 的 Np 或 `[weights]` 不会改变同一策略在相同观测下输出的 Np/Q/R；更改共享设置或 CarSim 车辆、道路、轨迹会改变闭环表现。`simulation.csv` 写出每周期实际应用的数值。

默认绝对搜索区间为：Qy 60–166.67、Qpsi 210–583.33、Qvy 0.24–0.667、Qr 4.8–13.33、Rsteer 7–14.29、Rdelta 245–500，Np 10–40（中性值 15）。这些数值由旧版固定权重与倍率区间换算得到，只是初始搜索区间，尚不能称为最优标定。

## 训练实现

Actor 和 Critic 各有两个 64 单元 tanh 隐藏层。观测为 17 维：当前车速、横向与航向误差、侧向速度、横摆速率、当前及前方曲率、曲率变化、上次 Np、按实际路面 μ 归一化的横向加速度需求、上次六个 Q/R 绝对值的归一化表示，以及路面 μ。横向与航向误差用 `asinh` 压缩，保留较大误差之间的区别。Actor 输出七个 `[-1,1]` 连续值，Np 以 RL 自有的 `nominal_np` 为中心取整，六个 Q/R 经各自绝对边界的对数映射解码。训练探索噪声标准差从 0.25 逐回合降到 0.04；评估关闭噪声。经验回放容量 20,000，Critic 每次采样 32 条、Actor 采样 4 条，使用 Adam、目标网络软更新 `τ=0.005` 和折扣 `γ=0.98`。整数 Np 的取整不可导，Actor 更新使用 Critic 对连续动作的梯度，回放记录实际执行的离散动作。

CarSim 训练每回合复制源 Simfile 与 Parsfile 到独立目录，仅修改私有 `Run_all.par` 的 `SPEED_TARGET_CONSTANT`、`MU_ROAD_CONSTANT` 和必要的 `TSTOP`。默认训练 200 回合，在 30–120 km/h 与 μ=0.3–0.9 的 6×5 网格中分层随机抽样，30–45 km/h 区间加密。求解器回显须与本回合请求的速度设定和 μ 一致，否则停止。实际车速由 CarSim 纵向驾驶员及动力学决定，可能不同于设定值。车辆参数、轮胎表和路面摩擦系数由该回合 CarSim Solver 回显提取；每回合重新初始化 Solver。MPC 状态顺序仍为 `[vy,vx,psi,r,Y,X]`：低于 32 km/h 使用实测侧偏角补偿的运动学预测，高于 40 km/h 使用已有的六状态三自由度动力学预测，二者在中间平滑融合。论文 Peng 等（2025）式 (2) 使用侧偏角补偿运动学位姿；本文直接从 CarSim 实测 `vx/vy` 计算侧偏角，不声称复现论文的 BiLSTM 估计器。未知的纵向轮胎驱动力通过 CarSim 实测纵向速度变化在当前工作点估计。

每 10 回合在原工况，以及 30/.3、30/.8、35/.6、36/.8、40/.8、45/.8、50/.8、60/.8、70/.8、75/.6、90/.8、120/.3、120/.9（车速 km/h、路面 μ）十三个工况无噪声验证。初始固定 MPC 策略作为保底检查点；只有原工况完成、完成工况数不低于固定 MPC、平均逐步奖励提高、各工况 RMSE 不超过允许阈值，而且 45 km/h 及以下工况的最大横向误差不超过固定 MPC 的 1.02 倍加 0.005 m 时，才保存学习检查点。若无检查点通过，输出策略仍是固定 MPC，报告会明确指出。训练曲线、逐回合 CSV、检查点验证 CSV、选中策略和 `selected_validation.csv` 保存在单独结果目录。曲率来自原始 CarSim 参考轨迹的当前和前方位置；此训练没有随机生成其他道路曲率，因此不能据此证明策略对任意曲率道路均有效。

## 奖励函数

每一步的奖励为 `−reward_scale × min(加权代价, 120)`。归一化横向误差使用 `Ly(z)=z²`（`|z|≤2`），超出后为 `4|z|−4`；另对超过 `peak_threshold_m` 的绝对横向误差加平方惩罚，使峰值误差在较大范围内仍有梯度。航向误差使用 `L(z)=z²`（`|z|≤1`），超出后为 `2|z|−1`；输入截断于 ±20。侧向速度和横摆速率误差使用有限幅度的平方项。控制项是转角和转角变化；计算与稳定项是 Np 长度、Np 跳变、六个 Q/R 相对各自搜索区间中心的归一化对数偏离及相邻决策变化。控制器进入轮胎受限模式时有较小额外代价；求解失败和未完成路段另有回合惩罚。该奖励不直接鼓励改变 Q/R，策略只有降低车辆状态与控制代价才可能获得更高回报。具体系数以本次 `effective_config.ini` 的 `[rl_reward]` 与训练报告中的公式为准。

每个采样速度和 μ 都另跑一次 RL 自有中性参考 MPC；该参考使用 `nominal_np` 和六个绝对搜索区间的几何中心，不读取固定 MPC 的 Np/Q/R。送入 Critic 的奖励是决策块平均回报与该同工况参考 MPC 平均回报之差，再按 `reward_scale` 归一化、截断在 ±20。`training_rewards.csv` 同时保留原始回报、配对参考回报、配对优势和探索噪声；`paired_advantage.svg` 展示同工况配对优势。`checkpoint_convergence.svg` 每 10 回合在固定的 14 个速度/附着工况无探索评估当前策略，并同时绘制通过误差约束的历史最佳。随机训练回合的原始回报因速度、μ 与回合长度变化，不能单凭曲线是否平滑判断收敛；历史最佳单调也不等于当前策略收敛。仍需检查 `checkpoint_validation.csv` 的误差、完成情况和独立评估。训练覆盖一个范围不等于整个范围都能稳定跟踪。

## 运行与结果

CarSim 的 Simfile、Solver DLL、IMPORT 转向通道配置以及训练和比较命令见 [CarSim_CPP_Integration.md](CarSim_CPP_Integration.md)。用 `--rl-mode train` 训练，`--rl-mode compare --rl-model <本次 ddpg_policy.txt>` 在同一 CarSim 工况运行固定与自适应对照。对照目录的 `comparison_report.html` 汇总结果；`adaptive/report.html` 包含 Np、Q/R 绝对值曲线，`adaptive/simulation.csv` 可逐周期核对实际权重。CarSim 原生 `.vs/.vsb` 独立归档，启用 Browser 同步后可在 Run Control 刷新并回放。

策略文件格式为 `DDPG_MPC_V15_ABSOLUTE`，记录 RL 自有的 Np 范围与中心、六个 Q/R 绝对边界、训练 μ 范围与 Actor 参数。DDPG 在所有速度计算 Np/Q/R 动作；低速动作幅度非零，在 36 km/h 的模型过渡中心附近平滑收紧，至 75 km/h 恢复完整幅度。整数 Np 在微小动作下可能不变。预测模型融合仅影响 MPC 内部计算；36 km/h 是本项目按用户建议设置的过渡中心，并非 Peng 等论文给当前 CarSim 车辆标定的精确阈值。V14 文件不能直接加载；若旧倍率边界互为倒数，且当前六个绝对边界与“旧训练权重×旧倍率边界”一致，可运行 `tools/convert_v14_policy.py` 原样迁移 Actor 权重。否则必须重新训练。结果只能说明对应的 CarSim 车辆、轨迹和已验证工况，不代表车辆极限或其他道路上的性能。

五组独立随机种子训练可在项目根目录运行 `./tools/run_five_seeds.ps1`，默认种子为 101–105，每组 200 回合，新结果存于 `results/absolute_v15_five_seeds`。脚本使用 `results/paper_model_five_seeds_frozen_source` 中已固定的 CarSim Simfile；用户此后在 CarSim Browser 修改工况不会影响五组比较。训练完成后运行 `./tools/evaluate_five_seeds.ps1`，再执行 `python tools/aggregate_five_seeds.py results/absolute_v15_five_seeds` 汇总五个策略在评估工况上的性能均值与样本标准差；网络权重不取平均。旧 V14 结果保存在原目录，不能当作 V15 验证结果。
