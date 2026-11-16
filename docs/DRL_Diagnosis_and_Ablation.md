# DRL–MPC 诊断与消融记录

## 冻结事实（2026-11-16）

- `configs/default.ini` 的固定 MPC 为 Np=40、Qy=11500、Qpsi=2800、Rdelta=10800。`configs/carsim_python_rl.ini` 的五档中心与其相同。未修改固定 MPC、CarSim 原始工况、车辆参数、参考轨迹或 MATLAB。
- `src/mpc.cpp` 中 `track_course=true` 时，MPC 第二项追踪航迹角（车身航向角加侧偏角）；该语义保持原样。
- 旧奖励 `src/rl.cpp` 在 `tracking_v2` 且 `track_course=true` 时把 `epsi` 改成航迹角误差。`w_yaw_error` 的实参来自 `src/rl_bridge_main.cpp` 的 `r-vx*curvature`，单位 rad/s，是横摆角速度误差，不是车身航向误差。新模式单独使用 `ey`、`wrap(psi-psi_ref)`、`wrap(psi+beta-psi_ref)`；横摆角速度差单独记录。
- `python/evaluate.py` 原有 `rmse_heading_deg` 取 CSV 的 `epsi_rad`，对应车身航向误差。旧版 `python/train_four.py` 使用 `--training-only`，因此只收集 final_model；旧版单模型 `EvalCallback` 以回报或横向 RMSE 选模，不保证两项同时改善。
- `python/train.py` 把 DDPG/TD3 Actor、Critic LR 分别传给算法构造器；探索噪声由配置的 sigma 控制。TQC 使用 `tqc_drop`、单独 Actor/Critic/entropy LR、`ent_coef` 和 `target_entropy`。本次没有改动第三方库或预设算法排名。

## 已实施

- `joint_tracking` 是独立奖励版本；原 INI 默认 `legacy`。候选比例为 `w_y=1.0`、`w_heading=1.0`、`ey_scale_m=0.20`、`heading_scale_rad=0.035`，不是最优结论。奖励轨迹新增三个损失分项，可求全程均值及占比。保留转向幅度/速率、侧偏超限、峰值横向误差及决策级 QP/未完赛惩罚。
- `continuous_weights` 对 Qy/Qpsi/Rdelta 使用以五档中心为零动作的有界对数映射，每决策变化受 `max_weight_ratio_per_decision` 限制；Np 仍为整数。请求动作及执行参数写入桥接 CSV。原配置继续使用五档 `legacy`。
- `road_preview_joint` 为 20 维，在原 14 维后加入车身航向误差、侧偏角、横摆角速度误差及三个已执行权重。Python 和 C++ 分别校验长度与有限值。训练 manifest 记录奖励/动作模式；恢复与评估检查签名。
- `train.py --joint-selection` 对周期保存的检查点及 final_model 做确定性验证。仅完赛且零 QP 失败者有资格；仅两项 RMSE 都优于配套固定 MPC 者标记 dual improvement，并按较小相对改善排序。未达标时记录 Pareto 候选，不生成虚假的 best_joint_model。`train_four.py` 已调用该流程。
- `control_sensitivity.py` 在相同 CarSim 工况上一次只改变 Np、Qy、Qpsi 或 Rdelta，输出转角指令与实际轮角量级、横向/航向 RMSE 与峰值、QP 失败及弱响应标记。其 CSV 保留每组独立运行目录。

## 消融矩阵与顺序

1. 固定 MPC 与原 `carsim_python_rl.ini`；先确认零动作一致性。
2. `carsim_python_rl_joint_legacy14.ini`：只换联合奖励，保留五档和 14 维。三算法用相同预算。
3. `carsim_python_rl_joint_continuous14.ini`：只换连续权重。
4. `carsim_python_rl_joint.ini`：再换 20 维观测；必须从头训练。
5. 每个单元采用种子 42–46，分别报告逐种子和均值/标准差。TQC drop 0/1/2/3、Actor LR 1e-4/3e-4、entropy auto/-2.0；TD3/DDPG Actor LR 1e-4/3e-4、sigma 0.10/0.15。每算法超参搜索预算需先等额固定，再比较验证集选出的策略。固定动作、随机动作和状态置乱可检验策略的状态依赖。

`configs/rl_training_pilot.ini` 是 8 回合、小规模试验入口。**未启动** 15 组 500 回合 CarSim 训练。训练、验证、最终测试必须使用不同冻结工况；`train_four.py` 可用 `--validation-simfile` 与 `--test-simfile` 分别指定。未提供时，最终比较仅是训练工况重复，不应写成独立测试或泛化结论。划分并冻结独立测试工况后，才可运行正式多种子实验。

## 已验证与待验证

- Release CMake 构建、CTest（1/1）、Python pytest（20/20）、桥接 `--self-test-np` 已在本机通过；C++ 测试覆盖联合奖励跨 ±π、beta 解耦、连续零动作、小动作连续性和 20 维有限观测。
- 已用真实 CarSim DLL 运行 `runs/control_sensitivity_20261116_101214/sensitivity.csv`：12 组均完赛、零 QP 失败。Np=28/40/52 的横向 RMSE 为 0.1086/0.0872/0.1020 m，航向 RMSE 为 0.734/1.014/1.448°；Qy=8050/11500/14950 的横向 RMSE 为 0.0947/0.0872/0.0817 m，航向 RMSE 为 1.007/1.014/1.040°。Qpsi 在 1960–3640 内的横向 RMSE 跨度仅 0.00013 m、航向跨度 0.0060°；Rdelta 在 7560–14040 内分别仅 0.00019 m、0.0078°。两项动作在此工况下影响极弱，尚需检查转向饱和及其他工况。`assessment.txt` 的弱响应阈值仅用于筛查。
- 新联合配置经 20 个控制步真实 CarSim 零动作闭环 parity 通过，结果位于 `runs/joint_zero_parity_20261116_101139`。该检查仅证明固定参数下桥接控制一致，不证明 DRL 提升。
- 联合奖励各损失均值及比例、正式种子统计、分段指标、全量真实闭环训练轨迹与独立测试尚未生成。训练运行时会生成 `joint_validation_pareto.png`，但当前未启动训练，所以还没有该图；已有 `python/report.py` 可绘制控制轨迹，需在冻结测试工况后补充全部指定曲线和分段统计。

## 命令

```powershell
& 'D:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe' --build out/python-rl-build --config Release
& 'D:/Program Files/Microsoft Visual Studio/18/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/ctest.exe' --test-dir out/python-rl-build -C Release --output-on-failure
python python/control_sensitivity.py --controller out/python-rl-build/Release/MPC_DoubleLaneChange.exe --config configs/default.ini --simfile <冻结工况.sim> --out runs/sensitivity_<唯一标识>
python python/train_four.py --config configs/carsim_python_rl_joint_legacy14.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --validation-simfile <验证工况.sim> --test-simfile <保留测试工况.sim> --out runs/pilot_joint14
```

所有新输出目录应使用不存在的唯一名称。验证完成前不得据此宣称 TQC、TD3、DDPG 的优劣排序。
