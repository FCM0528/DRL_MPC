# 强 MPC 与 DRL 实验记录（2026-11-16）

## 审计与配置

本轮开始时工作树干净，最近提交为 `1d8e1e8 Add joint feasibility analysis and action dimension ablations`。以该本地提交和实际程序为准审查。未覆盖旧模型、原配置、历史 runs 或 MATLAB。`configs/default.ini` 保持原样。

`configs/mpc_strong.ini` 继承 default，只覆盖 Np=36、Nc=13、Qy=18400、Qpsi=8960、Rdelta=10800。真实 CarSim 回放（`runs/strong_replay_20261116_v1`）得到横向 RMSE **0.07895276061694506 m**、车身航向 RMSE **0.9190924968426759°**，完赛、零 QP 失败，与扫描相同。

新增 `configs/strong_actions2.ini`、`strong_actions3.ini`、`strong_actions4.ini`。中性值分别为 Np=36、Qy=18400、Qpsi=8960、Rdelta=10800；Qy 范围 11500–23920，Qpsi 范围 4480–13440，Np=28–44，Rdelta 保留原第四维范围。Nc、其余 MPC 参数继承强基线。三份配置均通过全程 120 m、157 控制步零动作一致性验证，目录为 `runs/strong_actions{2,3,4}_parity_20261116_v1`。新模型必须从头训练。

## 参数变化率响应

`python/strong_response.py` 对相同的正向、负向、回中动作序列，分别运行 1.08、1.15、1.20。真实数据位于 `runs/strong_slew_response_20261116_v2/response_metrics.csv`；此前 v1 因配置继承路径约束失败，保留其日志，修复后使用新目录。

| 比率 | Qpsi 首次到达正向目标时间 | 转角逐控制步差分 RMS | 峰值 Beta | 峰值 Ay | QP失败 |
|---|---:|---:|---:|---:|---:|
| 1.08 | 0.44 s | 0.005871 rad | 1.952° | 7.045 m/s² | 0 |
| 1.15 | 0.20 s | 0.005935 rad | 1.945° | 7.039 m/s² | 0 |
| 1.20 | 0.20 s | 0.005943 rad | 1.949° | 7.043 m/s² | 0 |

全部完赛。时间由桥接 post-step 采样确定，分辨率为控制周期 0.04 s；这些数值仅对本次动作幅度和工况成立。较快变化的转角差分略增，尚无理由替换默认 1.08。

为补齐 QP 时延明细，在新目录 `runs/strong_slew_response_20261116_v3` 重复同一动作试验并导出完整控制报告，性能数值复现。平均求解时延为 0.212/0.216/0.242 ms，95分位为 0.256/0.268/0.354 ms，最大迭代数均为7。时延含操作系统负载变化，单次试验不足以判定比率对耗时的因果影响。

## 工况集合及预算

`python/scenario_sets.py` 接收 JSON 中 train、validation、test 的真实 Simfile 路径列表。每份 Simfile/Parsfile 保留完整源哈希并冻结；还保存速度、道路附着描述、LTARG 参考哈希和物理内容哈希。物理哈希移除注释、日志与数据集标签，保留数值指令，拒绝跨集合仅改标签的工况。它不能替代人工确认车辆、路径及工况语义；复杂外部引用仍需在 CarSim 导出完整 Run_all.par 后核查。

当前已核实可用来源只有 70 km/h、μ=0.8 工况，清单见 `configs/scenario_inventory_current.json`；验证和测试集合留空。没有制作虚假的可运行 CarSim 文件。请在 CarSim 中复制原 Run Control 到新数据集，分别设置有意义的车速/附着组合，运行并导出完整 Simfile 和 Run_all.par；核对车辆与 LTARG 后，将这些既有文件路径加入 JSON。建议先固定训练/验证/测试的车速与附着划分，再生成清单。不得将同一内容重命名后分到不同集合。

`train.py --scenario-set` 每次 reset 使用 Gym 自带种子随机数发生器均匀抽取训练工况，序列记录在 `episode_scenarios.jsonl`。训练仍是实时 CarSim 闭环。`--total-decisions` 提供精确交互预算（要求为 train_freq 的整数倍）；不提供时保留回合停止接口。所有指定验证工况按决策步周期确定性评价；测试集合不参与选择。

## 双基线模型选择

`python/joint_validation.py` 为每个验证工况运行原始 MPC 和强 MPC，保存两套相对改善率、单指标改善和双指标改善。合法要求完赛、零 QP 失败、峰值 Ay≤0.85g、Beta≤5°。强基线通过要求每个工况两项 RMSE 均优于强 MPC，以所有工况中最小的相对改善排序。

每算法保存 final_model、best_legal_model、全部周期检查点、joint_validation.csv、checkpoint_selection.json 和 Pareto 图。仅当强基线通过才生成 best_joint_model。final_model 只诊断。一个算法不达标不会删除或隐藏其他算法的 CSV。`joint_selection.py` 保留旧接口并支持双基线集合评价；train_four 支持集合及固定步预算，测试改由独立锁定模型入口执行。

## 已执行 TD3 小规模实验

种子统一为 42，奖励、网络、噪声及优化器配置使用 `rl_training_pilot.ini`；学习预热 550 决策。8 回合功能测试完成 632 决策、81 次更新。随后强起点三维训练完成 100 回合、7900 决策、7349 次更新。再运行原始起点三维及强起点二维、四维，每组也实际完成 100 回合、7900 决策。后面三组用固定步预算，目录名称中的 8episodes 来自旧回合参数，仅为标签；实际回合以 monitor 为准。

汇总目录：`runs/strong_pilot_comparison_20261116_v1`。四组均保存完整模型与逐检查点记录。以下为各组 best_legal 检查点的同工况诊断：

| 起点与动作 | 横向 RMSE m | 航向 RMSE ° | 超过强 MPC 双指标 |
|---|---:|---:|---|
| 原始 MPC，3维 | 0.089059 | 0.908145 | 否 |
| 强 MPC，2维 | 0.089934 | 0.708963 | 否 |
| 强 MPC，3维 | 0.083804 | 0.770674 | 否 |
| 强 MPC，4维 | 0.089997 | 0.719455 | 否 |

强三维最佳合法检查点在 1580 决策：相对原始 MPC 两项均改善；相对强 MPC 横向退化约 6.14%、航向改善约 16.15%。最终模型为 0.100796 m、0.689212°，横向进一步退化。前8/后8回合平均回报从 -33.03 降为 -35.67，不能认定稳定有效学习。

该最佳策略 Np 在 29–39、Qy 在约 14244–23919、Qpsi 在约 7785–11678 变化，未收敛成常数。其 Qy 频繁接近上限，参数每决策变化，但这不足以证明动态自适应带来额外收益。平均奖励损失：横向 0.17558、航向 0.14769、增长 0.04832、参数切换 0.00000887；切换项比跟踪项小几个数量级，需结合动作曲线继续检查抖动。

100回合输出在 `runs/strong_td3_100_v1/td3_v70kmh_100episodes_seed42_20261116_141614/`：`analysis/` 含回报、Actor/Critic loss、ey/航向、参数、请求动作与目标/执行权重曲线，`joint_validation_pareto.png` 为检查点双误差图，`joint_validation.csv` 含两基线改善率、峰值、QP、Ay、Beta，`validation_traces/` 含各检查点原始轨迹。其他三组也有对应 analysis 目录。

所有小试使用当前单个训练工况重复诊断，**没有独立验证/测试或泛化结论**。本轮不宣称三算法性能排名。

## 最小后续修改

优先检查奖励在强 MPC 附近的方向性：降低航向损失可能补偿横向退化，而模型验收要求两项均改善。建议下一轮仅新增一份版本配置，加入相对强 MPC 横向容差的超额惩罚，或仅调整 w_y；维持动作、网络、预算和种子不变，先做短验证。切换惩罚可在另一轮单独增加一个数量级，避免同时改变两个因素。需要先补独立验证工况，再决定这些候选；不增加到1000回合。

## 命令

```powershell
# 单工况诊断（本轮已运行；使用新的输出根目录复现）
uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm td3 --config configs/strong_actions3.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --episodes 8 --eval-frequency 200 --joint-selection --training-only --out runs/strong_smoke_NEW
uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm td3 --config configs/strong_actions3.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --out runs/strong_100_NEW
# 真正多工况：sets.json 中须有已导出的真实工况
uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm td3 --scenario-set sets.json --simfile <训练集首工况.sim> --config configs/strong_actions3.ini --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --out runs/multicase_NEW
# 正式三算法计划：默认只生成计划，不训练；固定种子42–46，每算法一个相同预算超参试验
uv run --no-project --with-requirements python/requirements.txt python python/formal_compare.py --scenario-set sets.json --total-decisions 40000 --out runs/formal_plan_NEW
# 最终测试：lock.json 为 algorithm/model/sha256/config 的列表，事先锁定后才读取保留测试集
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --lock lock.json --scenario-set sets.json --out runs/heldout_NEW
```

正式脚本需要物理不同且非空的 train/validation/test 集合。`--execute` 才会运行正式计划，本轮未执行。默认每算法一个预先固定超参试验；如扩展搜索，必须为每算法事先分配相同试验数和交互预算。最终测试只读取锁定模型，不挑选检查点，输出逐种子/工况数据、均值、标准差与失败率。

## 测试与变更文件

本轮 Release 构建、CTest 1/1、pytest 28/28 通过。三份强动作配置的完整 CarSim parity 通过。真实 CarSim 已完成强基线回放、三组变化率响应、8回合功能测试、100回合三维及三组等预算对照。未运行真实多工况训练、独立测试、DDPG/TQC正式比较，原因是尚未锁定足够的独立工况。

新增：`configs/mpc_strong.ini`、`strong_actions{2,3,4}.ini`、`scenario_inventory_current.json`；`python/scenario_sets.py`、`joint_validation.py`、`strong_response.py`、`strong_report.py`、`evaluate_dual.py`、`formal_compare.py`；集合及双基线选择测试。修改：`python/train.py`、`train_four.py`、`joint_selection.py`、`envs/carsim_mpc_env.py`、`tests/verify.cpp`。默认配置、历史模型与结果均保留。
