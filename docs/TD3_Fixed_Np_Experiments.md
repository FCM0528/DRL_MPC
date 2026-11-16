# 固定Np的TD3动作消融（2026-11-16）

## 已冻结的结论与本轮范围

起始工作树干净，本地最新提交为用户指定的 `4a2e7ed`。原始MPC及强MPC保留；原三维TD3最佳合法模型仍为0.083804 m / 0.770674°。冻结策略010/011为0.076074 m / 0.912591°、0.076756 m / 0.909604°，仅是屏蔽干预，不作为新训练结果。

本轮所有实测仍来自70 km/h、μ=0.8、120 m的同一冻结开发工况，没有独立验证/泛化结论。未运行500回合或DDPG/TQC训练。源工况与原100回合run一致，源Parsfile SHA-256为 `bb878204652e1351e0fa30e7c6c66654ff61539475e4d06846acbacfc198be7e`。

## 命名动作与兼容性

新增 `configs/strong_qy_only.ini`（一维q_y）和 `configs/strong_qy_qpsi.ini`（二维q_y,q_psi），均继承strong_actions3。20维观测、原奖励、网络、学习率、噪声、gamma、权重范围、1.08变化率和车辆约束全部保持原设置，Np固定36，未学习维度填0。

`python/action_mapping.py`集中维护名称映射与模型契约：支持q_y、q_y+q_psi、旧Np+Qy、旧Np+Qy+Qpsi和旧四维。桥接协议不变，始终发送四维归一化数值。manifest新增policy_action_names和action_mapping_version（named_actions_v1或legacy_prefix_v1）；旧manifest缺字段时按原前缀语义解释。相同维数不同名称或版本拒绝加载。C++解析、校验并保存名称元数据；Python完成实际映射。run_policy输出CSV列名也随动作名称变化。

## 恒定Qy对照

真实CarSim扫描目录：`runs/fixed_np_constant_qy_20261116_v1`。固定Np=36、Nc=13、Qpsi=8960、Rdelta=10800，仅改变Qy。全部完赛、QP零失败且符合Ay≤0.85g、Beta≤5°门槛。全指标和每组原始曲线在constant_qy_metrics.csv及子目录。

| 恒定Qy | 横向RMSE m | 车身航向RMSE ° |
|---:|---:|---:|
| 11500 | 0.089276 | 0.894481 |
| 14950 | 0.083750 | 0.908664 |
| 17000 | 0.080840 | 0.910987 |
| 18400（强基线） | 0.078953 | 0.919092 |
| 20000 | 0.077840 | 0.914195 |
| 22000 | 0.076741 | 0.926673 |
| 23920 | 0.075079 | 0.936555 |

按max min(Iy,Iheading)的预先说明准则，在这7个合法候选中选出Qy=20000：相对强MPC横向改善1.41%、航向改善0.53%。未替换原强基线。Qy=23920横向更好、航向更差，必须作为Pareto对照保留。这里的“重调静态MPC”仅指平衡准则选出的离散候选，不代表连续参数空间全局最优。

010的横向与航向均优于恒定Qy=20000（约2.27%/0.18%改善）；它不是简单复现该恒定候选。但010不在两项上支配全部静态候选，例如Qy=23920的横向仍更低；不能由此证明动态策略普遍优于静态最优。

## 两组从头训练

TD3、seed42、7900决策、每790步确定性评价、learning_starts=550、w_y=w_heading=1。两组初始经验池均为0，实际完成100回合、7350次梯度更新；没有resume。完整轨迹保留，包括SB3结束时自动reset产生的第101份起始轨迹，完成回合数以monitor的100为准。

- A：`runs/fixed_np_strong_qy_only_7900_20261116_v1/td3_v70kmh_100episodes_seed42_20261116_153408`。
- B：`runs/fixed_np_strong_qy_qpsi_7900_20261116_v1/td3_v70kmh_100episodes_seed42_20261116_153633`。
- 统一对照：`runs/fixed_np_comparison_20261116_v2/fixed_np_comparison.csv`；[随报告保留的小型指标摘要](archive/fixed_np_metrics_20261116.csv)。v1也保留。

| 控制器 | 检查点步数 | 全程横向RMSE m | 全程航向RMSE ° | 60–80m横向RMSE m | 60–80m横向峰值 m |
|---|---:|---:|---:|---:|---:|
| 原始MPC | — | 0.087240 | 1.014074 | 0.153484 | 0.216294 |
| 强MPC | — | 0.078953 | 0.919092 | 0.146443 | 0.214754 |
| 恒定Qy=20000 | — | 0.077840 | 0.914195 | 0.145899 | 0.215148 |
| 冻结010屏蔽 | — | 0.076074 | 0.912591 | 0.137398 | 0.199885 |
| 冻结011屏蔽 | — | 0.076756 | 0.909604 | 0.139438 | 0.203552 |
| 原三维最佳111 | 1580 | 0.083804 | 0.770674 | 0.168420 | 0.253849 |
| 新Qy-only最佳合法 | **4740** | **0.076195** | **0.900024** | **0.139531** | **0.204537** |
| 新Qy+Qpsi最佳合法 | **6320** | **0.076485** | **0.912872** | **0.141420** | **0.208179** |
| 新Qy-only最终 | 7900 | 0.079948 | 0.861160 | 0.139774 | 0.201672 |
| 新Qy+Qpsi最终 | 7900 | 0.079546 | 0.889614 | 0.141571 | 0.205690 |

区段指标按各次真实post-step样本的X筛选计算；与此前报告0.1m公共网格的空间RMSE定义不同，不应直接逐位比较。所有控制器本表均用同一统计口径。

Qy-only最佳相对强MPC改善 **3.49%/2.07%**，相对重调静态Qy=20000改善 **2.11%/1.55%**；二维最佳对应为 **3.13%/0.68%** 和 **1.74%/0.14%**。两组均保存best_joint_model、best_legal_model、final_model及全部周期检查点。所有周期/最终模型均完赛、QP零失败并通过既定稳定性门槛。

一维最佳峰值Ay=7.12577 m/s²、Beta=2.09308°；二维最佳为7.10483 m/s²、2.06461°。完整峰值误差和稳定性明细见统一CSV。两组最佳检查点的60–80m横向RMSE和峰值均低于强MPC，解决了本工况此前该区段的退化，但不保证其他工况。

## 参数变化与后期退化

Np全程严格36，未学习的Qpsi/Rdelta保留中性。按每决策权重变化超过1%统计，一维最佳Qy跳变频率41.0%，最终67.9%；二维最佳Qy为74.4%、Qpsi为100%，最终约73.1%/89.7%。一维最佳Qy标准差约2591，二维Qy/Qpsi约1889/1473，均不是恒定策略。

两组最终模型都失去强基线双指标达标：相对强MPC横向退化1.26%/0.75%，航向改善6.30%/3.21%。其60–80m横向改善仍保留，说明后期全程退化不能仅归咎该区段。训练前8/后8回合回报均值分别为一维−33.30/−31.31、二维−33.40/−32.19，回报总体提升与双指标失守并存。必须使用周期检查点，不能用final替代。

本轮优先保留**一维Qy-only**作为独立验证候选：它在本次种子上两项RMSE都优于二维最佳，映射更简单且调参跳变更少。仅一工况、一个种子不足以判定维数的普遍优劣，二维配置、模型及Pareto全部保留。后期退化在固定Np后仍出现，下一轮可单独比较Actor LR=1e-4与当前3e-4；或另开一轮比较sigma=0.10与0.15。每轮仅一个因素，其余预算、奖励及动作范围不变，不延长为500回合。

## 待生成的真实工况

当前可核实冻结来源仍只有70 km/h、μ=0.8工况。建议由用户在CarSim复制到新的真实数据集，生成至少训练60/70 km/h、验证65/75 km/h、最终测试另一组预先锁定速度/附着组合；每组均导出完整Simfile和Run_all.par并保留相同车辆/参考轨迹核查记录。具体速度与附着需确认工况可完成及物理可行后预注册，不能将建议值冒充已存在工况。训练与验证/测试集合用scenario_sets做内容和物理签名去重；只改文件名/注释无效。

## 复现命令与验证

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'
python python/parity.py --config configs/strong_qy_only.ini --blocks 200 --out runs/parity_qy_NEW
python python/parity.py --config configs/strong_qy_qpsi.ini --blocks 200 --out runs/parity_qyqpsi_NEW
python python/fixed_np_experiments.py scan --simfile runs/strong_td3_100_v1/td3_v70kmh_100episodes_seed42_20261116_141614/source_case/simfile.sim --out runs/constant_qy_NEW
# 两个配置分别运行，均不使用resume
python python/train.py --algorithm td3 --config configs/strong_qy_only.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --simfile runs/strong_td3_100_v1/td3_v70kmh_100episodes_seed42_20261116_141614/source_case/simfile.sim --seed 42 --episodes 100 --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --out runs/qy_only_NEW
python python/train.py --algorithm td3 --config configs/strong_qy_qpsi.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --simfile runs/strong_td3_100_v1/td3_v70kmh_100episodes_seed42_20261116_141614/source_case/simfile.sim --seed 42 --episodes 100 --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --out runs/qy_qpsi_NEW
python python/strong_report.py <训练run目录>
python python/fixed_np_experiments.py compare --scan <恒定扫描目录> --run-a <一维run目录> --run-b <二维run目录> --masks runs/td3_action_masks_20261116_v1 --out runs/compare_NEW
```

本机用uv及python/requirements.txt执行以上Python入口。Release构建、CTest 1/1、pytest 40/40通过；名称映射及旧前缀映射、同维不同语义拒绝加载、新配置中性参数均有测试。两份新配置各通过完整157控制步CarSim parity。完整旧动作兼容性与回放入口检查见清理审计。

新代码为action_mapping.py、fixed_np_experiments.py和project_cleanup.py；修改env/train/evaluate/run_policy的动作契约及C++配置元数据，控制器优化结构未改变。历史源码、配置、模型与实验数据按清理审计保留。
