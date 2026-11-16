# DRL–MPC研究主线恢复（2026-11-16）

## 论文主线与保留成果

起始工作树干净，HEAD=cf29149。恢复主线：**三自由度车辆预测模型+增量式LTV-MPC，以TQC进行MPC参数自适应，并与TD3、DDPG统一比较**。CarSim是高保真被控对象；三自由度指纵向、侧向、横摆动力学，当前预测模型保留轮胎松弛扩展状态。DRL只提出MPC参数请求，不输出转向；转向仍由同一C++ QP求解。

1D/2D TD3模型、五种子、奖励与屏蔽消融作为补充实验保留，不再是唯一正式方案。原始MPC、强MPC、旧Np+Qy+Qpsi三维、旧四维及全部模型、runs、MATLAB和转向机器人设计未删除或覆盖。本轮没有清理目录，也没有重扫45+26组参数。

## 统一动作设计

| Profile | 配置 | 策略动作 | 定位 |
|---|---|---|---|
| qy1 | strong_qy_only.ini | Qy | 1D消融，原结果可复现 |
| weights2 | strong_qy_qpsi.ini | Qy,Qpsi | 2D消融 |
| weights3 | **strong_weights3.ini** | **Qy,Qpsi,Rdelta** | 首个三算法统一预实验 |
| full4 | **strong_full4.ini** | **Np,Qy,Qpsi,Rdelta** | 完整自适应方案，Np受限 |
| 历史 | strong_actions3.ini | Np,Qy,Qpsi | 原语义保留，不覆盖 |

四套新/现有维数方案共享强MPC中性值Np=36、Qy=18400、Qpsi=8960、Rdelta=10800，Nc=13；Qy范围11500–23920，Qpsi为4480–13440，Rdelta为7560–14040；连续有界对数映射，每决策权重比率最多1.08。观测20维、联合奖励、采样0.04s、决策0.08s均相同，测试比较有效配置确认除动作名称、维数和Np变化限制外无差异。

C++桥接继续接收[Np,Qy,Qpsi,Rdelta]四维归一化请求。三维权重映射明确填Np=0，不能误加载历史Np前缀三维模型。旧配置未声明名称时仍走原映射；同维不同语义由manifest检查拒绝。

四维Np范围仍28–44、整数步长1，每决策最多变化**2步**。原四维映射、范围、观测和奖励可复用，但新限速配置是独立实验，旧模型应配原冻结配置运行，不能直接resume后当作同语义训练。限速不是对Np性能的保证，也不删除Np的研究地位。

## Np与Rdelta现有证据

真实Np单独响应：全程Np到达28和44，观测每决策最大跳变2；完赛、QP零失败，峰值Ay=7.0017m/s²、Beta=1.8216°。横向RMSE=0.093080m、航向RMSE=0.866079°。该人为切换序列仍存在横向/航向权衡，说明硬限速约束执行，不会自动获得更优跟踪。原冻结策略动作屏蔽中Np的负向横向作用只针对那个策略和工况，不能外推为所有Np自适应无效。

Rdelta保留原作用弱的证据：此前原始MPC预测代价中Rdelta约10.81，而Rdelta_change约349.34；1080–54000单因素扩展仍仅造成小输出变化。本轮没有放大Rdelta边界、改变奖励或削弱其他MPC项来制造作用。三维预实验允许其参与学习，但“动作变化”本身不是有效控制收益的证明。

## 统一预实验协议

真实70km/h、μ=0.8、120m工况，三算法各seed42、7900决策、550预热、每790步确定性评价，从随机网络和独立空经验池开始，完整CarSim在线交互。统一源Parsfile SHA-256为bb878204652e1351e0fa30e7c6c66654ff61539475e4d06846acbacfc198be7e；固定与强MPC、奖励、动作边界及选模规则相同。

使用原rl_training_pilot.ini的算法设计：DDPG/TD3 Actor与Critic LR=3e-4，sigma=0.15；TD3双Critic、policy_delay=2。TQC Actor LR=1e-4、Critic LR=3e-4、entropy LR=1e-4，3个Critic×25分位、每Critic截断2、熵系数/目标熵auto。原网络及各算法更新方式保持。环境交互预算相同，算法内部更新结构遵循各自原理，没有为预期排名更改评价标准。

只以完赛、零QP、Ay≤0.85g、Beta≤5°为合法；双指标达标要求两项RMSE都优于强MPC，按min(Iy,Iheading)选周期模型。final只诊断。当前仍是**同工况技术预实验**，不是独立验证、泛化或正式算法排名。

## 状态反馈与扰动验证计划

此前一个确定性工况上Qy(X)复现一维TD3，只说明位置表能复现那一条轨迹，不能断言DRL没有反馈价值。下一步固定道路/LTARG、车辆、附着和模型，分别从CarSim真实生成并核验下列扰动，不能修改已有CSV或伪造车辆状态：

| 单因素 | 预注册候选条件 | 其余条件 |
|---|---|---|
| 初始横向误差 | 0、±0.10、±0.20m | heading=0，70km/h |
| 初始航向误差 | 0、±1°、±2° | ey=0，70km/h |
| 合理速度变化 | 65、70、75km/h | ey=heading=0 |

这些是待生成设计，不是已存在或已通过的工况。CarSim中复制新数据集，保留当前2输入/20输出及车辆模型，在Solver echo与初始状态核查实际扰动。当前工作区只有70km/h来源满足当前I/O契约；历史60/80为3输入/15输出，不直接混入。需要重新真实导出兼容工况。

锁定原一维模型及其位置表作为既有对照；三维权重方案也应在开发工况完成后一次性锁定模型和对应三权重位置表。所有扰动测试中**不重新训练策略、不重新拟合位置表、不重选静态参数**。固定MPC、强MPC、固定位置调度、状态反馈策略在每个条件下重新运行真实闭环，报告误差、峰值、稳定性、QP及计算时延。只有这些对照才能支持扰动下的反馈优势结论。

已在新结果目录保存disturbance_lock.json，锁定三算法best模型、配置哈希及从开发回放采样的三权重位置表SHA-256，同时登记原一维固定表哈希。它只锁模型/表和待生成条件，没有创建物理Simfile；扰动回放仍待真实工况就绪。

## 正式论文实验与最小下一步

正式共同基线为固定MPC、优化静态MPC、DDPG-MPC、TD3-MPC、TQC-MPC。主结果同时报告横向/航向RMSE与峰值、Beta/Ay、转向平顺性、QP失败与计算时延；独立工况与种子42–46就绪后报告均值/样本SD和失败率。1D/2D是动作消融，3D权重与受限4D用于检验Rdelta和Np实际作用，不预设算法优劣。

formal_compare现按profile选择动作，默认weights3，qy1保留一维复现；显式旧--config strong_qy_only.ini会推断qy1。仍要求独立工况、双基线证明与验证超参锁，不能绕过预检启动15组训练。已生成weights3、full4及qy1计划，均只输出未就绪预检，没有执行正式训练。

最小顺序：①保留本轮统一结果及检查点；②在CarSim导出接口兼容的扰动与独立工况，先双MPC回放；③锁定策略/位置表做扰动对照；④再决定三维/四维及TQC超参的等预算多种子实验。若继续出现后期退化，只做单因素消融，不同时改变奖励、网络与动作范围。

## 复现入口

```powershell
$env:OMP_NUM_THREADS='1'; $env:MKL_NUM_THREADS='1'; $env:PYTHONDONTWRITEBYTECODE='1'
python python/parity.py --config configs/strong_weights3.ini --blocks 200 --out runs/weights3_parity_NEW
python python/parity.py --config configs/strong_full4.ini --blocks 200 --out runs/full4_parity_NEW
# algorithm分别为ddpg/td3/tqc，三组同预算；不resume
python python/train.py --algorithm tqc --config configs/strong_weights3.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --simfile runs/fixed_np_strong_qy_only_7900_20261116_v1/td3_v70kmh_100episodes_seed42_20261116_153408/source_case/simfile.sim --seed 42 --episodes 100 --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --out runs/weights3_tqc_NEW
python python/strong_response.py --config configs/strong_full4.ini --exercise np --ratios 1.08 --simfile <同一冻结工况.sim> --out runs/np_response_NEW
# 四维小试：algorithm分别td3/tqc，632决策、eval-frequency316、episodes8
python python/train.py --algorithm tqc --config configs/strong_full4.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --simfile <同一冻结工况.sim> --seed 42 --episodes 8 --total-decisions 632 --eval-frequency 316 --joint-selection --training-only --keep-training-traces --out runs/full4_tqc_smoke_NEW
python python/formal_compare.py --profile weights3 --scenario-set configs/scenario_inventory_current.json --total-decisions 7900 --out runs/mainline_plan_NEW
```

本轮实际使用uv与python/requirements.txt。--export-report扩展现有run_policy，可导出C++完整轨迹、QP时延和策略推理时延；旧调用默认行为保持。没有新增训练框架。

## 本轮实际结果与交付

三算法均完成100回合、7900决策、7350次梯度训练迭代，初始回放池0。原始MPC为0.087240m/1.014074°，强MPC为0.078953m/0.919092°。统一结果如下：

| 算法 | 模型 | 检查点 | 横向RMSE m | 航向RMSE ° | 横向峰值 m | 航向峰值 ° | 强MPC双改善 |
|---|---|---|---:|---:|---:|---:|---|
| DDPG | best | joint_2370_steps.zip | 0.078177 | 0.903339 | 0.211287 | 2.920882 | 是 |
| DDPG | final | final_model.zip | 0.079202 | 0.899322 | 0.215831 | 2.906642 | 否 |
| TD3 | best | joint_6320_steps.zip | 0.078778 | 0.903468 | 0.215249 | 2.905369 | 是 |
| TD3 | final | final_model.zip | 0.079513 | 0.887651 | 0.204778 | 2.840720 | 否 |
| TQC | best | joint_4740_steps.zip | 0.078770 | 0.918247 | 0.212993 | 2.943595 | 是 |
| TQC | final | final_model.zip | 0.079435 | 0.907356 | 0.218129 | 2.903672 | 否 |

三组best都略优于强MPC，但final都失去双指标达标。TQC best相对强MPC改善约0.23%/0.09%，没有在本轮显示优于DDPG/TD3的证据；收益很小，不能当作显著性或泛化结论。保留实际结果，不强制制造TQC排名。

| best回放 | 峰值Ay m/s² | 峰值Beta ° | 转角速率RMS °/s | QP均值/p95 ms | 策略推理均值/p95 ms |
|---|---:|---:|---:|---|---|
| DDPG | 7.0938 | 2.0560 | 9.0911 | 0.249/0.342 | 0.534/0.661 |
| TD3 | 7.0927 | 2.0732 | 8.8595 | 0.249/0.357 | 0.553/0.741 |
| TQC | 7.0863 | 2.0779 | 8.3732 | 0.254/0.359 | 0.753/0.961 |

上述best/final均完赛、QP零失败，符合共同Ay/Beta门槛。转角速率按相邻指令差/0.04s计算。时延为训练结束后的顺序回放实测局部CPU耗时，受本机负载影响，不作微小时差排名，也不是CarSim联仿端到端实时性证明。三算法新回放RMSE均与对应保存验证记录在1e-9内一致。

受限4D的TD3/TQC各完成632决策、8回合、82次梯度训练迭代，仅属技术小试：

| 算法 | final横向RMSE m | final航向RMSE ° | 确定性final的Np范围 | 最大相邻决策Np变化 | 完赛/QP |
|---|---:|---:|---|---:|---|
| TD3 | 0.079637 | 0.891870 | 33–38 | 1 | 是/0 |
| TQC | 0.078884 | 0.919853 | 36–36 | 0 | 是/0 |

TQC小试尚未学出明显Np调度；少量更新不能代表四维训练的最终表现。动作响应试验已单独证明Np端点可达及2步限速生效，不能把恒定Np策略当作映射失效。

八回合在线训练阶段，TD3/TQC各记录1256个控制步，Np分别覆盖30–42、29–43，最大相邻决策变化均为2；QP失败均为0，峰值Ay约7.122m/s²、Beta约2.15°以内。训练探索确实能作用于Np，但尚无四维长期学习收益结论。

产物：

- 三维训练：runs/mainline_weights3_{ddpg,td3,tqc}_20261116_v1，各自包含manifest、全部周期模型、best_legal/best_joint、final、验证CSV和完整训练轨迹；analysis目录由现有strong_report生成。
- 四维小试：runs/mainline_full4_{td3,tqc}_smoke_20261116_v1；动作响应：runs/mainline_np_response_20261116_v1。
- 统一指标/来源/黑白曲线：runs/mainline_recovery_summary_20261116_v1；附[小型指标摘要](archive/mainline_recovery_metrics_20261116.csv)。
- 训练结束后回放：runs/mainline_replay_{algorithm}_{best_legal_model,final_model}_20261116_v2，包含实际动作、执行参数、MPC代价分项及QP/推理时延。此前v1回放保留，不用其并发训练时延作比较。
- 零动作检查：mainline_strong_weights3_parity_20261116_v1与mainline_strong_full4_parity_20261116_v1，各全程157控制步通过。

本轮Release构建、CTest 1/1、pytest 54/54通过。问题已解决：主线和维度定义恢复、旧三维不混用、四维Np限速、三算法等交互预算实际运行、统一双指标选模与时延输出。仍待解决：后期退化、多种子TQC/TD3/DDPG可靠性、独立多工况鲁棒性、扰动下反馈优势，以及Rdelta/Np的实际收益。未启动15组500回合正式训练。
