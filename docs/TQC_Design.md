# CarSim-MPC 的 TQC 设计与训练

参考：Kuznetsov 等，*Controlling Overestimation Bias with Truncated Mixture of Continuous Distributional Quantile Critics*，ICML 2020，https://proceedings.mlr.press/v119/kuznetsov20a.html 。论文的 MuJoCo 数值不是本项目的性能保证。

## 控制任务

当前主线读取 C++ 的20维 `road_preview_joint` 观测。首个统一预实验采用 `configs/strong_weights3.ini`：策略输出 `[Qy,Qpsi,Rdelta]` 三个归一化权重动作；`strong_full4.ini` 另加入Np。1维Qy和2维Qy+Qpsi保留为动作消融；历史3维Np+Qy+Qpsi保持原义。`policy_action_names` 明确排列，Python补齐未学习维度的零动作，C++桥接始终接收 `[Np,Qy,Qpsi,Rdelta]` 四维协议。DRL不直接输出转向指令。旧14维 `road_preview_v2`、旧模型及旧配置继续配套使用，不能与新策略混用。

零动作对应强MPC的Np=36、Qy=18400、Qpsi=8960、Rdelta=10800。权重采用有界连续对数映射，每决策变化比例上限1.08；完整4维profile将Np每决策变化限制为2。控制周期0.04s，决策包含2个控制周期。折扣系数仍从C++的 `discount_gamma` 读取。默认 `track_course=true` 时，Qpsi控制航迹角误差 `wrap(psi+beta-psi_ref)`，不等同于车身航向误差 `wrap(psi-psi_ref)`；联合奖励分别记录两者。

环境采用C++的 `joint_tracking` 奖励，横向、车身航向、航迹角、横向峰值/增长、转向、参数切换和安全项分别记录，桥接按有效控制步平均并扣除事件失败罚值。当前候选参数wy=1、w_heading=1、w_course=0.05，横向尺度0.2m、角度尺度0.035rad；本轮不改变它们。TQC的熵项只由算法内部处理，不伪装成环境回报。终点保持X=120m。实现与最新实测审计见 [下一阶段分析](DRL_MPC_Next_Stage_Analysis.md)。

## TQC 更新

采用 SB3-Contrib 的 TQC 实现。每个评论家输出 M=25 个分位数；下一状态的 N 个目标评论家输出合并排序，删去最高的 `N*d` 个分位数后构造 Bellman 目标。评论家通过分位数 Huber 损失学习。演员按全部 N×M 个分位数的平均值和自动调整的熵系数更新，避免对演员估计再截断一次。目标网络软更新系数为 0.005。`d=2` 是起点，论文显示最佳截断数因环境而异，若验证出现系统性低估或性能退化，可独立试验 `d=0/2/3`。

`practical` 配置为 N=3、每层 256 单元的两层网络，供当前 CPU 上的 CarSim 训练使用。`paper` 配置为论文实验的 N=5、每层 512 单元的三层网络。两种配置均保留 M=25；论文消融显示评论家数量增加到约 3 后收益趋缓，因此 CPU 配置是计算量与表达能力之间的工程选择，并非论文原实验复现。

训练参数由 `configs/rl_training.ini` 提供，已完成的统一预实验使用 `configs/rl_training_pilot.ini`：回放容量100000、预热550决策、批量256，Actor/Critic/熵优化器学习率为1e-4/3e-4/1e-4；每决策一次梯度更新。三算法各7900决策，每790决策确定性验证。TQC随机策略负责探索，不另加DDPG/TD3的动作噪声。自动熵目标随实际动作维数确定，当前3维为−3，4维为−4，不能照搬旧4维目标。已保存模型确认最终7350次梯度更新。

```powershell
# 复现单工况技术预实验；使用新输出目录，本轮不执行训练。
python python/train.py --algorithm tqc --config configs/strong_weights3.ini --train-config configs/rl_training_pilot.ini --simfile <真实冻结Simfile> --episodes 100 --total-decisions 7900 --seed 42 --eval-frequency 790 --tqc-profile practical --tqc-drop 2 --training-only --joint-selection --keep-training-traces --out runs/tqc_weights3_NEW
```

## 判断训练是否有效

训练回报不能替代双误差验证。`joint_validation.csv`、`checkpoint_selection.json` 保存全部周期检查点及Pareto候选；完赛、QP零失败、Ay≤0.85g、Beta≤5°才合法，强MPC双指标达标另要求两项RMSE均降低。`best_legal_model.zip` 可合法但未双指标达标；`best_joint_model.zip` 仅在达标时存在，`final_model.zip` 单独保留且不自动升级为最优模型。当前三算法结果均为70km/h、mu=0.8、seed42同工况诊断，不能用于算法正式排名。真实独立开发验证工况就绪前，不开展截断数0/2/3消融或正式多种子大训练。

2026-11-16审计确认：TQC最佳合法4740步的两项RMSE为0.078770m/0.918247°，最终为0.079435m/0.907356°；最终横向退化。没有证据要求TQC领先。自动熵、探索量级、critic分位数及动作弱效问题应结合真实跨工况结果研究，不能仅凭单次loss曲线判定算法结构失效。

## 2026-11-02 旧奖励版本的首轮试跑

以下数字使用此前的正值奖励版本，仅用于记录历史实验，不可与当前负值奖励直接比较。在 60 km/h 源工况上以 `practical`、`d=2`、种子 42 从零训练 100 回合（5500 次策略决策）。训练回报的前 20 回合平均值为 22.807，后 20 回合为 24.903；第 3000 步的确定性验证回报达到本轮最高 33.726，但第 5500 步降至 26.356。最后模型不宜只因训练结束就被选用。

对最佳检查点重跑时，同工况中性 MPC / TQC 的回报为 31.722 / 33.745，横向 RMSE 为 0.06176 / 0.05358 m，峰值横向误差为 0.35679 / 0.26216 m，航向 RMSE 为 1.00270° / 0.97601°；两者 QP 失败与附着保护步数均为 0。最终模型的回报低于中性 MPC，因此本轮应使用验证最佳检查点供后续独立工况验证。这是单工况、单种子的初步结果，尚不足以说明已收敛或泛化。
