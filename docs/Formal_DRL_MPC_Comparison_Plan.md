# 正式一维DRL–MPC比较计划与预检（2026-11-16）

## 当前状态

**只生成计划，不启动正式训练。** 当前强制主方案为configs/strong_qy_only.ini，不能误用strong_actions3.ini。已完成单工况五种子TD3和两个独立超参消融；详见TD3_Qy_Stability_and_Generalization.md。局部可重复的检查点收益不等于多工况泛化，位置调度也可复现当前反馈策略。

已生成runs/qy_formal_plan_20261116_v3/plan.json及preflight.json，ready=false、executed=false。阻塞项是：验证/测试真实集合为空，双MPC全集合回放证明不完整，尚无验证选出的超参配置锁。当前70 km/h双基线实际完赛、QP零失败；并非许可不可用。

## 冻结工况与公平预算

1. 按real_scenario_generation_plan.json在CarSim真实导出物理不同的train/validation/test，保持车辆和当前2输入20输出，先核查内容/物理签名再运行scenario_preflight；不伪造Simfile。
2. 训练集合可包含60/70/80 km/h和曲率A/B；验证65/75及曲率C；最终测试另行预注册未见速度/曲率D/稳定附着变化。具体值以真实核验可行为准，不能在看到策略测试结果后调整。
3. 三算法统一Np=36、Qpsi=8960、Rdelta=10800、唯一Qy动作、20维观测、同一联合奖励、边界/1.08变化率及决策采样。工况随机采样规则、总决策步数和评估周期一致。
4. 超参搜索先固定每算法4次试验、每次7900决策、seed42，预算为每算法31600决策。默认确认试验为每算法每seed40000决策，seed42–46共15组；本轮未执行。

## 预注册的四次超参试验

只使用验证集选择，Critic LR=3e-4，网络和其余设置保留已固定设计。四次是预算上限，不允许某算法额外尝试后只报告最好者。

| 试验 | DDPG Actor LR / sigma | TD3 Actor LR / sigma | TQC Actor LR / drop / target_entropy |
|---:|---|---|---|
| 1 | 1e-4 / 0.10 | 1e-4 / 0.10 | 1e-4 / 0 / auto |
| 2 | 1e-4 / 0.15 | 1e-4 / 0.15 | 3e-4 / 1 / auto |
| 3 | 3e-4 / 0.10 | 3e-4 / 0.10 | 1e-4 / 2 / −2.0 |
| 4 | 3e-4 / 0.15 | 3e-4 / 0.15 | 3e-4 / 3 / −2.0 |

TQC使用其原理对应的熵策略/截断设计，不强迫三算法相同探索机制；该四点设计不是TQC全因子网格。不预设TQC>TD3>DDPG。若将来扩大搜索，先修改预注册预算且三算法等额。

当前方法开发阶段已对TD3进行更多探索，这部分历史预算不能隐去。正式四次搜索的“公平”仅指预注册的新训练/验证集合及相同预算阶段；最终报告应同时披露历史开发偏向，使用事先锁定的独立测试评估，不能把全部历史探索声称为三算法等预算。

## 模型选择和报告

每验证工况均报告横向/车身航向RMSE、峰值、Ay、Beta、QP及完赛；同时报告原始MPC、强MPC和开发集冻结静态Pareto参照。

- 合法：全部完成、零QP、Ay≤0.85g、Beta≤5°；安全标准需在正式工况核验前预先固定，不针对算法变化。
- 严格通过：每个验证工况的两项RMSE都优于强MPC；按所有工况最小的min(Iy,Iheading)排序。
- 补充报告：平均Iy/Iheading、最差Iy/Iheading、双改善工况比例、合法比例、所有合法的逐工况双RMSE向量Pareto候选。部分通过不标记全部通过，但记录不隐藏。
- 每个seed同时保留final、best_legal、通过时的best_joint、全部周期模型、回放轨迹和来源哈希；最终汇总逐seed、均值、样本标准差和失败率。无合格模型明确未达标，不能拿final补成通过。

训练与选参后生成selection-lock.json：ddpg/td3/tqc各有train_config路径、该训练配置SHA-256、完全相同的validation_signatures列表。正式计划核对锁与工况；还需要scenario_preflight的baseline_proof.json，匹配源工况签名及基线配置哈希。控制配置及训练配置哈希写入计划。没有锁或证明时--execute也会被预检阻止。

最终测试前另锁定每算法每seed一个模型及配套配置/哈希，再用evaluate_dual.py读取test集合；测试集不参与选模、选Qy常量或构造Qy(X)。baseline安全核验不是根据算法表现挑测试集。

## 命令

```powershell
# 真实完整工况集合就绪后先检查双基线
python python/scenario_preflight.py --scenario-set sets.json --out runs/formal_baselines_NEW
# 默认仅生成正式计划；缺条件也输出阻塞原因
python python/formal_compare.py --config configs/strong_qy_only.ini --scenario-set sets.json --baseline-proof runs/formal_baselines_NEW/baseline_proof.json --total-decisions 40000 --out runs/formal_plan_NEW
# 验证选择完成后，重新生成带锁计划。此处不加--execute，不自动训练
python python/formal_compare.py --config configs/strong_qy_only.ini --scenario-set sets.json --baseline-proof runs/formal_baselines_NEW/baseline_proof.json --selection-lock selection-lock.json --total-decisions 40000 --out runs/formal_locked_plan_NEW
# 最终模型锁定后才运行测试
python python/evaluate_dual.py --lock locked_models.json --scenario-set sets.json --out runs/formal_test_NEW
```

工况生成清单中的null项不是可运行集合，不能直接当成sets.json。所有输出目录使用新名称。当前具备代码与单工况实测基础，尚不具备独立三算法比较启动条件。
