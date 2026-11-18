# 日常配置（根目录10文件）

|文件|用途与继承|
|---|---|
|default.ini|原始 MPC：40/13/11500/2800/10800|
|mpc_strong.ini|继承 default：36/13/18400/8960/10800|
|strong_actions4.ini|继承强基线；共同20维观测、连续权重、奖励、边界和1.08限速|
|strong_weights3.ini|继承共同基础；主线 Qy/Qpsi/Rdelta|
|strong_full4.ini|继承共同基础；Np/Qy/Qpsi/Rdelta，Np每决策限速2|
|rl_training.ini|通用算法学习参数；不改变车辆/MPC|
|rl_training_pilot.ini|当前技术预实验网络/学习率/预热；三算法共用|
|magic_tire.csv|保留轮胎数据表|
|legacy.json|退休配置 SHA256、父提交、换行布局；不存配置参数文本|
|README.md|本说明|

全部保留的INI/CSV与本轮开始时逐字节一致。Np/Nc/Qy/Qpsi/Rdelta 的两套固定基线不变。1D/2D不再复制专用INI：

```powershell
python python/train.py --algorithm td3 --config configs/strong_weights3.ini --action-profile qy1 ...
python python/train.py --algorithm td3 --config configs/strong_weights3.ini --action-profile weights2 ...
```

profile=weights3/full4 分别为主线三维/完整四维。profile变更只在新run内写一份引用原基础的 delta；有效参数和源SHA存manifest。未学习维度严格中性。旧 Np+Qy+Qpsi 是另一种历史语义，不能自动改成三维权重策略。

奖励单因素使用 `--set rl_reward.w_y=2.0`；优化器单因素使用 `--train-set td3.actor_learning_rate=0.0001`，分别开展实验，勿一次改多个因素。训练覆盖使用原有类型和数值校验；控制器覆盖仅允许已存在的 rl/rl_reward 字段，不允许借此改车辆或MPC物理参数。既有配置文件不写回，实际训练参数冻结在run内。

## 历史配置恢复

18份非日常INI已从源码树删除，另删除旧 real_scenario_sets.local.json（真实集合已由 dataset_v1 接管）。原生工况、模型和run内冻结配置没有删除。

首选模型自己的冻结版本，同名INI在不同年代可能不同：

```powershell
python python/dataset_manager.py restore-config --model '实际run/best_legal_model.zip' --out runs/recovered_model_NEW
# 或恢复本轮父提交的指定配置及完整继承链：
python python/dataset_manager.py restore-config --name strong_qy_only.ini --out runs/recovered_qy_NEW
```

恢复默认从 Git 5f2ece1 重建，并验证每个字节的 SHA256，包括旧文件混合 LF/CRLF；不得覆盖已有目录。`--name ... --snapshot 实际run/source_configs` 可使用匹配目录代替Git。更早模型优先 --model，以它的 manifest 哈希为准，不拿同名新配置冒充旧版。缺失manifest/来源/hash时明确报错，不猜配置。

旧JSON或完整旧研究CLI用隔离Git工作树恢复，方法见实验指南。自动恢复需完整Git历史或冻结副本；浅克隆先取回父提交。CMake仅在构建目录生成旧配置测试夹具，不把退休文件重新放回日常目录。
