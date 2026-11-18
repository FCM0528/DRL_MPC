# CarSim · LTV-MPC · DRL

以 TQC 为重点，研究 DDPG/TD3/TQC 对三自由度车辆预测模型和增量式 LTV-MPC 参数的自适应调节。CarSim 提供真实闭环反馈；DRL 调整 MPC 参数，不直接输出转向。不预设算法排名。

`src/` 控制器，`python/` 15个日常/必要内部模块，`configs/` 10份日常文件与哈希恢复目录，`datasets/carsim/` 真实工况和不可变版本，`runs/` 历史模型与日志，`reports/` 离线可重建结果。MATLAB 是独立工程。

## 快速开始

```powershell
cmake --build out/python-rl-build --config Release
ctest --test-dir out/python-rl-build -C Release --output-on-failure
uv run --no-project --with-requirements python/requirements.txt --with pytest python -m pytest python/tests -q -p no:cacheprovider
python python/dataset_manager.py list
python python/report.py --help
```

需真实 CarSim 2024.1、有效许可及本地原生工况快照；原始数据和大模型不随Git克隆。dataset_v1：train=70/60/80、validation=65/75 km/h（μ=.8），test=70 km/h、μ=.6；后者尚未作DRL最终评价。已有seed42技术预实验不代表正式算法排名或道路轨迹泛化。

## 日常入口

|任务|入口|
|---|---|
|原始/强固定MPC|configs/default.ini、mpc_strong.ini|
|工况注册、校验、版本|python/dataset_manager.py|
|三算法、动作消融训练|python/train.py --algorithm tqc/td3/ddpg|
|锁定模型验证/最终测试|python/evaluate_dual.py --split validation/test|
|单模型回放|python/run_policy.py|
|正式比较计划|python/formal_compare.py（默认不执行）|
|日志离线总览|python/report.py --runs ... --out reports/NEW|

完整步骤见 [实验指南](docs/EXPERIMENT_GUIDE.md)，配置见 [配置表](configs/README.md)。原理：[项目架构](docs/PROJECT_OVERVIEW.md)、[MPC与基线](docs/MPC_MODEL_AND_BASELINES.md)、[奖励与算法](docs/DRL_DESIGN_AND_REWARD.md)。数据：[工况指南](docs/DATASET_GUIDE.md)、[研究结果](docs/RESEARCH_RESULTS.md)。[文档索引](docs/INDEX.md)、[清理审计](docs/PROJECT_REFACTOR_AUDIT.md)、[历史恢复](docs/HISTORY_INDEX.md)。

## 配置不再随实验复制

三算法统一 `train.py`；动作使用 `--action-profile qy1/weights2/weights3/full4`。奖励和学习率变化分别使用 `--set`、`--train-set`，只在独立run内冻结覆盖与manifest，不增加共享INI或训练脚本。旧模型优先使用原run冻结配置，或 `dataset_manager.py restore-config --model ... --out NEW` 核验后恢复。

本轮退休16个旧模块、18份INI和1份旧集合JSON；独特研究流程可从父提交5f2ece1隔离恢复。量化结果、逐项退休原因、回归证据和恢复命令见[实验指南](docs/EXPERIMENT_GUIDE.md)。不计移动、报告产物或Git历史为代码缩减。
