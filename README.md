# CarSim · LTV-MPC · DRL

研究主线为三自由度车辆预测模型与增量式LTV-MPC，使用TQC自适应调节MPC参数，并与TD3、DDPG统一比较。CarSim提供真实车辆闭环状态；DRL输出Np、Qy、Qpsi、Rdelta参数请求，C++ MPC求解转向。三维权重方案是首个统一预实验，受限四维保留完整参数自适应研究位置，不预设算法排名。

最新入口为[可扩展真实工况库](docs/CarSim_Extensible_Dataset_System.md)与[三算法多工况在线技术预实验](docs/DRL_MPC_MultiScenario_Training.md)。此前[80km/h验收](docs/CarSim_80kmh_Training_Case.md)、[60km/h训练源](docs/CarSim_60kmh_Training_Case.md)、[低附着预留测试](docs/CarSim_70kmh_mu06_Preflight.md)、[75km/h验证](docs/DRL_MPC_Multi_Scenario_Preparation.md)、[65km/h验证](docs/CarSim_65kmh_Validation.md)、[目标审计](docs/DRL_MPC_Next_Stage_Analysis.md)、[主线恢复](docs/DRL_MPC_Mainline_Recovery.md)和全部动作消融均保留；详见[文档索引](docs/INDEX.md)。

六份真实源已注册为不可变 dataset_v1：train=70/60/80km/h、mu=0.8；validation=65/75；test=70km/h、mu=0.6，未作DRL策略评价。三算法各完成 seed42、7900 决策的多工况真实在线预实验；TQC、TD3 有全部验证工况双改善周期模型，DDPG 未达标但保留合法模型与 Pareto 结果，不据单种子确立正式排名。正式超参数锁尚未冻结，不启动15组正式训练。现有三列表 JSON 和旧模型仍兼容，新训练推荐版本 manifest；scenario-set 必须使用 --joint-selection。formal_compare 默认 weights3，qy1/weights2/full4 继续用于消融。

## 构建和检查

在项目根目录PowerShell运行，需要CarSim许可、64位Solver DLL和已导出的完整工况。运行使用私有工况副本。

```powershell
cmake -S . -B out/python-rl-build -G 'Visual Studio 18 2026' -A x64
cmake --build out/python-rl-build --config Release
ctest --test-dir out/python-rl-build -C Release --output-on-failure
uv run --no-project --with-requirements python/requirements.txt --with pytest python -m pytest python/tests -q
```

## 配置与动作语义

| 配置 | 用途 |
|---|---|
| configs/default.ini | 原始MPC：40/13/11500/2800/10800 |
| configs/mpc_strong.ini | 强MPC：36/13/18400/8960/10800 |
| configs/strong_qy_only.ini | 一维动作消融，仅Qy；Np固定36 |
| configs/strong_qy_qpsi.ini | 二维动作消融，Qy+Qpsi；Np固定36 |
| configs/strong_weights3.ini | 主线三维权重：Qy+Qpsi+Rdelta；Np固定36 |
| configs/strong_full4.ini | 完整四维：Np+Qy+Qpsi+Rdelta；Np每决策最多变化2步 |
| configs/strong_actions2/3/4.ini | 旧Np前缀动作消融，保持原语义 |
| configs/rl_training_pilot.ini | 网络、学习率、预热、噪声等设置 |

参数顺序为Np/Nc/Qy/Qpsi/Rdelta。policy_action_names指定策略输出顺序；桥接始终接收[np,q_y,q_psi,r_delta]，未学习维度填0。旧配置未声明名称时保留2/3/4维前缀语义。manifest保存名称、维数及映射版本，同维数不同语义的模型不得混用。

## 开发工况诊断与训练

输出根目录应是新的唯一目录。以下命令明确使用7900决策，不启动500回合套件。使用python或uv运行均可。

```powershell
python python/parity.py --config configs/strong_weights3.ini --blocks 200 --out runs/parity_NEW
python python/train.py --algorithm tqc --config configs/strong_weights3.ini --fixed-config configs/default.ini --train-config configs/rl_training_pilot.ini --simfile <冻结开发工况.sim> --seed 42 --episodes 100 --total-decisions 7900 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --out runs/weights3_tqc_NEW
```

相同三维配置与预算分别使用ddpg、td3、tqc完成统一预实验。1D/2D按对应历史配置复现；full4先做632决策技术试验。经验从空缓存开始，车辆状态和奖励来自真实CarSim；静态参数只在开发工况选择。

## 验证与模型选择

--scenario-set 支持旧 sets.json 及新的 datasets/carsim/manifests/dataset_v1.json，以可复现序列采样全部合法训练工况。新增真实工况用 dataset_manager.py 注册、验收并发布新版本，不需要修改训练框架。训练记录版本/hash、逐工况实际样本量；验证与测试不会进入经验池。不同工况采用相同参考轨迹时不能声称道路轨迹泛化。

train.py --joint-selection按决策步在全部验证工况确定性评价，同时计算相对原始和强MPC的改善。合法要求完赛、零QP失败和既定Ay/Beta门槛；强基线达标要求每个验证工况的两项RMSE都改善。输出全部周期模型、best_legal_model.zip、final_model.zip、验证CSV、选择JSON和Pareto图。只有强基线双指标达标才生成best_joint_model.zip，final模型只诊断。

train_four.py保留旧接口并支持等决策预算与工况集合；不达标算法仍保留验证数据。最终测试由evaluate_dual.py读取事先锁定的模型路径、哈希与配套配置，报告双MPC基线，不用测试集挑模型。formal_compare.py默认只生成计划，独立验证完成前不执行正式三算法训练。

本地模型、runs和构建产物按.gitignore保留在本机；Git追踪源码、配置、测试及报告。历史模型必须使用对应冻结配置。CarSim接口与算法说明见文档索引。
