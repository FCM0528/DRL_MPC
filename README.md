# CarSim · LTV-MPC · DRL

研究主线为三自由度车辆预测模型与增量式LTV-MPC，使用TQC自适应调节MPC参数，并与TD3、DDPG统一比较。CarSim提供真实车辆闭环状态；DRL输出Np、Qy、Qpsi、Rdelta参数请求，C++ MPC求解转向。三维权重方案是首个统一预实验，受限四维保留完整参数自适应研究位置，不预设算法排名。

最新状态见[70km/h低附着预留测试基线验收](docs/CarSim_70kmh_mu06_Preflight.md)、[多工况准备与75km/h验证](docs/DRL_MPC_Multi_Scenario_Preparation.md)，前序[65km/h验证](docs/CarSim_65kmh_Validation.md)、[目标与训练审计](docs/DRL_MPC_Next_Stage_Analysis.md)、[主线恢复](docs/DRL_MPC_Mainline_Recovery.md)均保留。[固定Np实验](docs/TD3_Fixed_Np_Experiments.md)及一维TD3五种子结果作为动作消融保留；全部资料见[文档索引](docs/INDEX.md)。

65/75km/h、mu=0.8开发验证已完成；70km/h、mu=0.6已冻结为test，仅完成固定MPC预检，未作DRL最终测试。四份真实源的8组双基线全部通过，但train仍单一70km/h，且缺多工况超参数锁，正式训练尚未就绪。真实回合切换已验证，scenario-set训练必须使用--joint-selection，训练源由清单确定。formal_compare默认weights3，qy1/weights2/full4保留各维度消融；只生成预检计划，不自动执行15组训练。[一维稳定性与机制对照](docs/TD3_Qy_Stability_and_Generalization.md)及[历史一维正式计划](docs/Formal_DRL_MPC_Comparison_Plan.md)保留。

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

--scenario-set sets.json支持冻结的train/validation/test工况列表，以可复现序列采样训练工况。必须提供物理不同的真实工况；单工况重复回放仅为探索性诊断。

train.py --joint-selection按决策步在全部验证工况确定性评价，同时计算相对原始和强MPC的改善。合法要求完赛、零QP失败和既定Ay/Beta门槛；强基线达标要求每个验证工况的两项RMSE都改善。输出全部周期模型、best_legal_model.zip、final_model.zip、验证CSV、选择JSON和Pareto图。只有强基线双指标达标才生成best_joint_model.zip，final模型只诊断。

train_four.py保留旧接口并支持等决策预算与工况集合；不达标算法仍保留验证数据。最终测试由evaluate_dual.py读取事先锁定的模型路径、哈希与配套配置，报告双MPC基线，不用测试集挑模型。formal_compare.py默认只生成计划，独立验证完成前不执行正式三算法训练。

本地模型、runs和构建产物按.gitignore保留在本机；Git追踪源码、配置、测试及报告。历史模型必须使用对应冻结配置。CarSim接口与算法说明见文档索引。
