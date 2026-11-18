# 构建、训练、验证与报告

在仓库根目录 PowerShell 运行。需要本机 CarSim 2024.1、真实2输入20输出原生工况和有效许可。uv 方式可在 python 命令前加 `uv run --no-project --with-requirements python/requirements.txt`。输出 NEW 必须替换为未存在的目录；报告离线重建不需要启动训练。

```powershell
cmake -S . -B out/python-rl-build -G 'Visual Studio 18 2026' -A x64
cmake --build out/python-rl-build --config Release
ctest --test-dir out/python-rl-build -C Release --output-on-failure
uv run --no-project --with-requirements python/requirements.txt --with pytest python -m pytest python/tests -q -p no:cacheprovider
python python/dataset_manager.py validate
```

原始/强 MPC 的控制配置分别为 configs/default.ini、configs/mpc_strong.ini。实际已有来源可从 dataset_manager list 和 dataset_v1 查询；不要依赖可变的 CarSim Browser simfile。固定回放命令：

```powershell
out/python-rl-build/Release/MPC_DoubleLaneChange.exe --config configs/default.ini --no-open --out runs/original_NEW --set 'carsim.simfile=实际冻结Simfile绝对路径' --set carsim.sync_browser_result=false
out/python-rl-build/Release/MPC_DoubleLaneChange.exe --config configs/mpc_strong.ini --no-open --out runs/strong_NEW --set 'carsim.simfile=实际冻结Simfile绝对路径' --set carsim.sync_browser_result=false
python python/parity.py --config configs/strong_weights3.ini --simfile '实际冻结Simfile绝对路径' --blocks 200 --out runs/parity_NEW
```

三算法只有一个训练入口。修改 --algorithm 为 tqc、td3、ddpg，分别独立运行，不共享经验或 resume。1D/2D/4D 对应 configs/README 所列配置，旧模型必须使用原配置和 action contract。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
python python/train.py --algorithm tqc --config configs/strong_weights3.ini --train-config configs/rl_training_pilot.ini --scenario-set datasets/carsim/manifests/dataset_v1.json --total-decisions 7900 --seed 42 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --out runs/tqc_NEW
```

7900 是技术试验预算，不是正式训练要求。formal_compare.py 默认 weights3，其他 profile 为 qy1/weights2/full4；默认只计划，只有真实基线证明和独立验证选参锁齐备才允许 --execute。严禁将15组正式训练隐式作为构建或报告动作。

验证/最终测试统一 evaluate_dual，必须显式选择 split 和预先锁定的 best 模型 registry。只有最终锁定策略可访问预留 test；不得由测试挑选检查点。旧70km/h最佳模型锁保留 docs/archive/next_stage_20261116_locks.json，新数据集预实验锁位于 datasets/carsim/reports/dataset_v1_seed42_pilot/pilot_model_locks.json。

```powershell
python python/evaluate_dual.py --lock datasets/carsim/reports/dataset_v1_seed42_pilot/pilot_model_locks.json --scenario-set datasets/carsim/manifests/dataset_v1.json --split validation --out runs/validation_NEW
python python/run_policy.py --help
python python/formal_compare.py --help
```

模型、原生文件、完整轨迹仍在原 runs；Git 只保存代码、元数据及必要证据。备份必须包含这些本地数据。final_model 是最终诊断模型，best_legal 可能尚未双改善，best_joint 才表示全部验证工况通过；不要互换。

## 统一离线报告

```powershell
$r='runs/dataset_system_20261116_222225'
$tqc=(Get-ChildItem "$r/tqc" -Directory | Select-Object -First 1).FullName
$td3=(Get-ChildItem "$r/td3" -Directory | Select-Object -First 1).FullName
$ddpg=(Get-ChildItem "$r/ddpg" -Directory | Select-Object -First 1).FullName
python python/report.py --runs $tqc $td3 $ddpg --dataset-manifest datasets/carsim/manifests/dataset_v1.json --out reports/dataset_v1_NEW
```

输出 index.html、metrics.csv、summary.json、来源 SHA256 和 figures/ PNG/PDF。默认包括全部65/75验证工况、best/final区别、Reward/Actor/Critic/TQC熵、误差/轨迹/车辆状态、实际学习参数和数据集统计。不会训练或读取预留test策略结果。多seed/动作消融可以传多个同 dataset_hash 的已完成 run；分别显示，不能用单seed伪造置信带。旧 evaluation_dir 位置参数和 --locked-validation 保留用于历史单工况/固定 MPC 报告；旧图选 case 的能力不替代新全工况总览。

新增工况完整流程见 DATASET_GUIDE.md。原生Simfile/Parsfile真实性、完整性和当前任务适用性分别检查；尚未支持的变速/分段附着允许存档但禁止自动训练。
