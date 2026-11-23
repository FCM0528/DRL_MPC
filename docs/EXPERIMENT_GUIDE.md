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

三算法只有一个训练入口。修改 --algorithm 为 tqc、td3、ddpg，分别独立运行，不共享经验或 resume。1D/2D/3D/4D 分别使用 --action-profile qy1/weights2/weights3/full4，旧模型必须使用原配置和 action contract。

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


## 第二轮精简后的唯一入口（基准5f2ece1）

日常Python根模块15个，configs根文件10个。未保留完整legacy脚本或INI目录，也未将被删实现简单搬进archive。所有src、数据集版本、模型、原生工况和实验结果保持。

### 单因素参数变更

在上面的训练命令中，选择 --action-profile qy1、weights2、weights3、full4。奖励单因素仅追加 `--set rl_reward.w_y=2.0`；Actor单因素仅追加 `--train-set td3.actor_learning_rate=0.0001`；噪声单因素用 `--train-set td3.exploration_noise_sigma=0.10`。不要将这些变更合并成同一次消融。覆盖只影响新run；实际配置和所有CLI参数记录到manifest，不再在共享configs增加wy2/wy4等副本。

当前所有日常入口默认 strong_weights3；旧14维或旧动作前缀模型必须显式使用原冻结配置，不会静默套用新默认值。

### 老模型和研究流程恢复

```powershell
# 更早模型按其自己的manifest/hash恢复；比按文件名猜版本更可靠。
python python/dataset_manager.py restore-config --model '实际run/best_legal_model.zip' --out runs/recovered_model_NEW
# 恢复父提交指定INI及其全部依赖；逐字节核验，包括混合换行。
python python/dataset_manager.py restore-config --name strong_qy_only.ini --out runs/recovered_qy_NEW
# run_policy 的 --config 指向返回的恢复路径，--simfile指向该模型原冻结真实工况。
# 需要完整的旧实验CLI时，以父版本建立隔离工作树，绝不覆盖当前代码：
git worktree add --detach ../DRL_MPC_study_5f2ece1 5f2ece1
```

恢复需要完整Git历史（浅克隆先 fetch 原提交），或与该版本SHA匹配的完整 source_configs（--name加--snapshot）。更早版本优先 --model；模型缺manifest/配置/hash则报错，不猜测。旧Git工作树的构建、依赖及Solver位置需按其指南配置；不要将旧脚本复制回新日常目录继续扩展。旧工况列表JSON也可由该提交查看，但当前训练只使用已冻结dataset版本。

CMake自动将少量旧配置重建为out下的测试夹具，测试数值预期不变；这不是把退休配置留在源码目录。所有旧82项测试的断言保留，配置读取测试改为经过SHA验证的恢复路径；新增恢复/覆盖测试验证等价性和失败边界。

### 逐项退休与替代

|旧模块|决定与替代/保留的独特能力|
|---|---|
|train_four|删除重复训练编排；train统一学习，formal_compare统一计划|
|compare_four|删除旧单基线套件；evaluate_dual+report；comparison_limits移入evaluate，保留原测试|
|sweep|一次性已完成扫描退休；不自动重扫；完整扫描CLI在父版本|
|strong_report|独立旧run绘图入口退休；当前统一report，旧格式用隔离父版本|
|strong_response|旧限速试验退休；独特闭环方法保留Git版本与原数据|
|fixed_np_experiments|已完成静态Qy扫描/对照退休；当前profile+统一评价，原扫描由Git恢复|
|reward_ablation_report|专用奖励图退休；统一report与evaluate.reward_terms|
|qy_stability_report|五种子专用图退休；report支持多run；summarize原函数移至report供历史统计/原测试|
|qy_mechanism|已完成位置调度归因退休；inverse_qy移入action_mapping，独特试验CLI在父版本|
|mainline_audit|一次性训练审计CLI退休；quantile_summary并入algorithms，原测试保留|
|action_mask_diagnosis|批量8组介入CLI退休；MaskedPolicy移入run_policy供冻结策略介入，原测试保留|
|joint_feasibility|已完成45+26真实扫描退休，不重扫；dominates/pareto移入evaluate|
|control_sensitivity|一次性敏感性CLI退休；底层fixed_episode仍可用于独立覆盖回放，旧扫描方法从Git恢复|
|lateral_regression|专用报告退休；write_rows/reward_terms移入evaluate，旧X分段工具从Git恢复|
|joint_selection|旧单工况选择器退休，正式选择只用joint_validation；rank_candidate保留为评价纯函数与原回归测试|
|diagnostics|传感/单位数值函数原样并入evaluate，核心继续调用，并非取消诊断|

18份退休INI：carsim_curvature_schedule_40ms、carsim_python_rl、carsim_python_rl_joint及其actions2/3/4、continuous14、legacy14变体；mpc_heading_original/strong；strong_actions2/3及wy2/wy4；strong_qy_only/qy_qpsi；td3_qy_actor_lr1e4/sigma010。其逐文件SHA均在configs/legacy.json，不能修改冻结副本。1D/2D由通用profile生成；奖励/优化器小变化由覆盖代替；原始基线legacy/目标消融/Np前缀旧义由精确恢复读取。共同strong_actions4仍是当前真实依赖，保留。旧real_scenario_sets.local.json只是路径列表，已由dataset_v1覆盖，不是原生工况或控制参数。

### 本轮量化结果与验收

|指标|5f2ece1整理前|本轮整理后|
|---|---:|---:|
|configs根文件数|28|10|
|configs全目录文件数|28|10|
|python根模块数|31|15|
|python全目录模块数（含测试/init）|54|39|
|Python有效代码行（含测试）|5590|4065|

实际删除16个Python文件（没有迁入legacy目录），新加1个恢复测试文件，故全目录净少15个模块。实际删除18份INI和1份旧路径JSON，新加1份仅存hash/换行信息的legacy.json，配置总数净少18。有效代码按Python tokenize逐物理行统计，排除空行、注释和AST文档字符串；净减1525行（约27.3%），包含新增恢复/覆盖和原测试迁移代码，不包含C++、文档、报告或构建输出。没有把移动的10个纯函数/1个类当作删除；CSV writer一组3份实现归一，实际消除2个重复实现，其他只是归属调整，不虚报合并数。

独立历史兼容文件为1个 configs/legacy.json；保留legacy Python/INI源码文件为0。历史研究通过Git 5f2ece1或既有run冻结副本恢复；CMake生成的测试夹具位于out，不属于源码配置数。当前保留的15模块分别是六日常入口及action_mapping/algorithms/evaluate/joint_validation/paper_style/parity/scenario/scenario_preflight/scenario_sets等实际依赖，不为减少数量再剥除核心能力。

验收：Release成功、CTest 1/1；pytest107项通过（原82项完整保留，新增25项恢复/覆盖/契约测试）。原始/强MPC及三算法在65/75真实回放，横向/航向RMSE、峰值、Ay、Beta对父版证据差异<1e-9；3D/4D和生成式1D/2D全程157控制步zero-action parity通过。恢复旧1D/2D配置后真实回放仍为0.076194532m/0.900024047°与0.076485246m/0.912872472°；真实六回合切换与原seed42序列一致。统一报告离线重建成功，所有既有best/final指标逐项一致。dataset_v1、src和保留配置/轮胎数据SHA不变；本轮不训练、不访问预留test策略结果。

机器证据与重建HTML在 runs/deep_slim_20261118/：before.json、retired.json、results.json、compatibility_replays.json、closed_loop_final/、parity3/4、parity_qy1/weights2、report/index.html。统计只衡量源码/配置，不把实验报告输出文件增加或转移当作代码缩减。
## 弱起点四控制器演示

该演示从 `dataset_v1` 派生独立集合。70 km/h 用于学习与同工况诊断，65/75 km/h 用于开发验证。三种速度均为 μ=0.8、120 m 双移线。原低附着 test 保留其划分。

以下 PowerShell 命令使用本机 CarSim 与当前 Release 程序。输出目录应为新目录。设置 `$r` 后逐步执行并检查每步退出码。原始运行证据保存在 `runs/weak_mpc_demo_20261121`。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$r='runs/weak_mpc_demo_NEW'
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --weak-search --scenario-set datasets/carsim/manifests/dataset_v1.json --out $r
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --weak-preflight $r
$lock=Get-Content "$r/preflight/experiment_lock.json" -Raw | ConvertFrom-Json
foreach($algo in @('ddpg','td3','tqc')){
  uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm $algo --config $lock.policy_config --train-config $lock.training_config --scenario-set "$r/scenario_set.json" --total-decisions 7900 --episodes 100 --seed 42 --eval-frequency 790 --joint-selection --training-only --keep-training-traces --parity-blocks 200 --out "$r/training/$algo"
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $algo" }
}
$a=(Get-ChildItem "$r/training/ddpg" -Directory | Select-Object -First 1).FullName
$b=(Get-ChildItem "$r/training/td3" -Directory | Select-Object -First 1).FullName
$c=(Get-ChildItem "$r/training/tqc" -Directory | Select-Object -First 1).FullName
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --weak-demo $r --runs $a $b $c --out "$r/replay"
uv run --no-project --with-requirements python/requirements.txt python python/report.py --weak-demo $r --replay "$r/replay" --runs $a $b $c --out reports/weak_mpc_vs_3drl_NEW
```

实验主表统一使用开发验证选出的 best_legal，强基线仍作为 best_joint 门槛。报告以相同速度的弱 MPC 为改善率分母，final 指标和图位于补充区。已有回放时，最后一条命令即可离线重建 HTML、300 dpi PNG、矢量 PDF 和相对路径的 C++ 原始报告。报告核验实际训练预算、工况与配置哈希、检查点、源日志和回放指标。内部搜索与强基线核验位于生成目录的 `audit/`。

## 弱起点 300 完整回合实验

复用上面的冻结工况、弱中性控制配置和 pilot 优化器设置。从空经验池开始，seed=42，每 1000 个决策验证一次。回合停止由 `StopTrainingOnMaxEpisodes` 执行。Monitor 与 CarSim 完赛账本逐回合核验，并从日志读取实际决策数、经验池样本数和梯度更新次数。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$old='runs/weak_mpc_demo_20261121'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
$r="runs/weak_mpc_300episodes_$stamp"
$report="reports/weak_mpc_vs_3drl_300episodes_$stamp"
if ((Test-Path $r) -or (Test-Path $report)) { throw 'Choose unused output paths' }
$lock=Get-Content "$old/preflight/experiment_lock.json" -Raw | ConvertFrom-Json
foreach($algo in @('ddpg','td3','tqc')){
  uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm $algo --config $lock.policy_config --train-config $lock.training_config --scenario-set "$old/scenario_set.json" --episodes 300 --seed 42 --eval-frequency 1000 --joint-selection --training-only --keep-training-traces --parity-blocks 200 --out "$r/$algo"
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $algo" }
}
$a=(Get-ChildItem "$r/ddpg" -Directory | Select-Object -First 1).FullName
$b=(Get-ChildItem "$r/td3" -Directory | Select-Object -First 1).FullName
$c=(Get-ChildItem "$r/tqc" -Directory | Select-Object -First 1).FullName
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --weak-demo $old --episodes 300 --eval-frequency 1000 --runs $a $b $c --out "$r/replay"
if ($LASTEXITCODE -ne 0) { throw 'Replay or evidence verification failed' }
uv run --no-project --with-requirements python/requirements.txt python python/report.py --weak-demo $old --replay "$r/replay" --runs $a $b $c --out $report
```

300 回合训练命令只传 `--episodes`。回放阶段再次核验原运行模型、完整检查点、完赛记录、工况和配置哈希。HTML 使用原有四控制器样式，另附 `training_comparison.csv` 和 `audit/learning_diagnostics.json`。可用报告命令从 100、300 回合原始日志及存好的回放离线重建。100 回合每 790 个决策验证，300 回合每 1000 个决策验证，比较选中模型时应保留这项取样差异。

动作诊断按决策取样。近中性定义为 |a|<0.1，请求饱和定义为 |a|≥0.98；实执行用相对于同侧边界的对数距离归一。变化率限制按目标相对上一实际权重是否超出每步 1.08 倍计算。目标 Q 诊断使用冻模与回放的真实 70 km/h 观测，每个状态用固定 seed 抽样 16 次，按原 TQC 公式计算软 Bellman 目标，包含共同输入对照。它反映离线探针的目标值分布；实际优化过程由原训练 loss 和熵日志呈现。

## 弱起点四维 500 回合实验

`--full4-preflight` 读取原弱起点实验锁，以三维弱起点配置生成运行目录内的 `run_delta.ini`。增量使用与 `--action-profile full4` 相同的函数，只覆盖动作维数、名称和 Np 限速。通过 70、65、75 km/h 的全程真实零动作核验后，保存独立锁和配置、工况、预检轨迹的 SHA256。训练读取锁内的完整四维配置，每 1000 个真实决策在两种开发速度确定性验证。`--episodes 500` 由 SB3 回合停止回调执行。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$old='runs/weak_mpc_demo_20261121'
$prior='runs/weak_mpc_300episodes_20261121/replay'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
$r="runs/weak_mpc_full4_500episodes_$stamp"
$report="reports/weak_mpc_full4_500episodes_$stamp"
if ((Test-Path $r) -or (Test-Path $report)) { throw 'Choose unused output paths' }
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --full4-preflight $old --out "$r/preflight"
if ($LASTEXITCODE -ne 0) { throw 'Four-action preflight failed' }
$lockFile="$r/preflight/experiment_lock.json"
$lock=Get-Content $lockFile -Raw | ConvertFrom-Json
foreach($algo in @('ddpg','td3','tqc')){
  uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm $algo --config $lock.policy_config --train-config $lock.training_config --scenario-set "$old/scenario_set.json" --episodes 500 --seed 42 --eval-frequency 1000 --joint-selection --training-only --keep-training-traces --parity-blocks 200 --out "$r/$algo"
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $algo" }
}
$models=@('ddpg','td3','tqc') | ForEach-Object { (Get-ChildItem "$r/$_" -Directory | Select-Object -First 1).FullName }
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --weak-demo $old --experiment-lock $lockFile --episodes 500 --eval-frequency 1000 --runs $models --out "$r/replay"
if ($LASTEXITCODE -ne 0) { throw 'CarSim replay or evidence verification failed' }
uv run --no-project --with-requirements python/requirements.txt python python/report.py --weak-demo $old --replay "$r/replay" --history-300 $prior --runs $models --out $report
```

离线重建只运行最后一条命令，指向现有模型和回放目录以及新的报告目录。回放先对账 Monitor、CarSim 完赛及重置记录、训练 manifest 和模型内的 SB3 计数。回合停止回调可能在最后一条终止转移写入经验池前退出，保留真实交互和经验池各自的计数。

`training_episode_axis.csv` 逐行保留原始决策、已完成轮次、当前轮内决策数、源日志行、指标值、检查点及工况。映射按 Monitor 累计回合长度取不大于当前决策的最后一个完整边界。损失与熵图在同一轮次有多条日志时取均值，CSV 保留原值与样本数。权重和整数 Np 图使用实际距离 X，分开绘制预限速的目标 Np 与执行 Np。历史对照同时改变了轮次数和动作维度，须用同为 500 回合的三维组才可隔离动作维度效应。

## 固定中性 MPC 与五控制器报告

已有四维 500 回合模型及回放可直接用于五控制器报告。新搜索只在 70 km/h 执行 18 组固定 MPC 候选，在落入目标区间后依次比较航向 RMSE、转向速率 RMS 和稳定裕量。锁定参数后验证 70、65、75 km/h。报告重用 21 份旧回放和 C++ 报告，并纳入三个新增的固定中性 MPC 回放。

```powershell
$r='runs/weak_mpc_full4_500episodes_20261121_225029'
$weak='runs/weak_mpc_demo_20261121'
$prior='runs/weak_mpc_300episodes_20261121/replay'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
$neutral="runs/weak_neutral_3drl_500episodes_$stamp"
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --neutral-search $weak --replay "$r/replay" --out $neutral
if ($LASTEXITCODE -ne 0) { throw 'Neutral tuning or replay failed' }
$models=@('ddpg','td3','tqc') | ForEach-Object { (Get-ChildItem "$r/$_" -Directory | Select-Object -First 1).FullName }
uv run --no-project --with-requirements python/requirements.txt python python/report.py --weak-demo $weak --replay "$r/replay" --neutral $neutral --history-300 $prior --runs $models --out "reports/weak_neutral_3drl_500episodes_$stamp"
```

离线重建时，将 `$neutral` 指向已有的中性调参及回放目录，再运行最后一条命令并使用新的输出路径。正式目录为 `runs/weak_neutral_3drl_500episodes_20261123/`。训练与开发源重新核验原生 SHA，预留测试源仅校验冻结清单元数据。报告从各自的训练 Monitor、逐步桥接数据和决策日志重建环境回报。两波段外端点取 LTARG 两个有符号斜率波瓣的 10% 交点，分界取两斜率极值之间的最大反向几何曲率点。分界样本只计入第二段。

## 双区段均衡诊断与前瞻训练

先归档旧检查点的真实开发验证轨迹，并逐项计算相对弱与中性两者最优值的改善率。旧规则的 `best_legal` 和 `best_joint` 归档保持原样。只有各算法均存在严格达标旧检查点时，流程在回溯诊断处结束。新奖励训练从空经验池开始，经旧奖励数值回归、离线损失验收和三速度全程零动作检查后使用独立选模器。训练清单只含 70 km/h 训练和 65/75 km/h 验证。

```powershell
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:PYTHONDONTWRITEBYTECODE='1'
$old='runs/weak_mpc_demo_20261121'
$originalReplay='runs/weak_mpc_full4_500episodes_20261121_225029/replay'
$neutral='runs/weak_neutral_3drl_500episodes_20261123'
$stamp=Get-Date -Format yyyyMMdd_HHmmss
$r="runs/dual_segment_balanced_4drl_$stamp"
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --wave-prepare $old --replay $originalReplay --neutral-run $neutral --out $r
if ($LASTEXITCODE -ne 0) { throw 'Checkpoint or reward audit failed' }
$lock="$r/wave_lock.json"
if (!(Test-Path $lock)) { return }
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --wave-preflight $lock
if ($LASTEXITCODE -ne 0) { throw 'Wave parity failed' }
foreach($algo in @('ddpg','td3','tqc')){
  uv run --no-project --with-requirements python/requirements.txt python python/train.py --algorithm $algo --dual-segment-lock $lock --training-only --keep-training-traces --parity-blocks 200 --out "$r/$algo"
  if ($LASTEXITCODE -ne 0) { throw "Training failed for $algo" }
}
$models=@('ddpg','td3','tqc') | ForEach-Object { (Get-ChildItem "$r/$_" -Directory | Select-Object -First 1).FullName }
uv run --no-project --with-requirements python/requirements.txt python python/evaluate_dual.py --wave-replay $lock --runs $models --out "$r/replay"
if ($LASTEXITCODE -ne 0) { throw 'Wave replay failed' }
uv run --no-project --with-requirements python/requirements.txt python python/report.py --weak-demo $old --neutral $neutral --replay "$r/replay" --runs $models --out "reports/dual_segment_balanced_4drl_$stamp"
```

`dual_segment_candidates.csv` 保留最差改善率和未达标项。前瞻训练仍每 1,000 个真实决策验证一次，合法性及两个工况的全程 RMSE 门槛先于四项分区改善。通过时最大化两个工况合计八项中的最小改善率，差值相同再比较全程横向、航向 RMSE。若均未达标，则按相同顺序保留合法候选并标记失守项。`final_model.zip` 保持诊断角色。离线重建可直接使用保存的 replay，最后一条命令无需启动求解器。
