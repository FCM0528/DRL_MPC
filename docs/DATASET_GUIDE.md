# 可扩展 CarSim 工况库

本系统存储真实 CarSim 原生工况，而非监督学习样本。DDPG、TD3、TQC 的 transition 仍由各自的在线 CarSim 闭环产生，分别进入独立 Replay Buffer。库中不存在合成车辆状态或固定 MPC 标签。

## 目录与当前版本

```text
datasets/carsim/
  registry.json                 # 当前工况索引，可增加记录
  scenarios/cs_<signature>/
    simfile.sim                 # 原生文件逐字节保存，本地保留
    Run_all.par                 # 完整原生文件逐字节保存，本地保留
    metadata.json               # 来源、物理信息、检查状态与验收证据
  manifests/dataset_v1.json      # 不可覆盖的训练数据集版本
  reports/                      # 统计表、实际样本数量和论文图
```

原生许可数据文件由 `.gitignore` 排除，保留本地及原历史 runs；Git 保存注册信息、manifest、统计表与报告。克隆代码不等于获得原生工况，需要同时恢复经过 SHA256 核验的本地数据文件及 CarSim 安装/许可。历史模型和工况未移动。

`python python/dataset_manager.py restore` 可从记录的原历史冻结路径恢复缺失的原生文件；异机备份用 `restore --from-root 'E:\实际备份\datasets\carsim'`。只恢复缺失文件，逐个核验 SHA256，拒绝覆盖已有不同内容，失败清除本次恢复的文件；不删除历史源。应备份整个 datasets/carsim 和报告引用的原始验收 runs，单独 Git 元数据不能代替数据备份。

当前 `dataset_v1` 的内容哈希为 `920bec95afaee29ab441a8ce60cc1f5b11c04bcdb04325f7bf8287debaa5dc9f`。train 顺序保留 70/60/80，validation 保留 65/75，test 保留 70 km/h、μ=0.6；未将开发验证源调入训练集。六份原始 Simfile 与 Parsfile 均按旧冻结快照复制，SHA256 相同。验收复用此前真实的 12 组固定 MPC 预检，不冒称本轮重新执行了这 12 组。

| ID | split | km/h | μ |
|---|---|---:|---:|
| cs_4b2833a3a0ee9d260693 | train | 70 | 0.8 |
| cs_c3795eb276ef9ab53a0f | train | 60 | 0.8 |
| cs_3666b760495501987d4d | train | 80 | 0.8 |
| cs_34db60360e2714ac115d | validation | 65 | 0.8 |
| cs_9f33c9f2d89737f3ae79 | validation | 75 | 0.8 |
| cs_37580cac3f141fec7281 | test | 70 | 0.6 |

六工况车辆、轮胎和 I/O 内容一致，参考轨迹均为原 0–120 m LTARG。只能研究速度/附着变化，不能宣称道路轨迹泛化。

## 注册及验收

以下命令在项目根目录运行；用已安装依赖的 Python，或在命令前加 `uv run --no-project --with-requirements python/requirements.txt`。路径参数必须指向实际文件；`NEW` 表示自行选择尚不存在的输出目录。

```powershell
python python/dataset_manager.py list
python python/dataset_manager.py validate
python python/dataset_manager.py add --simfile 'D:\实际原生导出\simfile.sim' --split train --type double_lane_change
# 上一命令返回 scenario_id；新工况未经求解器验收，默认不能训练。
python python/dataset_manager.py preflight --scenario-id cs_实际ID --out runs/case_preflight_NEW
python python/dataset_manager.py accept --scenario-id cs_实际ID --baseline-proof runs/case_preflight_NEW/preflight/baseline_proof.json
python python/dataset_manager.py export --version dataset_v2
python python/dataset_manager.py summary --manifest datasets/carsim/manifests/dataset_v2.json --out datasets/carsim/reports/dataset_v2_before_training
```

`preflight` 复用 `scenario_preflight.py`，只运行原始和强固定 MPC。它先从库中校验的原生文件生成私有运行副本，不修改 CarSim Browser 数据。安全条件保持完赛、QP=0、峰值 Ay≤0.85g、|Beta|≤5°；单工况预检的整体 ready 可能因缺少其余 split 为 false，`accept` 核验该物理工况两条真实基线记录。失败证据保留，不通过则不得启用。

已有冻结集合的导入命令为：

```powershell
python python/dataset_manager.py import --scenario-set configs/real_scenario_sets.local.json --baseline-proof runs/training_case_80_mu08_20261116_213738/preflight/baseline_proof.json --type double_lane_change
```

相同物理签名再次注册会报告 `already_registered`，更名不能增加采样次数。跨 split 重复被拒绝，尤其禁止通过重新注册将 test 转为 train。本工具不提供隐式改划分命令；如研究设计必须改变，需要单独、明确授权的新研究方案，并承认原测试已被使用。

注册先准备临时目录、核验源字节，再安装工况并原子更新 registry；失败回滚本次新目录。已存在目录和版本不覆盖。不要同时运行多个修改 registry 的命令。

## 元数据与适用性

metadata 包含用户要求的 ID、名称、类型、源路径、速度/附着及其 profile、LTARG 原始数值、起止 station、车辆 ID/hash、I/O hash、参考 hash、物理签名、两个原始文件 hash、split、enabled、validation_status、时间、notes、extra_metadata。

速度及 μ 由原生数值项提取，非单一恒定情况写 null/custom，不能用数据集标题代替真实物理值。`scenario_type` 是用户声明的研究分类，`type_verification` 明确保留其来源；当前六源的双移线由此前真实来源与参考表核验，不声称程序可仅凭名称识别任意正弦、稳态圆周或加减速工况。训练适用性依据实际参考表、物理命令、车辆/I/O 和求解器验收，不依据名称。

分层状态包括：可存储、文件完整、车辆/I/O 兼容、有效横向参考、当前 task 兼容、真实求解器通过。无参考轨迹、不同 I/O 或不同车辆可以存档，但标记 blocked、不导出训练。与既有参考表不同的工况不因名称被拒绝；需核验有限、递增 station 的 SPLINE_FLAT LTARG 和两套真实基线。

目前支持在库中保存所有列出的 scenario_type；当前 `lateral_ltarg_v1` **尚不启用变速或分段附着工况**，这些将以 custom profile/blocked 保存，须补充专门解析和验收后用独立 task_profile 开放。没有有效 LTARG 的转向激励、纯纵向任务也不自动用于横向 RL。不能将可存储等同于已支持训练。

没有写死训练源数量、速度列表或 120 m。当前 C++ 依据参考终点停止，原语义未更改；150 m 等新长度需有实际有效参考、终点覆盖和真实双基线验收后才可启用。单元测试以明确标记的解析器 fixture 验证了 150 m 和 50 个训练源；这些不是已生成或已仿真的真实工况。

## 路径、版本与泄漏防护

registry 和 manifest 使用相对库根路径；CLI `--root` 可配置库位置，读取 manifest 时可用环境变量 `CARSIM_DATASET_ROOT` 指向完整库副本。原生 Simfile 中的 INPUT 保留原字节，包括其历史绝对路径；**不要绕过 manifest 直接运行库内存档 Simfile**。适配器以 metadata 中指定、哈希核验的本地 Parsfile 为准，经现有 snapshot_scenario 重定向私有运行 INPUT。目录迁移单元测试已覆盖原 INPUT 路径失效的情况。PROGDIR、DATADIR 等本机 CarSim 外部依赖仍需真实存在，并非跨主机独立求解器包。

每次 export 产生新版本，旧版本拒绝覆盖；dataset_hash 覆盖划分、有序 ID、完整元数据及来源哈希，但不覆盖可重定位的 dataset_root。manifest 内嵌工况记录，读取时不查询可变 registry。因此新增工况不会改变正在运行或历史版本的训练集合。

train.py 继续接受旧的三列表 JSON，也接受版本 manifest。新版本训练记录 dataset_version/hash、各 split IDs/signatures、随机种子、采样规则、控制器配置链 SHA256、训练配置 SHA256。每次运行再次复制私有工况、控制配置和原始 manifest，历史旧接口及模型不变。formal_compare.py 的版本化输入也保留版本/hash，不再降级成丢失版本的路径列表。

训练只把 train 传给环境，validation 只被确定性验证器读取，test 不进入采样池、Replay Buffer、周期验证或选模。test 的原生文件完整性校验不等于对其运行策略评价。本轮不评价预留 test。

## 采样与真实样本统计

第一版仅支持 `uniform_episode_v1`，由 Gym 环境局部 RNG 在 episode 边界选工况，关闭旧 Bridge/Solver，再启动新进程。保留原 `integers(N)` 调用与顺序。weighted_episode、balanced_decisions 暂未实现，明确拒绝未知规则，不能静默改变历史 seed。

工况越长可能得到越多 decision；等概率 episode 不等于相同样本比例，各算法有限步数内的采样计数也未被人为配平。每个真实 decision 才增加样本数。完整回合、部分预算截断回合分别计数；末尾无 step 的自动 reset 不计入经验回合。QP 是当前决策块失败数，逐块求和。回报采用实际桥接 reward。

```powershell
python python/dataset_manager.py summary --manifest datasets/carsim/manifests/dataset_v1.json --run 'runs/实际算法目录/实际训练子目录' --out datasets/carsim/reports/实际新统计目录
# 可重复 --run 汇总相同 dataset_hash 的不同算法/种子；不能重复传同一 run。
```

输出 dataset_summary.csv、dataset_manifest.json、dataset_statistics.json、dataset_table.md/png/pdf 和训练后的 decision_exposure 图；按速度、附着、参考类型、速度模式、split、工况类型分组。训练前 episode/decision 为 null/待统计。样本来自 manifest 中真实 exposure_by_scenario；每回合明细在 solver_train/episode_outcomes.jsonl，源采样序列在 episode_scenarios.jsonl。不同版本日志或样本总数不符会被拒绝。

`mean_episode_return` 按所有有样本的回合求均值，包含预算结束的部分回合；它不是仅完整回合的平均回报。完整训练回报曲线另从 Monitor 的实际完整回合日志生成。指定完成的联合选模 run 时，summary 同时输出逐算法周期验证、最佳合法/final 指标、模型锁、训练曲线和双误差图，不会重新选模或运行 test。

训练与实际验收结果见 [RESEARCH_RESULTS.md](RESEARCH_RESULTS.md)。
