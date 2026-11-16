# 项目清理审计（2026-11-16）

## 检查范围和测量口径

任务起始先执行git status，输出为空，本地HEAD为4a2e7ed。读取AGENTS.md，确认按完整任务提交、正常推送上游且排除runs/模型/构建产物。本轮修改未覆盖其他未提交工作。

实际检查git ls-files（起始266个跟踪文件）、git status、Python AST import关系、脚本CLI调用与README/docs路径引用、测试依赖，以及所有本地目录文件数量和逻辑大小。记录位于 `runs/project_cleanup_audit_20261116_v1/`：git_ls_files.txt、git_status_before.txt、python_imports.csv、cli_and_document_references.csv、summary.json等。

清理前/后是清理操作两个相邻时点的快照，排除.git和审计自身输出，计入MATLAB但不清理它。以文件逻辑字节计，不把稀疏文件/NTFS分配量当成逻辑大小；后续报告生成、CTest日志更新和Git提交不属于此窗口。

| 指标 | 清理前 | 清理后 | 减少 |
|---|---:|---:|---:|
| 文件数 | 186448 | 186348 | **100** |
| 逻辑字节 | 22033873269 | 22033015894 | **857375**（约0.818 MiB） |

审计文件自身本地约74234字节（记录时），独立列出，不隐瞒审计开销。完整前后清单最初共约55MB CSV，已另存工作区外的complete_inventory.zip（1957401字节）并逐文件SHA-256验证，再移除本轮生成的两份本地CSV；清单地址与原哈希记录于inventory_location.json。这两份审计自生成文件不计入上表原项目100项。

## 实际删除和归档

**完整100项逐文件清单及理由**见[cleanup_removed_files_20261116.csv](archive/cleanup_removed_files_20261116.csv)，包括95个可再生Python/pytest缓存文件和5个已备份失败实验文件。仅删除不受Git跟踪且不在MATLAB内的__pycache__、*.pyc、.pytest_cache内容。本机未发现需要处理的.mypy_cache或.ruff_cache。空缓存目录随之移除。

指定失败目录均在验证修复输出后处理；不存在的目录会跳过，未执行git clean。

| 失败目录 | 文件数 | 修复版本核验 | 处理 |
|---|---:|---|---|
| runs/strong_slew_response_20261116_v1 | 3 | v3的3组均完赛、QP零失败 | 外部ZIP逐文件哈希验证后移除本地失败目录 |
| runs/td3_lateral_audit_20261116_v1 | 2 | v2分项重算残差全部小于1e-9 | 同上；不含独有模型或训练轨迹 |
| runs/td3_reward_comparison_20261116_v1 | 0 | v2六条记录均完赛且稳定性合格 | 备份空目录清单后移除空目录 |

完整归档来源、成员哈希、修复依据和ZIP哈希见[cleanup_external_archives_20261116.json](archive/cleanup_external_archives_20261116.json)。失败内容只有无效配置、空日志/初始化记录和部分重复分析CSV，没有模型、经验数据或关键唯一轨迹。v2/v3成功结果保持原路径。

文档归档：原README全文（相对链接同步调整）存为 `docs/archive/README_legacy_20261114.md`；原 `docs/Main_Project_Audit_20261114.md` 的全文（数据链接层级调整）存为 `docs/archive/Main_Project_Audit_20261114.md`。原报告路径保留跳转stub，源码注释/历史链接不失效。新增INDEX.md连接当前入口与历史报告；README和RL_Training_Parameters纠正“只比较final模型”和单一固定基线的过时说明。历史研究结论未改写为当前算法排名。

## 外部备份与恢复

工作区外目录：`C:/Users/fengchengmin/Desktop/JQRKZ/DRL_MPC_cleanup_backup_20261116_v1/`。包含三个失败目录ZIP及complete_inventory.zip。该目录不在Git仓库中，不随GitHub推送；需要迁移电脑时另行复制备份。每份ZIP均执行testzip与逐文件SHA-256比对，校验成功后才清理。

```powershell
# 先确认目标不存在，避免覆盖；将目录名替换为要恢复的失败目录
$name = 'strong_slew_response_20261116_v1'
if (Test-Path -LiteralPath "runs/$name") { throw '目标已存在，停止恢复' }
Expand-Archive -LiteralPath "C:/Users/fengchengmin/Desktop/JQRKZ/DRL_MPC_cleanup_backup_20261116_v1/$name.zip" -DestinationPath "runs/$name"
# complete_inventory.zip解压到独立审计目录可恢复两份完整CSV；空ZIP可按元数据手动创建空目录
```

95个缓存文件不备份，运行Python/pytest即可再生。历史文档还可从Git起点4a2e7ed恢复原文。新增project_cleanup.py将来默认仅审计；--apply才按同样的路径边界、保护范围和外部备份规则执行，输出及备份均必须新目录。

## 保留项目与空间大项

| 目录 | 清理前文件数 | 清理前逻辑大小约 | 决定 |
|---|---:|---:|---|
| runs | 157300 | 14.80 GiB | 成功实验全部保留，含周期模型、manifest、完整训练轨迹和冻结来源 |
| matlab | 1375 | 3.52 GiB | 完整保留，未删除、未移动 |
| out | 26092 | 1.80 GiB | 所有构建目录保留，只移除2个可再生.pyc缓存 |
| .vs | 21 | 0.286 GiB | 未确认是否仍用于本机开发，保留 |
| results | 1459 | 0.088 GiB | 保留历史结果 |

`out/python-rl-build/Release` 当前可用程序完整保留。另有约30个历史out子目录，可能仍被旧CLI报告、CarSim接口试验或历史复现使用，未证实可删除，因此本轮不清理。根目录~$cache1虽像临时文件，但来源不明，保留；PDF、AGENTS.md、.gitignore也保留。

MATLAB结果检测到一组内容相同的495字节CSV：`matlab/results/prediction_correction/corrected_planned_metrics.csv`与`latest_experiment_metrics.csv`，SHA-256为155c3a8588774f80a2d99e3ec48ec898eb73ad0ffa065341f3d73fd2b15c2682。这可能是有意义的固定版本与latest别名，**仅列出归档建议，两个文件均保留**；未发现需处理的backups重复组。详细记录于matlab_duplicate_candidates.json。

## 依赖审查与核心结构

保留sweep.py（旧扫参默认行为）、joint_feasibility.py（双指标45+26扫描）、control_sensitivity.py（单因素）、action_mask_diagnosis.py（真实动作归因）、lateral_regression.py（X对齐奖励审计）、reward_ablation_report.py（奖励三组比较）、strong_response.py、strong_report.py及parity.py等历史工具。它们分别有文档或CLI复现用途，没有足够证据认定可删。新的fixed_np_experiments复用lateral_regression.write_rows与report.control_trace；动作映射从环境抽到action_mapping，env/evaluate/run_policy/训练契约共同使用，避免重复实现。

没有移动Python脚本或配置；历史模型的冻结INI依赖及配置哈希继续有效。旧2维Np+Qy、旧3维和旧4维语义保留，不复用为新的Qy+Qpsi。

```text
src/                  C++ CarSim接口、MPC、奖励与桥接
python/               训练/验证/测试入口，命名动作映射，共享报告及历史诊断
python/tests/, tests/ Python与C++测试
configs/              默认、强MPC、新旧动作及奖励消融配置
docs/INDEX.md         当前入口和历史出处
docs/archive/         历史报告、旧README、完整删除/备份清单
runs/, results/       本机可复现模型、轨迹、指标及冻结工况（Git忽略）
out/                  当前与历史构建程序（保留）
matlab/               独立研究工程（不清理）
```

## 已验证入口

Release构建、CTest 1/1、全部pytest 40/40通过；新旧动作映射、名称/维数校验、同维不同语义拒绝加载、中性强MPC参数均通过测试。新一维/二维配置和旧三维配置分别完成全程157控制步真实CarSim零动作parity。train.py已真实完成两个7900决策试验；run_policy.py已分别加载一维/二维最佳合法模型并全程完赛；fixed_np_experiments scan/compare与strong_report实际生成了结果。成功实验没有删除checkpoints或训练数据。

清理后复测采用PYTHONDONTWRITEBYTECODE=1及pytest -p no:cacheprovider，避免刚清理的缓存被测试重建。未执行正式三算法或500回合训练。
