# 结构整理、依赖与回归审计（2026-11-18）

起点 e83f7fb，git status 干净。先检查 AGENTS、README、CMake、git ls-files、ignore、目录清单，再做 Python AST、CLI/文本引用、INI 继承、C++配置使用、64份训练 manifest 和数据集 manifest 审查。本轮不改控制器、车辆、轮胎、MPC、奖励、动作边界、安全门槛或训练行为。

## 可审查明细

逐文件原路径、用途、引用、模型 manifest、兼容要求和最终处理见 [完整清单](archive/refactor_20261118_file_actions.csv)。完整文件系统库存、AST import、来源引用与本轮仿真留在 runs/refactor_20261118/。不以名称或“无直接 import”单独判定冗余；CLI、测试和历史复现同样算依赖。

## 实际删除、整合和保留

删除原路径共12个，原逻辑内容共69342字节（其中三份说明全文转入新核心说明，因此这不是净磁盘释放）：

- `docs/Control_Algorithm_Implementation.md`
- `docs/CarSim_Extensible_Dataset_System.md`
- `docs/DRL_MPC_MultiScenario_Training.md`
- `docs/RL_Training_Parameters.md`
- `docs/TQC_Design.md`
- `docs/Main_Project_Audit_20261114.md`
- `docs/Formal_DRL_MPC_Comparison_Plan.md`
- `configs/next_stage_scenario_request.json`
- `configs/real_scenario_generation_plan.json`
- `configs/scenario_inventory_current.json`
- `configs/real_cases_65.local.json`
- `python/project_cleanup.py`

三份全文整合：Control_Algorithm_Implementation→MPC_MODEL_AND_BASELINES；CarSim_Extensible_Dataset_System→DATASET_GUIDE；DRL_MPC_MultiScenario_Training→RESEARCH_RESULTS。另两份算法/参数说明提取为 DRL_DESIGN_AND_REWARD（包含完整公式、20维顺序、TQC机制和选择规则）；原始精确参数仍在未修改的INI。旧正式计划与重复索引由当前实验指南和HISTORY_INDEX替代。

16份含独特结果/真实数据出处的阶段报告移入 archive，完整路径见 HISTORY_INDEX；并修复可解析的相对文档链接。这些迁移不算删除数据。archive 中指标 CSV、模型锁 JSON 与来源证据不删除、不改写。旧文本中的过时规划路径按父提交追溯，不把旧设计值冒充真实工况。

配置：4份 JSON 仅是旧规划/路径子集，不是车辆参数或冻结模型链。全部既有 INI 逐字节与任务开始时 SHA256 比较一致；magic_tire.csv、default/strong及所有1D/2D/旧3D/新3D/4D、wy2/wy4、Actor/噪声及heading目标消融保留。没有发现可安全合并且完全等义、无复现依赖的 INI，故不强行删改或移动共同 strong_actions4 基础。

Python：dataset_manager 内107行报告绘图迁入 report.py，原 write_training_report 函数保留委托兼容入口，数据注册不再承担训练图的实现。实际删除164行的一次性 project_cleanup 工具，旧行为可从Git恢复。诊断工具其余均有不同实验问题或现有测试/历史引用：joint_feasibility（45+26扫描）、action_mask_diagnosis（动作因果闭环）、lateral_regression（X对齐与奖励重建）、qy_mechanism（反馈/位置调度）、strong_report（未版本化旧run）、strong_response（限速）、joint_selection（旧批量选模）、train_four/compare_four（旧套件）、sweep（旧扫描）、diagnostics（传感一致性）等保留。未通过删测试掩盖失效。

未发现允许范围内的 __pycache__/pyc/pytest/mypy/ruff 缓存，本轮缓存删除=0。未清理 runs、MATLAB、模型、成功实验、Release程序、构建目录或外部文献；不为追求比例删除有用研究工具。

## 目录前后

```text
之前                                  之后
configs/ 31文件（含旧规划）              configs/ 28文件（新增用途表；INI不动）
docs/ 24份根Markdown混合                 docs/ 9份根Markdown（六核心+导航/历史/审计）
python/ 32个根模块                       python/ 31个根模块（单一日常报告实现）
无统一HTML离线总览                      reports/research_overview_20261118/index.html
src、datasets、原runs、MATLAB            内容和职责保持
```

## 文件数量和空间口径

文件系统测量排除 .git 与本轮 runs/refactor_20261118 审计/验证输出；包含原实验、构建、MATLAB及新报告。测量时刻（不含随后写入的本审计文档自身）如下：

|范围|整理前|整理后测量点|
|---|---:|---:|
|文件数|216268|216342|
|逻辑字节|26186669394|26191864777|

净空间没有减少：新增可重建报告和来源证据超过删除量，且保留约20GB历史runs。本轮重点是入口减少和重复实现消除，不能把69342字节旧路径删除宣称为净释放；也不把真实验收生成文件当冗余清空。数字是逻辑字节，不是文件系统分配块或压缩后大小。审计原始CSV/JSON与恢复基点在本机保留。

## 报告与数值验证

统一入口 report.py --runs ... --out NEW；全部65/75开发验证默认显示，原始/强MPC、best/final和未达标结果同时保留。输出 HTML、完整精度 CSV、含源SHA/配置/版本的 summary.json、PNG/PDF。HTML只将显示精度舍入，CSV不改值；单seed无伪标准差，多seed以实际样本标准差记录，未学习的固定Np不画成自适应。补齐旧纯固定MPC报告，避免没有DRL行时 StopIteration。

离线示例读取真实 dataset_v1 三算法run，逐项对比旧12条best/final记录，误差指标及改善率差异≤1e-12；并加入4条原始/强MPC基线记录。浏览器已实际打开本地HTTP页面、检查标题/表格/图像显示；截图位于本轮审计目录。没有重新训练或用test结果选图/排序。

## 验收

- Release构建成功，CTest 1/1。
- pytest 82项全部通过（保留原80项，新增真实离线示例的全工况、原指标一致与无虚假单seed不确定性测试）。
- evaluate_dual --split validation 真实回放：三算法×65/75六组，加对应原始/强MPC。RMSE、峰值、Ay、Beta对比既有锁定最佳记录，最大差异小于1e-9；计算时延不作为逐位一致要求。
- 3D/4D各全程157控制步 zero-action parity 通过。
- 旧1D/2D模型原配置加载并真实回放完赛：0.076194532m/0.900024047°；0.076485246m/0.912872472°。
- dataset_v1 内容及 hash 保持920bec95…baa5dc9f；train有序70/60/80，validation65/75，test70/.6未作策略评价。
- 六回合真实切换70→80→60→60→60→80正常，全部终点及QP通过；原生源和配置没有改写。

回归机器证据：runs/refactor_20261118/regression.json、legacy_replays.json、closed_loop/dual_baseline_metrics.csv、parity3/4、switch/、fixed_offline/。论文指标/原生文件不移动。Git e83f7fb 提供所有被删源码和说明的恢复基点，恢复方式见 HISTORY_INDEX；不覆盖当前分支或重写历史。
