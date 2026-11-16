# MATLAB工程去冗余记录

本次整理范围为matlab迁移工程。当前原生控制器、原始CarSim参考、Np=40/Nc=12、最终Agent和选取Agent160保持；原C++工程的src/python等目录没有参与整理。

## 实际结果

- 原工程约20.06 GiB、752个文件。
- 586个旧文件已从活动目录移入可恢复隔离区，合计19.16 GiB；另收拢了87个空目录。
- 已校验历史压缩归档约2.51 GiB。
- 活动目录只保留1个原生模型、14个控制器MATLAB源码、9个根入口（含3个兼容入口）、3个RL工具和9个实用工具。
- 当前训练保留final_agent.mat和checkpoints/Agent160.mat。其余58个周期检查点已隔离；这些周期检查点没有额外压入历史ZIP。

**没有执行永久删除，已释放物理磁盘空间为0。** 自动审批拒绝了批量Remove-Item，返回信息仅为blocked by policy。因此采用可恢复移动完成工作目录整理，不能将隔离体积表述为已释放空间。

## 位置与恢复

- 隔离区：`archives/retired_20261113_211219/`，原相对路径保持，可按清单移回。
- 历史ZIP：`archives/legacy_migration_20261113_211219.zip`，保留旧模型、历史最终Agent、原始实验、旧控制器副本及迁移工具，逐文件SHA256校验通过。
- `manifest.json`记录原计划、路径、大小和摘要；`retirement_results.json`记录实际移动结果。planned_action是被拦截前的计划，实际执行模式明确为reversible_retirement。
- 58个非保留周期检查点只在隔离区，删除隔离区后不再保留这些训练中间状态；原生最终和选取策略仍在活动训练目录。
- 若用户决定移除已验证的隔离区，该目录本身体积约19.16 GiB；历史ZIP仍独立保留。当前并未代为执行该永久删除动作。

不要把archives加入MATLAB路径；.gitignore已排除archives。旧模型、旧setup/训练入口和C++接口工具不在活动目录。

## 依赖调整与验证

- 四组历史数值参考压缩为tests/fixtures/controller_reference.mat，632个记录点，不包含C++可执行模块或旧Agent。
- 算法回放与边界检查已改为使用此独立数值夹具；真实闭环回归改为使用当前原生Agent与当前保存结果。
- 模型、14个控制器源文件、原生最终Agent与Agent160的共17个受保护文件，与整理前SHA256完全一致。
- 清理后静态分析通过、632点回放通过，四组真实CarSim闭环回归通过，状态／转角／奖励最大差异全部为0，没有加载旧控制器MEX。
- 证据为../matlab_native_validation/regression_after_cleanup.json、regression_120m.csv、code_review.json、replay.csv与本目录verification.log。

## 统计口径修正

同时修复了指标只筛选X≤120的问题：现在严格0≤X≤120，排除两个负X启动样本，共155个样本。旧指标保存在metrics_before_range_fix.csv，未改变控制器或Agent。当前固定横向RMSE为0.054056m，选取策略为0.051562m。

基础MPC独立诊断见../mpc_diagnosis/DIAGNOSIS.md。当前默认算法没有被静默更换。
