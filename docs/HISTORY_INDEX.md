# 历史入口与恢复

本轮父提交固定为 `e83f7fb`。任何删除文件均可查看：

```powershell
git show e83f7fb:python/project_cleanup.py
# 只恢复确需复现的单个旧文件，不重置当前分支或覆盖当前修改：
git restore --source=e83f7fb --worktree -- docs/RL_Training_Parameters.md
```

- `CarSim_60kmh_Training_Case.md` → docs\archive\CarSim_60kmh_Training_Case.md
- `CarSim_65kmh_Validation.md` → docs\archive\CarSim_65kmh_Validation.md
- `CarSim_70kmh_mu06_Preflight.md` → docs\archive\CarSim_70kmh_mu06_Preflight.md
- `CarSim_80kmh_Training_Case.md` → docs\archive\CarSim_80kmh_Training_Case.md
- `CarSim_CPP_Integration.md` → docs\archive\CarSim_CPP_Integration.md
- `DRL_Diagnosis_and_Ablation.md` → docs\archive\DRL_Diagnosis_and_Ablation.md
- `DRL_MPC_Mainline_Recovery.md` → docs\archive\DRL_MPC_Mainline_Recovery.md
- `DRL_MPC_Multi_Scenario_Preparation.md` → docs\archive\DRL_MPC_Multi_Scenario_Preparation.md
- `DRL_MPC_Next_Stage_Analysis.md` → docs\archive\DRL_MPC_Next_Stage_Analysis.md
- `High_Speed_Validation.md` → docs\archive\High_Speed_Validation.md
- `Joint_Feasibility_and_Action_Ablation.md` → docs\archive\Joint_Feasibility_and_Action_Ablation.md
- `PROJECT_CLEANUP_AUDIT.md` → docs\archive\PROJECT_CLEANUP_AUDIT.md
- `Strong_MPC_DRL_Training_Plan.md` → docs\archive\Strong_MPC_DRL_Training_Plan.md
- `TD3_Fixed_Np_Experiments.md` → docs\archive\TD3_Fixed_Np_Experiments.md
- `TD3_Lateral_Regression_Diagnosis.md` → docs\archive\TD3_Lateral_Regression_Diagnosis.md
- `TD3_Qy_Stability_and_Generalization.md` → docs\archive\TD3_Qy_Stability_and_Generalization.md


实际删除：Control_Algorithm_Implementation → MPC_MODEL_AND_BASELINES；CarSim_Extensible_Dataset_System → DATASET_GUIDE；DRL_MPC_MultiScenario_Training → RESEARCH_RESULTS。RL_Training_Parameters/TQC_Design → DRL_DESIGN_AND_REWARD 与原始 INI；Formal_DRL_MPC_Comparison_Plan 被正式实验入口和当前指南替代；Main_Project_Audit_20261114 只是一份已归档原文的重复指针。

删除四份旧工况规划 JSON：next_stage_scenario_request、real_scenario_generation_plan、scenario_inventory_current、real_cases_65.local。它们不是原生数据，也不是模型配置链；已被不可变 dataset_v1 实际划分取代。历史文中提及的设计值不代表已生成工况，可从父提交核查。project_cleanup.py 为上轮一次性备份清理器，本轮有完整新审计与显式 git 删除记录，不再作为日常入口；不删除独特控制诊断工具。

历史核心消融、动作屏蔽、45+26扫描、工况验收、模型锁、唯一指标与原始引用保留在 archive/ 及原 runs。旧命令使用旧版本源码/原模型冻结配置复现，不能将旧配置名称改义。历史报告的当时测试数和就绪状态不代表当前状态。
