# 实验与工程文档索引

## 当前入口

- [可扩展 CarSim 原生工况库、注册、版本与样本统计](CarSim_Extensible_Dataset_System.md)
- [版本化多工况在线训练：三算法 seed42 真实预实验](DRL_MPC_MultiScenario_Training.md)
- [80km/h、mu=0.8验收与60/70/80三训练源切换](CarSim_80kmh_Training_Case.md)
- [60km/h、mu=0.8训练源验收与真实训练集回合切换](CarSim_60kmh_Training_Case.md)
- [70km/h、mu=0.6预留测试源冻结与固定MPC验收](CarSim_70kmh_mu06_Preflight.md)
- [多工况准备、75km/h验证与真实回合切换审计](DRL_MPC_Multi_Scenario_Preparation.md)
- [65km/h真实工况、双MPC预检与冻结三算法跨速度验证](CarSim_65kmh_Validation.md)
- [当前优化目标、三维动作、TQC训练审计与独立工况待办](DRL_MPC_Next_Stage_Analysis.md)
- [TQC-MPC主线恢复、三算法统一预实验与受限四维](DRL_MPC_Mainline_Recovery.md)
- [一维Qy五种子动作消融与反馈归因](TD3_Qy_Stability_and_Generalization.md)
- [一维profile的历史公平比较计划与阻塞预检](Formal_DRL_MPC_Comparison_Plan.md)
- [固定Np的TD3与恒定Qy对照](TD3_Fixed_Np_Experiments.md)
- [文件、依赖、备份与清理审计](PROJECT_CLEANUP_AUDIT.md)
- [强MPC与多工况训练流程](Strong_MPC_DRL_Training_Plan.md)
- [CarSim接口](CarSim_CPP_Integration.md)、[MPC实现](Control_Algorithm_Implementation.md)

## 研究历史与原始数据出处

- [横向退化、8组动作屏蔽、奖励消融](TD3_Lateral_Regression_Diagnosis.md)
- [双误差可行性与45+26组参数搜索](Joint_Feasibility_and_Action_Ablation.md)
- [联合奖励初次诊断](DRL_Diagnosis_and_Ablation.md)
- [2026-11-14原始扫描与历史三算法实验](archive/Main_Project_Audit_20261114.md)
- [历史README全文快照](archive/README_legacy_20261114.md)
- [训练参数历史及实现说明](RL_Training_Parameters.md)、[TQC设计](TQC_Design.md)、[高速工况历史](High_Speed_Validation.md)

历史报告保持当时实验的选择规则与结论；当前规范以README和当前入口为准。历史结果应结合冻结配置及工况阅读，不能跨实验拼接成算法排名。
