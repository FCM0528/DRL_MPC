# MATLAB/Simulink 原生 MPC 使用与验收

## 当前实现

模型为`matlab/models/DRL_MPC_MATLAB.slx`。预测、线性化、Tustin离散化、灵敏度递推、代价、约束、ADMM QP及滚动控制全部由`matlab/controllers/matlab/`内的MATLAB源文件实现；不调用原C++控制器。Np=40、Nc=12、Ts=0.04s。

仅车辆动力学保留CarSim官方`vs_sf`，控制器自定义S-Function数量为0。Simulink为MATLAB Function自动生成的执行包装属于平台编译机制，不是对历史C++ MPC的调用。模型有12个MATLAB Function、6个实际Scope。

## 直接运行

建议在新的MATLAB会话中验证，便于检查历史控制器MEX未加载。不要在其他会话同时运行同一个CarSim工况。

```matlab
cd('C:\Users\fengchengmin\Desktop\JQRKZ\DRL-MPC\matlab')
open_matlab_model
outFixed = run_matlab_preview; % 原生固定MPC，X达到120m后离散终止
open_matlab_scopes('Tracking_Errors')
open_matlab_scopes('Steering')
```

本版已经在原生环境完成300回合、47100次交互，Actor参数最大绝对变化0.8821015432。查看本版策略：

```matlab
outSelected = run_matlab_preview('selected'); % 本版训练奖励选出的检查点，关闭探索噪声
open_matlab_scopes('Tracking_Errors')
outFinal = run_matlab_preview('final');       % 本版第300回合最终Agent
```

Scope名称为`Tracking_Errors`、`Steering`、`Weights`、`Lambda`、`Reward`、`QP_Status`。也可进入Monitoring_Evaluation直接双击。后一次仿真会更新Scope；若要并排比较，保留返回的out变量或查看离线图。

Scope横轴为时间，距离对比图限定0–120m，训练曲线横轴为回合。车辆在X≥120m后离散停止，末个原始记录点可能超过120m，误差指标只使用X≤120m的样本。

为兼容原操作习惯，`open_paper_model`、`run_paper_preview`和`open_paper_scopes`已转到本版。旧的历史C++接口入口已移入archives，当前原生工作流只保留本版入口。

## 控制器算法在哪里

| Simulink阶段 | MATLAB实现 | 内容 |
|---|---|---|
| Operating_Point | matlabOperatingPoint.m | 坐标与状态重排、相邻车速差分得到纵向加速度 |
| LTV_Prediction | matlabNominalPrediction.m、matlabVehicleModel.m | Nc解移位热启动、八状态局部Jacobian、Tustin、预测与灵敏度 |
| Prediction_Reference | matlabPredictionReference.m、matlabReference.m | 完整LTARG自然三次样条、导数和曲率前馈 |
| Quadratic_Cost | matlabCostMatrices.m | 增量控制凝聚、H/g、指数分步Q及转角／增量惩罚 |
| Actuator_Bounds | matlabConstraintBounds.m | 累加转角与单步增量上下界 |
| ADMM_QP_Solver | matlabAdmmQp.m | MATLAB编写的Cholesky、ADMM、原始／对偶残差与停止条件 |
| Receding_Control | matlabSteeringCommand.m | 约束投影、首步施加、失败保持及记忆量 |

状态顺序为[vy,vx,psi,r,Y,X,alpha_f,alpha_r]；保留轮胎松弛、实测轴侧向力残差衰减及纵向加速度衰减。`loadMatlabMpcParameters.m`直接读取CarSim echo、轮胎CSV和INI；参考来自完整原LTARG表，不另外生成轨迹。

上次转角、增量、车速、Nc控制序列和解有效标志由Simulink Unit Delay保存并随每次仿真复位，没有隐藏的C++全局状态。算法MATLAB源码可直接编辑、审查和离线调试；`matlabMpcStep.m`按相同阶段提供离线入口。

## 数值验证

`replay.csv`：四组共632个真实记录状态，MATLAB独立回放对比历史结果，转角最大差异约1.86e-13rad，QP收敛标志与迭代次数全部一致。历史数据仅作数值校验依据。

`closed_loop_120m.csv`与`acceptance.json`：四组真实CarSim闭环，在不加载历史控制器MEX的进程中运行。最大转角差异约2.62e-13rad，最大状态分量差异约3.65e-9，最大逐步奖励差异约1.02e-10；四组QP失败数均为0，均到达120m。

这里固定、最终策略、λ置零与检查点Agent来自历史实验，目的仅为隔离后端实现差异，不能称为原生环境重新训练的成绩。原生训练结果单独保存在下述目录。

`code_review.json`记录MATLAB Code Analyzer结果、QP非收敛投影、无效预测保持、缺失实测力回退、有限转角速率边界及λ实际改变控制结果的检查。

## 原生训练和比较

```matlab
train_matlab_td3(300,42) % 从新Agent开始，已存在最终结果时拒绝覆盖
compare_matlab_controllers
```

新训练目录：`results/matlab_td3_ep300_seed42/`；新比较目录：`results/matlab_comparison/`。是否完成原生300回合以`acceptance.json`中的`native_training_completed`以及新目录`summary.json`为准。旧`observed_td3_ep300_seed42`属于历史C++后端训练。

本轮原生训练的奖励选取检查点仍为第160回合，但与历史接口版是分别训练、分别保存的Agent。完整原生指标见`../matlab_comparison/COMPARISON.md`。最终交付复跑的状态与转角差异均为0，`code_review.json`的静态分析消息为0。

公共预览检查本次仿真没有新加载旧控制器MEX；旧桌面会话已经加载过某个MEX，不等于本版调用了它。严格验收在独立新进程进行，并确认运行前后均没有加载任何旧控制器MEX，记录在`delivery_audit.json`。

比较固定MPC、本版最终TD3、本版最终策略λ置零、本版按训练奖励选取检查点。λ置零是推理消融，不是单独训练的另一种Agent。检查点按每10回合保存点之前10回合平均训练奖励最高选取，未用评价误差筛选。仅在同一名义工况训练和评价，不宣称统计显著改善或跨工况泛化。

## 依赖和执行边界

需要MATLAB R2025a、Simulink、Stateflow/MATLAB Function支持、Reinforcement Learning Toolbox、Deep Learning Toolbox和当前CarSim许可。不需要调用原C++ MPC二进制，也不需要Optimization Toolbox的quadprog；QP求解器已用MATLAB重写。实际网络在CPU上训练。

本次读取了building-simulink-models、matlab-review-code及simulating-simulink-models等本地Skills；模型创建和验证使用独立MATLAB批处理进程与官方Simulink/Stateflow API。此前SATK自动编辑路径触发过崩溃，保持停用；MCP工具发现不代表附着成功，本次结果没有依赖MCP附着成功。

所有改动位于matlab目录，原C++工程保留为参考与校验来源。历史适配版模型和结果作为档案保留，当前模型的回调、MATLAB Function脚本和运行函数均不引用它们。

## 后续诊断与去冗余更新

基础MPC独立检查见../mpc_diagnosis/DIAGNOSIS.md。跟踪指标修正为0≤X≤120，共155个样本；固定横向RMSE为0.054056m，选取策略为0.051562m。此为统计口径修正，没有改变控制算法。旧的157样本等价验证记录保留为历史证据。

原始参考数值已压缩为tests/fixtures/controller_reference.mat；历史Agent、旧模型和迁移脚本已归档隔离。validate_matlab_closed_loop现使用本版Agent及本版保存结果，输出regression_after_cleanup.json，四组状态／转角／奖励差异均为0。Native最终Agent和Agent160保持不变。清理说明见../cleanup/CLEANUP.md。
