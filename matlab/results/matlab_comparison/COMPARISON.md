# MATLAB/Simulink 原生实现及300回合实测

本轮已经纠正为用相同数学方法在MATLAB/Simulink中实现控制器。运行模型为`DRL_MPC_MATLAB.slx`，不是调用历史C++控制器。原C++结果仅以已保存数据的形式用于数值校验。

## 实现与独立性证据

- 算法在`controllers/matlab/`的14个MATLAB文件中：八状态车辆模型、轮胎插值、Tustin离散化、LTV预测与灵敏度、完整参考样条、指数分步误差权重、H/g组装、约束、ADMM、热启动与滚动施加。
- Simulink内部有七个可查看的控制计算阶段，记忆量由Unit Delay保存；共12个MATLAB Function和6个实际Scope。控制器手写S-Function为0，仅外部车辆动力学接口CarSim vs_sf保留。
- 初始化直接读CarSim echo、轮胎CSV、完整LTARG和INI；模型没有旧MEX重置回调或隐藏的控制器桥接调用。
- 四组共632个状态点的独立MATLAB回放，转角最大差约1.86e-13rad，QP收敛与迭代次数一致。
- 四组真实CarSim闭环最大转角差约2.62e-13rad，最大状态分量差约3.65e-9，QP失败均为0。独立进程运行前后均未加载mpc_mex/mpc_paper_mex。
- 最终公共入口复跑与本轮原生评价的状态和转角差异均为0；14文件Code Analyzer消息为0，非收敛投影、无效预测保持、缺失力回退、转角速率边界及λ真实作用检查均通过。

数值一致证明的是方法实现正确，不代表更换语言本身提高跟踪精度。证据见`../matlab_native_validation/acceptance.json`、`replay.csv`、`closed_loop_120m.csv`、`code_review.json`与`delivery_audit.json`。

## 本版新训练的真实结果

指标已修正为0≤X≤120，移除了两个负X启动样本，未修改仿真数据或Agent。旧口径记录见../cleanup/metrics_before_range_fix.csv；基础MPC独立诊断见../mpc_diagnosis/DIAGNOSIS.md。

从新Agent开始，在原生MATLAB/Simulink环境实际训练300回合、47100次交互，种子42，Actor参数最大绝对变化0.8821015432。Np=40、Nc=12、Ts=0.04s。历史C++接口版的300回合未计入本版。

评价关闭探索噪声，车辆、参考、控制周期和配置一致；误差严格统计0≤X≤120m的155个采样点，各组均到达120m，距离图均限定0–120m。离散停止的最后原始点可以略大于120m。

| 控制器 | 横向RMSE/m | 最大横向误差/m | 航向RMSE/rad | QP失败 |
|---|---:|---:|---:|---:|
| 原生固定MPC | 0.054056 | 0.148822 | 0.020752 | 0 |
| 原生TD3最终第300回合 | 0.058495 | 0.170981 | 0.017496 | 0 |
| 最终策略λ置零（推理消融） | 0.057331 | 0.162565 | 0.018097 | 0 |
| 原生训练奖励选取第160回合 | 0.051562 | 0.137237 | 0.019675 | 0 |

本轮奖励选取策略相对原生固定MPC：横向RMSE改善4.61%，最大绝对横向误差改善7.78%，航向RMSE改善5.19%。最终第300回合策略横向指标仍有退化，所有结果完整保留。该对比只反映单一种子、单一名义工况，尚无统计显著性或泛化结论。

检查点规则是在每10回合保存点上，选此前10回合平均训练奖励最高者，本轮为第160回合，均值-7496.7485；没有用评价误差筛选检查点。λ置零只是同一最终策略的推理时消融，并非另行训练的另一Agent。绝对误差奖励、控制惩罚与RMSE/峰值评价之间的目标差异仍存在，本轮实现方式更正没有暗中改变奖励后再混用旧结果。

## 运行

```matlab
cd('C:\Users\fengchengmin\Desktop\JQRKZ\DRL-MPC\matlab')
open_matlab_model
outFixed = run_matlab_preview;
open_matlab_scopes('Tracking_Errors')
outSelected = run_matlab_preview('selected');
open_matlab_scopes('Tracking_Errors')
open_matlab_scopes('Steering')
```

其他Scope为Weights、Lambda、Reward、QP_Status。旧open_paper_model/run_paper_preview/open_paper_scopes已转到本版。操作说明及逐阶段源码映射见`../matlab_native_validation/USER_GUIDE.md`。

## 产物

- `comparison_120m.png/.fig`：本版固定、最终策略、λ消融与奖励选取策略的路径、误差、转角和权重。
- `training.png/.fig`：本版实际300回合奖励曲线。
- `metrics_120m.csv`、`*_raw.mat`：完整指标与SimulationOutput。
- `checkpoint_selection.json`、`provenance.mat`：本版Agent路径与选取依据。
- `../matlab_td3_ep300_seed42/`：本版新训练Agent、检查点、逐回合CSV、日志、summary.json。
- `../matlab_native_validation/`：实现验收、算法图、Scope图、参数快照及交付文件摘要。

当前仍保留论文方法适配中已明确说明的采样周期、模型、权重尺度及求解器差异；此次完成的是本项目当前方法的MATLAB/Simulink原生实现，不能据此声称已经严格复现论文全部数值和工况。
