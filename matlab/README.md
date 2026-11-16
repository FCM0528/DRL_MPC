# MATLAB/Simulink 原生 DRL-MPC 工程

当前模型为 `models/DRL_MPC_MATLAB.slx`，默认保持原70 km/h工况并关闭降速规划。纵向速度用于三自由度预测与侧向约束，不作为降低至约56 km/h的优化目标。固定 Np=40、Nc=12、Ts=0.04 s；保留原始LTARG和两路前轮转向接口。控制器由MATLAB函数和Simulink离散状态实现，不调用旧C++控制器；外部车辆接口为CarSim官方vs_sf。

## 打开与验证

```matlab
cd('C:\Users\fengchengmin\Desktop\JQRKZ\DRL-MPC\matlab')
open_matlab_model
out = run_corrected_preview;          % 原70 km/h工况，关闭降速规划
open_matlab_scopes('Tracking_Errors')
open_matlab_scopes('Speed_Curvature')
open_matlab_scopes('Lateral_Capacity')
open_matlab_scopes('Constraint_Slack')
```

共有10个实际连接的 Scope，中文注释使用宋体。其他 Scope：Steering、Weights、Lambda、Reward、QP_Status、Sideslip。所有距离图和指标限定0–120 m。

此前大幅降速到约56 km/h的结果已撤回。调整前的同速校正也未改善基线；将轮胎刚度同步到转向前馈并减少不必要的转角变化惩罚后，同车速RMSE降至约0.0500 m，原基线约0.0541 m。结果与边界检查见 `results/nominal_speed_correction/`。旧降速记录不作为当前效果证据。

默认保留实测轮胎力的瞬时残差及衰减补偿。额外积分式扰动观测器在当前工况的闭环消融中表现较差，默认关闭，保留在profile 2中。

| profile | 预测及补偿 | 新增侧向约束 |
|---|---|---|
| 0 | 原始预测模型 | 关闭 |
| 1 | 校正模型、实测轮胎力残差补偿 | 关闭 |
| 2 | profile 1 + 有界积分扰动观测器 | 关闭 |
| 3（默认） | profile 1 | 开启，额外积分观测器关闭 |

```matlab
run_corrected_preview(1,false) % 同原70 km/h工况，去掉新增侧向约束
run_corrected_preview(2,false) % 同车速下的可选扰动观测器消融
run_corrected_preview(0,false) % 原始70 km/h基线
```

默认模型、预览、训练及比较均使用 `carsim/case` 原始70 km/h工况。历史 `carsim/coordinated_case` 不接入当前工作流；预览入口拒绝启用它。新增转角速度120°/s是实验设计值，尚未确认车辆执行器规格。侧向预测约束是带惩罚的软约束；若同车速存在跟踪／约束冲突，应报告超限和误差，不能通过降低车速隐藏。

## 历史策略与后续训练

```matlab
run_matlab_preview('fixed')    % 原预测模型、原70 km/h工况
run_matlab_preview('selected') % 历史300回合的第160回合策略
run_matlab_preview('final')    % 历史300回合最终策略
```

这些入口显式使用旧环境。已有 `results/matlab_td3_ep300_seed42` 的300回合、47100次交互属于旧预测环境；本次校正环境的训练回合数为0。

```matlab
train_matlab_td3(300,42)       % 后续需要时，从新Agent训练校正环境
```

新训练写入 `results/matlab_td3_corrected70_ep300_seed42`，不覆盖旧Agent；对比写入 `results/nominal_speed_correction/td3_comparison`。训练目标车速保持70 km/h；本次任务未启动新训练。

## 源码与证据

- `controllers/matlab/`：纯MATLAB预测、补偿、约束和QP源码。
- `configs/prediction_calibration.mat`、`configs/nominal_speed_tuning.json`：冻结的有效预测参数和经原速度闭环对照的固定MPC参数，不是CarSim全部物理参数的唯一辨识。
- `utils/run_correction_experiments.m`：闭环消融和历史基线逐点回归。
- `utils/verify_correction_math.m`：新增QP与独立quadprog核对。
- `results/nominal_speed_correction/`：当前同车速结果、对照图和审计；本工况横向RMSE约改善7.6%。
- `results/prediction_correction/`：历史预测校验及降速实验，降速结果不适用于当前要求。
- `results/matlab_native_validation/`、`results/matlab_comparison/`：历史原生实现与旧环境训练证据。
- `results/mpc_diagnosis/DIAGNOSIS.md`：本次修正前的问题诊断。
- `results/cleanup/CLEANUP.md`：上一阶段归档清理记录；archives不加入MATLAB路径。

本轮沿用已读取的MATLAB/Simulink Skills流程。MCP附着与SATK编辑路径此前失败或崩溃，实际模型修改及验证通过独立MATLAB批处理进程和原生API完成；MCP工具可见性不作为其成功执行或Skills已安装的证据。当前结果属于离线联合仿真验证，不代表实时部署合格。
