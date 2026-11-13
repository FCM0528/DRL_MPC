# 实际迁移验收记录

生成时间：2026-11-12T23:25:30+08:00

以下结果来自实际本机执行。MCP 初始会话 PID 38724 完成接口读写与仿真；该会话退出后，继续通过本机 MATLAB COM 执行。

最终 MATLAB PID：2300；版本：25.1.0.2943329 (R2025a)。

## 功能与证据

| 项目 | 实际状态 | 证据 |
| --- | --- | --- |
| 模块化模型 | 已创建、保存、编译，结构检查通过 | model_validation.json、model_structure.txt、model_overview.png |
| 20 路状态与单位转换 | 已在真实 CarSim 联合仿真中验证 | interface_units.txt、各组 trace.csv |
| 两路前轮控制权限 | 零、正、负指令与两侧反馈已验证；零指令覆盖 DLC 区段 | steering_authority.csv、steering_*.mat |
| CarSim 原始目标轨迹 | 完整 241 行，源与副本一致 | source_audit.json |
| 固定 LTV-MPC | 完成真实闭环，QP 状态与终点见下表 | fixed/raw.mat、fixed/trace.csv |
| 原 C++ 与 MEX | 首步控制、QP、预测状态、观测、参考、奖励数值对照通过 | native_parity.csv |
| 七阶段 MPC 在线调参 | Np 与所有权重候选回放通过 | adaptive_pipeline_parity.json、mex_pipeline_parity.csv |
| 重复 Reset | 完整固定闭环重复运行测试通过 | reset_validation.mat、test_results.txt |

## 正式训练

| Agent | 完成回合 | 策略交互 | Actor 最大参数变化 |
| --- | ---: | ---: | ---: |
| DDPG | 500 | 49000 | 2.0408666 |
| TD3 | 500 | 49000 | 1.1534967 |

## 同工况确定性对照

| 控制器 | 横向 RMSE (m) | 横向峰值 (m) | 航向 RMSE (rad) | QP 失败 | 完成 | 实际末点 X (m) |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| fixed | 0.255544 | 0.732051 | 0.039278 | 0 | 1 | 150.6803 |
| ddpg | 0.084690 | 0.223255 | 0.015065 | 0 | 1 | 150.5417 |
| td3 | 0.078728 | 0.242617 | 0.013189 | 0 | 1 | 150.5914 |

## 控制动作代价

| 控制器 | 最大转角 (deg) | 最大转向速率 (deg/s) | 最大侧偏角 (deg) | Np 切换次数 |
| --- | ---: | ---: | ---: | ---: |
| fixed | 5.950 | 15.763 | 2.177 | 0 |
| ddpg | 8.438 | 60.618 | 2.321 | 89 |
| td3 | 6.923 | 51.689 | 2.103 | 84 |

本次两种 RL 策略降低了跟踪误差，同时转向峰值和速率高于固定 MPC，且存在 Np 频繁切换。保留原动作映射与代价配置，未为美化曲线增加额外平滑；这些结果不构成整体控制品质均优的结论。

训练完成不等于策略优于固定 MPC；上表是所保存最终 Agent 的实际回放结果。全部使用同一 CarSim 工况、相同初始状态和质心 150 m 的终止判据。策略采样导致末点略超过 150 m，表中保留实际值。

## 适用范围与未验证项

- 已验证本机 R2025a / CarSim 2024.1 / 当前车辆 / DLC SYX / 70 km/h / μ=0.8；未据此证明跨车辆、跨工况或多随机种子的泛化。
- TQC 未实现。其他随机种子提供入口，本次正式实验为 seed=42。
- 先前声称成功的历史 4 秒模型没有独立历史快照，无法追溯证明与初始文件完全相同；本次另有真实重新仿真记录。
- 评估调度采用每 6 回合，不同于原 Python 的精确 550 步调度；Simulink 硬运行错误停止训练并保留检查点。
- 不声称控制器稳定性、所有异常分支或所有软件版本已被穷尽验证。

## 自动化测试记录

```text
PASS Model structural connectivity
PASS Neutral and action endpoints
PASS Real steering authority: zero/positive/negative
PASS Original executable vs MATLAB MEX numeric parity
PASS Seven MEX stages: online Np/Q/R and forced QP failure
PASS Full fixed-loop completion, QP status and repeat Reset
Reset maximum state error: 0
Pipeline vs previous fixed control maximum error: 0
```
