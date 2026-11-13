from pathlib import Path
import csv,json,datetime,hashlib,math,configparser
root=Path(__file__).resolve().parents[1]
r=root/'results'
cfg=configparser.ConfigParser();cfg.read(root/'configs'/'default.ini')
if (cfg.getint('mpc','np'),cfg.getint('mpc','nc'))!=(47,7):
    raise SystemExit('Historical 47/7 report preserved. Current revision: results/CURRENT_CONFIGURATION.md')
def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))
audit=json.loads((r/'source_audit.json').read_text())
model=json.loads((r/'model_validation.json').read_text())
comparison=rows(r/'comparison'/'metrics.csv')
tests=(r/'test_results.txt').read_text()
assert model['compiled'] and model['structure_healthy']
assert tests.count('PASS')==6
text=['# 实际迁移验收记录','',f'生成时间：{datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")}','',
      '以下结果来自实际本机执行。MCP 初始会话 PID 38724 完成接口读写与仿真；该会话退出后，继续通过本机 MATLAB COM 执行。',
      '',f'最终 MATLAB PID：{model["matlab_pid"]}；版本：{model["matlab_version"]}。','',
      '## 功能与证据','',
      '| 项目 | 实际状态 | 证据 |','| --- | --- | --- |',
      '| 模块化模型 | 已创建、保存、编译，结构检查通过 | model_validation.json、model_structure.txt、model_overview.png |',
      '| 20 路状态与单位转换 | 已在真实 CarSim 联合仿真中验证 | interface_units.txt、各组 trace.csv |',
      '| 两路前轮控制权限 | 零、正、负指令与两侧反馈已验证；零指令覆盖 DLC 区段 | steering_authority.csv、steering_*.mat |',
      f'| CarSim 原始目标轨迹 | 完整 {audit["reference_rows"]} 行，源与副本一致 | source_audit.json |',
      '| 固定 LTV-MPC | 完成真实闭环，QP 状态与终点见下表 | fixed/raw.mat、fixed/trace.csv |',
      '| 原 C++ 与 MEX | 首步控制、QP、预测状态、观测、参考、奖励数值对照通过 | native_parity.csv |',
      '| 七阶段 MPC 在线调参 | Np 与所有权重候选回放通过 | adaptive_pipeline_parity.json、mex_pipeline_parity.csv |',
      '| 重复 Reset | 完整固定闭环重复运行测试通过 | reset_validation.mat、test_results.txt |','',
      '## 正式训练','', '| Agent | 完成回合 | 策略交互 | Actor 最大参数变化 |','| --- | ---: | ---: | ---: |']
for algorithm in ['ddpg','td3']:
    summary=json.loads((r/f'training_{algorithm}_seed42'/'summary.json').read_text())
    assert summary['completed_episodes']==500 and summary['actor_max_parameter_change']>0
    text.append(f'| {algorithm.upper()} | {summary["completed_episodes"]} | {summary["interactions"]} | {summary["actor_max_parameter_change"]:.8g} |')
text+=['','## 同工况确定性对照','', '| 控制器 | 横向 RMSE (m) | 横向峰值 (m) | 航向 RMSE (rad) | QP 失败 | 完成 | 实际末点 X (m) |', '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
for x in comparison:
    text.append(f'| {x["controller"]} | {float(x["lateral_rmse_m"]):.6f} | {float(x["max_abs_lateral_m"]):.6f} | {float(x["heading_rmse_rad"]):.6f} | {x["qp_failures"]} | {x["completed"]} | {float(x["end_x_m"]):.4f} |')
effort=[]
text+=['','## 控制动作代价','', '| 控制器 | 最大转角 (deg) | 最大转向速率 (deg/s) | 最大侧偏角 (deg) | Np 切换次数 |','| --- | ---: | ---: | ---: | ---: |']
for label in ['fixed','ddpg','td3']:
    trace=rows(r/label/'trace.csv')
    t=[float(x['time_s']) for x in trace];steer=[float(x['command_rad']) for x in trace]
    e=dict(controller=label,max_steer_deg=math.degrees(max(map(abs,steer))),
           max_steer_rate_deg_s=math.degrees(max(abs((steer[k]-steer[k-1])/(t[k]-t[k-1])) for k in range(1,len(t)))),
           max_sideslip_deg=math.degrees(max(abs(float(x['Beta_rad'])) for x in trace)),
           np_changes=sum(trace[k]['Np']!=trace[k-1]['Np'] for k in range(1,len(trace))))
    effort.append(e)
    text.append(f'| {label} | {e["max_steer_deg"]:.3f} | {e["max_steer_rate_deg_s"]:.3f} | {e["max_sideslip_deg"]:.3f} | {e["np_changes"]} |')
with (r/'comparison'/'control_effort.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(effort[0]));w.writeheader();w.writerows(effort)
text+=['','本次两种 RL 策略降低了跟踪误差，同时转向峰值和速率高于固定 MPC，且存在 Np 频繁切换。保留原动作映射与代价配置，未为美化曲线增加额外平滑；这些结果不构成整体控制品质均优的结论。']
text+=['','训练完成不等于策略优于固定 MPC；上表是所保存最终 Agent 的实际回放结果。全部使用同一 CarSim 工况、相同初始状态和质心 150 m 的终止判据。策略采样导致末点略超过 150 m，表中保留实际值。','',
       '## 适用范围与未验证项','',
       '- 已验证本机 R2025a / CarSim 2024.1 / 当前车辆 / DLC SYX / 70 km/h / μ=0.8；未据此证明跨车辆、跨工况或多随机种子的泛化。',
       '- TQC 未实现。其他随机种子提供入口，本次正式实验为 seed=42。',
       '- 先前声称成功的历史 4 秒模型没有独立历史快照，无法追溯证明与初始文件完全相同；本次另有真实重新仿真记录。',
       '- 评估调度采用每 6 回合，不同于原 Python 的精确 550 步调度；Simulink 硬运行错误停止训练并保留检查点。',
       '- 不声称控制器稳定性、所有异常分支或所有软件版本已被穷尽验证。','',
       '## 自动化测试记录','', '```text',tests.strip(),'```','']
(r/'VALIDATION.md').write_text('\n'.join(text),encoding='utf-8')
manifest={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'models'/'DRL_MPC.slx',root/'build'/'mpc_mex.mexw64',r/'comparison'/'metrics.csv']}
(r/'delivery_sha256.json').write_text(json.dumps(manifest,indent=2))
print('\n'.join(text))
