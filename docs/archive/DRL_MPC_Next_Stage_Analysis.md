# DRL-MPC下一阶段分析：优化目标、三维动作与TQC审计

基于64f70af，2026-11-16在本机真实CarSim 2024.1执行。研究主线保持为三自由度预测模型与增量式LTV-MPC，DRL调节MPC参数，以TQC为重点并公平比较TD3/DDPG，不直接输出转角。本轮没有训练、改奖励、改车辆/控制结构、重扫45+26组参数或启动正式多种子训练。历史模型、配置、runs和MATLAB工程均保留。

## 1. 当前三算法结果的适用范围

所有数字均为同一70km/h、mu=0.8、120m双移线、seed42的开发探索。三个算法各7900真实在线决策、相同20维观测、三维 `[Qy,Qpsi,Rdelta]`、联合奖励及选模规则。最佳合法模型已先锁定，再进行本轮回放；不依据动作屏蔽结果重新选模型。

| 控制器 | 最佳步数 | 横向RMSE m | 车身航向RMSE ° | 航迹角RMSE ° | 相对强MPC横向/航向改善 |
|---|---:|---:|---:|---:|---:|
| 原始固定MPC | — | 0.087240 | 1.014074 | 0.903569 | — |
| 强固定MPC | — | 0.078953 | 0.919092 | 0.841331 | 0 / 0 |
| DDPG最佳合法 | 2370 | 0.078177 | 0.903339 | 0.834548 | 0.9824% / 1.7140% |
| TD3最佳合法 | 6320 | 0.078778 | 0.903468 | 0.841067 | 0.2208% / 1.7000% |
| TQC最佳合法 | 4740 | 0.078770 | 0.918247 | 0.840331 | 0.2309% / 0.0920% |

三者均完赛、QP零失败，峰值Ay分别7.093791/7.092700/7.086251m/s²，峰值Beta分别2.055955/2.073225/2.077893°。TQC的严格双改善幅度很小，不能赋予统计显著性。没有单一算法在两项RMSE上支配另外两者，也没有多种子/跨工况排名证据。

| 最终模型 | 横向RMSE m | 航向RMSE ° | 强MPC双改善 |
|---|---:|---:|---|
| DDPG | 0.079202 | 0.899322 | 否 |
| TD3 | 0.079513 | 0.887651 | 否 |
| TQC | 0.079435 | 0.907356 | 否 |

三个最终模型均出现横向退化，不能替代验证选出的检查点。现有1D/2D结果继续作为动作消融，历史3D `[Np,Qy,Qpsi]` 和受限4D均保留。

冻结文件：`runs/next_stage_audit_20261116_v1/model_lock.json`，记录最佳模型SHA256、配置全继承链SHA256、控制契约、工况全内容哈希、选择来源和桥接/控制器二进制哈希。便于版本管理的精简锁见 [locks.json](../archive/next_stage_20261116_locks.json)。最佳模型SHA256为：

```text
DDPG b5fe60505e9adbdb0de3bca64e7dd9859823dd58b64ff0db0e0905b9f9a5c095
TD3  ea445194ca595e01cadabceae8144a249b88f1805ae344e18931c38fd672474b
TQC  7d2cb8b2878c0da553890ace245d2d5c50dbef0cdfbbf0b68150f96925811cf4
```

## 2. MPC航迹角与车身航向目标

定义 `wrap(a)=atan2(sin(a),cos(a))`，所有控制内部角度采用rad：

\[
e_y=Y-Y_{ref}(X),\quad e_{heading}=wrap(\psi-\psi_{ref}),\quad
\beta=atan2(v_y,v_x),\quad e_{course}=wrap(\psi+\beta-\psi_{ref}).
\]

车身航向是车体纵轴方向，航迹角是速度方向；横摆角速度误差另为 `r-r_ref`，不是航向误差。这里ey是全局横向坐标差，不是道路法向投影距离。

`configs/default.ini` 的 `weights.track_course=true`。`src/mpc.cpp` 的 `course_offset` 和 `error_offset[1]` 使Qpsi惩罚预测航迹角误差；false时仅惩罚预测车身航向误差。该项还经过局部线性化用于QP。Qy以m²、Qpsi以rad²计，不能直接比较权重数字大小。

`src/rl.cpp:jointRewardTerms` 单独计算 `w_heading*(e_heading/0.035)^2` 和 `w_course*(e_course/0.035)^2`，采用配置权重1和0.05（有裁剪）；横向以0.2m归一化。Beta优先来自CarSim实测，MPC预测Beta来自atan2(vy,vx)。故控制器与奖励对车身航向的强调程度不同；这是已确认的目标差异，尚不能单独解释所有策略表现。本轮三算法各7900条训练决策的奖励分项求和与 `reward_total_loss` 最大残差均为0。

只新增 `mpc_heading_original.ini` 和 `mpc_heading_strong.ini`，分别继承原始/强MPC，**仅覆盖track_course=false**。本轮同一冻结工况的真实固定MPC回放：

| 参数基线 | track_course | 横向RMSE m | 航向RMSE ° | 航迹角RMSE ° | 峰值Ay m/s² | 峰值Beta ° |
|---|---|---:|---:|---:|---:|---:|
| 原始 | true | 0.087240 | 1.014074 | 0.903569 | 7.070283 | 2.074793 |
| 原始 | false | 0.087306 | 1.016833 | 0.902705 | 7.072406 | 2.094699 |
| 强 | true | 0.078953 | 0.919092 | 0.841331 | 7.086899 | 2.079141 |
| 强 | false | 0.079961 | 0.903784 | 0.849134 | 7.092599 | 2.038114 |

全部完赛、QP零失败。强参数的body目标改善航向却损失横向和航迹角；原始参数下连航向也未改善，因此不能认定切换目标必然有利。旧基线和策略继续使用true；新目标若要研究必须独立训练并重新建立配套固定基线。已加强 `check_model_contract`，即使动作/观测维数相同，也拒绝与冻结控制/奖励/观测配置不同的命名策略回放，包括track_course改变；旧prefix策略兼容路径保留。

## 3. 三维权重动作有效性

在每个已冻结最佳合法模型上执行8组真实闭环动作屏蔽，位序明确为 **Qy,Qpsi,Rdelta**，0为强MPC中性动作、1为当前车辆状态下的策略动作。未重训，不是8个独立算法；屏蔽后车辆状态与其他反馈动作也可能改变。每算法000复现强MPC，111复现其最佳模型（RMSE差异低于1e-9）。24组均完赛且QP零失败。

| 算法 | 000 | 001：仅Rdelta | 110：Qy+Qpsi | 111：全三维 |
|---|---|---|---|---|
| DDPG | .078953 / .919092 | .078963 / .919015 | .077463 / .907132 | .078177 / .903339 |
| TD3 | .078953 / .919092 | .078989 / .919001 | .079202 / .898420 | .078778 / .903468 |
| TQC | .078953 / .919092 | .078985 / .919197 | .078803 / .918350 | .078770 / .918247 |

每格为横向m / 车身航向°；全部24组含峰值、安全指标和轨迹来源见 [masks.csv](../archive/next_stage_20261116_masks.csv)。

仅Rdelta相对000的横向差值仅约1.0e-5/3.6e-5/3.2e-5m，对齐X后的转角指令差RMS为0.004233/0.003463/0.001907°，支持当前范围内直接作用较弱。完整策略开启Rdelta后，DDPG横向变差0.000714m、航向改善0.003792°；TD3横向改善0.000424m、航向变差0.005048°；TQC两项各小幅改善。不能把这些交互效果当成Rdelta对固定Qy/Qpsi控制器的独立灵敏度：DDPG/TD3的Qy执行轨迹也随干预发生约1768/1408权重单位的RMS变化。

最佳合法策略每决策执行参数变化超过1%的比例：

| 算法 | Qy / Qpsi / Rdelta变化比例 | 目标未达比例Qy / Qpsi / Rdelta | 归一化请求饱和比例Qy / Qpsi / Rdelta |
|---|---|---|---|
| DDPG | 92.3% / 91.0% / 92.3% | 60.8% / 63.3% / 57.0% | 0 / 0 / 3.8% |
| TD3 | 100% / 96.2% / 92.3% | 64.6% / 84.8% / 45.6% | 5.1% / 8.9% / 10.1% |
| TQC | 5.1% / 7.7% / 1.3% | 0 / 0 / 0 | 0 / 0 / 0 |

统计按0.08s决策去重，饱和定义|a|≥0.99；执行边界统计使用端点0.1%容差，不能与动作饱和混同。[actions.csv](../archive/next_stage_20261116_actions.csv)给出最佳与最终模型的全部目标/执行范围、边界比例及相对滞后。DDPG/TD3达到1.08限速，TD3的Qpsi平均相对目标差16.0%；TQC全部权重即时达到小幅目标。前两者确有参数抖动/到达滞后，但转向率RMS仅9.091/8.859/8.373°/s，不能将调参频率直接当成车辆失稳。

预测QP代价均值中，Rdelta项8.58/9.56/9.30，占全部已记录项0.035%/0.039%/0.038%；Rdelta_change项297.45/286.09/273.31，Rsteer项480.35/477.87/468.60。Qy项约4949/4999/4852，Qpsi的course项约102/123/139，占0.42%–0.57%。这是**真实预测误差量纲下的代价贡献**，含轮胎预测及不能独立消除的误差；不是梯度贡献或算法重要性评分，不能仅据占比判断无效。

代码中Rdelta惩罚转角增量平方；Rdelta_change惩罚相邻增量差平方；Rsteer惩罚与曲率前馈轮角的偏差。后两者与轮胎软代价、轮角边界共同影响控制变化，可能削弱Rdelta可辨识性；不把轮胎软代价误称为硬约束。仅Rdelta回放提供了弱作用实证，但全策略有交互，不删除该维度，也不放大代价来制造效果。主线保留完整三维；1D/2D作对照，4D保留Np每决策限幅2。尚不能仅凭当前数据证明前馈反馈冲突或约束饱和是主要原因。

## 4. 独立真实工况：当前缺失，未进行跨工况回放

本轮用现有 `scenario_sets.describe_case` 读取项目、旧C++工作区及CarSim数据目录，共检索10048份Simfile、334份不同Parsfile内容、214种当前语义签名，16份源引用不可读并逐项记录。未找到所需65/.8、75/.8、70/.6且同时符合当前车辆/轮胎、I/O和参考轨迹的新工况。历史同名速度或附着工况存在，但车辆、I/O或轨迹不匹配；不得改名后使用。明细位于本轮 `scenario_discovery.json`、`scenario_inventory.csv`。

现有集合仍只有训练70/.8，validation/test为空。`runs/next_stage_current_preflight_20261116_v1` 新回放原始/强MPC均完赛且安全、QP零失败，`baseline_proof.json.ready=false`。没有创建伪造Simfile，也没有对不兼容工况运行策略。**真实跨工况指标、泛化结论和TQC截断消融均未执行。**

预先登记 [工况请求单](https://github.com/FCM0528/DRL_MPC/blob/e83f7fb/configs/next_stage_scenario_request.json)：65/.8和75/.8归开发验证，70/.6归最终保留测试。它是待办设计JSON，包含null路径，不能作为可运行工况集合。当前训练源Parsfile SHA256：`bb878204652e1351e0fa30e7c6c66654ff61539475e4d06846acbacfc198be7e`。

CarSim操作与验收顺序：

1. 在CarSim Browser复制已验证的Run Control `Baseline`；复制其Procedure `steer001`及需要修改的道路附着子数据集，给三个副本独立名称。编辑副本，不修改原始或共享子数据集。车辆Assembly、轮胎、转向系统、初值、驾驶速度控制方式和 `DLC SYX` 路径链接保持原样。
2. 在副本Procedure的恒定目标速度字段分别设置65/75/70km/h，生成文件应唯一包含对应 `SPEED_TARGET_CONSTANT`。道路常附着分别0.8/0.8/0.6，生成文件核对 `MU_ROAD_CONSTANT`；保持原横向附着设置、道路几何和LTARG表，不通过改文件名制造差异。
3. 保持两个导入 `IMP_STEER_L1,IMP_STEER_R1`，顺序不变、Replace、deg；20导出依次为 `Xcg_TM,Ycg_TM,Vx,Vy,Yaw,AVz,Ax,Ay,Steer_L1,Steer_R1,Beta,Rho_Road,AlphL_L1,AlphL_R1,AlphL_L2,AlphL_R2,Fy_L1,Fy_R1,Fy_L2,Fy_R2`。核对原生生成的PORTS_IMP/EXP和echo；不手工拼接Simfile/Run_all.par，不变更转向模式以便运行。
4. 用Browser原生生成完整Run_all.par及其Simfile，将每对文件保存在独立目录，保留DATADIR/PROGDIR和源数据集出处；只用现有snapshot工具冻结。求解器应为当前可用CarSim 2024.1 DLL，工况自身应支持运行至120m以后。桥接继续按SI读取位置/速度/角度/加速度，不能把导出显示的km/h、deg、g当SI状态；转向导入保留57.2957795增益。
5. 用 `describe_case` 比较车辆/轮胎内容哈希和I/O哈希完全相等，参考numeric哈希、曲率和覆盖区间一致；车速/附着应产生真实不同签名。完整Simfile/Parsfile哈希也保存。该比较是可读数据块的契约检查，不能替代真实求解器的单位、I/O及闭环验收。
6. 建立真实路径集合JSON：train=[已验证70/.8]，validation=[新65/.8,新75/.8]，test=[新70/.6]。先运行 `scenario_preflight.py`，核对原始和强MPC逐工况完赛、QP=0、峰值Ay≤0.85g、|Beta|≤5°及echo的2入20出；不合格就记录阻塞，不以调低基线掩盖问题。mu=.6不保证可安全完成当前双移线。
7. 只在两项开发验证工况用已经锁定的三个模型回放；不重训或重新选择静态参数。最终测试只允许预先固定模型的最终报告阶段使用，预检测试基线的安全性不等于允许用它调参。

现有 `evaluate_dual.py` 新增 `--split validation`，默认仍为test；可直接读取本轮锁字典或旧JSON列表，验证模型/配置链哈希、车辆I/O及物理泄漏，并保存双基线改善、峰值、安全、转向率、MPC和推理时延。合法性与双改善单独记录。真实新集合未具备，因此该新路径仅完成单元测试与缺失集合阻塞检查，未声称通过真实跨工况集成验收。

## 5. TQC训练专项审计

实际保存的practical模型：3 critics×25 quantiles，drop=2/critic，目标池75个删去最高6个保留69个；actor按全部分位数均值更新。target_entropy=auto实际解析为−3，alpha从约0.992降至最终0.479669。Actor/Critic/熵优化器学习率实测1e-4/3e-4/1e-4，无被共享学习率覆盖的证据。train_freq=1、gradient_steps=1、warmup550，最终7350次更新；周期回调发生在当次训练更新前，4740步模型保存4189次更新，末次CSV日志7349次，而final模型7350次，不是少训一步。

| 冻结模型探针（同一批真实最佳回放观测） | 最佳4740 | 最终7900 |
|---|---:|---:|
| alpha | 0.657792 | 0.479669 |
| deterministic动作饱和 | 0 | 0 |
| log_std均值（tanh前） | -0.149318 | -0.120722 |
| 固定seed分布采样饱和比例 | 0.1055% | 0.1714% |
| 采样微分熵均值 | 2.034938 | 2.029672 |
| alpha×采样熵 | 1.338566 | 0.973572 |
| 当前critic全部分位数均值 | 18.519833 | 17.932696 |
| 删最高6个后的均值 | 18.294881 | 17.646668 |
| 平均单状态分位数跨度 | 9.644056 | 6.947903 |

以上对实际记录的79个20维观测离线无更新探针，分布采样仅用于检查策略分布，**不替代CarSim状态或训练奖励**。完整75个分位数/状态保存于 `analysis/tqc_replay_quantiles.csv`，各周期模型探针见 [tqc.csv](../archive/next_stage_20261116_tqc.csv)。这些是冻结critic对共同输入的估计，不是训练时历史分位数分布、真实回报分位数或无偏过估计证据。历史经验池未保存，不能重建训练目标分布；存在约3%–6%邻接分位数交叉，不能据此单独断言训练失效。

实际7900次训练决策中三维动作饱和比例约0.316%/0.304%/0.405%，没有动作卡边界证据。最佳确定性Qy仅18080–19281，Qpsi8396–9122，Rdelta10448–10886。训练记录的actor loss从−3.687到−18.322，critic loss从0.0659到0.4969；软价值尺度在变化，loss的单调性不等于物理误差收敛。环境回报最后8回合均值较前8回合从−33.294提升至−32.895，但确定性最终模型相对最佳仍横向退化；两项RMSE必须共同检查。

最佳/最终每决策环境回报约−0.417/−0.415，探针熵项约1.339/0.974；critic对单维动作±0.2的平均估计值跨度仅约0.03–0.06。结合小幅确定性动作，**有限预算内探索熵与弱物理动作信号可能限制学习**，目前只是受数据支持的假设，不是已确定因果。正软Q值含未来熵贡献，不能与负环境回报直接比较后认定过估计。没有证据支持必须调整截断数、加网络或强行提高TQC排名。

待真实开发验证工况通过后，才允许单因素drop=0/2/3，各自同7900交互预算、同seed42、同配置与选模规则、从空经验池重训；不得同时改奖励、学习率或动作范围。此轮未执行。

## 6. 文件、数据与复现

核心修改为两份独立目标配置、待生成工况请求单、模型契约防误用、复用动作屏蔽工具支持三算法及动作名、回放保存真实观测、既有双基线评价增加开发验证分集，以及 `mainline_audit.py` 统一调用现有工具进行冻结诊断；未新增训练框架。TQC设计和接口文档已更新20维与多profile，历史结果段保留。

本轮最佳模型回放的MPC计算均值/p95约DDPG 0.220/0.342ms、TD3 0.237/0.353ms、TQC 0.252/0.375ms；策略推理分别0.434/0.562ms、0.501/0.617ms、0.755/0.985ms。MPC计时包含该次控制求解，策略计时仅含CPU `predict`，不含CarSim积分、IPC、进程启动或日志；本机测量不是硬实时保证，也不用于算法性能排名。

本轮真实输出 `runs/next_stage_audit_20261116_v1/`：四组目标回放、24组动作屏蔽、三算法最佳/最终六组完整回放、TQC真实观测、模型锁和工况检索；`analysis_final/` 保存黑白打印友好曲线，Times New Roman、线型区分，另有PDF；`analysis/` 保存TQC探针CSV。精简CSV在 `docs/archive/next_stage_20261116_*`，所有旧数据仍在原处。

- `analysis_final/best_tracking_and_weights.png`：ey、航向和三权重随X。
- `analysis_final/requests_targets_execution.png`：归一化请求、目标、实际执行三者分离。
- `analysis_final/training_and_validation.png`：三算法回报、loss、周期双指标与Pareto。
- `analysis_final/tqc_frozen_critic_probes.png`：alpha、熵、分位数及critic动作响应。

下列命令在项目根目录运行；每次使用未存在的输出目录。已有环境可用 `uv run --no-project --with-requirements python/requirements.txt python ...` 替代python：

```powershell
cmake --build out/python-rl-build --config Release --parallel 4
ctest --test-dir out/python-rl-build -C Release --output-on-failure
uv run --no-project --with-requirements python/requirements.txt --with pytest python -m pytest python/tests -q -p no:cacheprovider

# 完整同工况诊断，不训练；$d/$t/$q填写各原实验子目录（含manifest）。
$d = (Get-ChildItem runs/mainline_weights3_ddpg_20261116_v1 -Directory)[0].FullName
$t = (Get-ChildItem runs/mainline_weights3_td3_20261116_v1 -Directory)[0].FullName
$q = (Get-ChildItem runs/mainline_weights3_tqc_20261116_v1 -Directory)[0].FullName
python python/mainline_audit.py --ddpg-run $d --td3-run $t --tqc-run $q --out runs/next_stage_NEW --search-root . --search-root 'D:/Program Files (x86)/CarSim2024_Date'

# 单独零动作全程检查；默认读取冻结70km/h源工况。
python python/parity.py --config configs/strong_weights3.ini --blocks 200 --out runs/weights3_parity_NEW
python python/parity.py --config configs/strong_full4.ini --blocks 200 --out runs/full4_parity_NEW

# 仅在Browser真实导出并创建该集合JSON之后；不可使用设计请求单代替集合。
python python/scenario_preflight.py --scenario-set configs/real_cases.local.json --out runs/real_preflight_NEW
python python/evaluate_dual.py --lock docs/archive/next_stage_20261116_locks.json --scenario-set configs/real_cases.local.json --split validation --out runs/frozen_dev_NEW

# 最终协议锁定后的保留测试入口；本轮不执行。
python python/evaluate_dual.py --lock docs/archive/next_stage_20261116_locks.json --scenario-set configs/real_cases.local.json --split test --out runs/frozen_test_NEW

# 正式计划预检，不附--execute，不启动15组训练。
python python/formal_compare.py --profile weights3 --scenario-set configs/real_cases.local.json --total-decisions 7900 --out runs/formal_precheck_NEW
```

必要检查：Release构建、CTest 1/1、pytest 60项；weights3/full4真实零动作全程各157控制步通过。四组目标与24组屏蔽全部完赛/QP=0，六组模型回放复现历史指标。未运行：缺失的三个新工况及其双基线/跨工况策略回放、drop消融、多种子正式实验。尚不具备正式多种子比较条件；下一步最小工作是生成并核验上述三份原生工况，先双基线安全预检，再冻结模型开发回放，依据实际结果决定TQC单因素消融。不能把同工况小幅优势或失败扩展为泛化与算法结构结论。
