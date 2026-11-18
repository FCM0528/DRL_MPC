# DRL 设计、观测、动作与奖励

控制器和 reward 以 src/rl.cpp、src/rl.h、src/rl_bridge_main.cpp 为准；配置采用 strong_weights3.ini 的继承链。算法实现见 python/algorithms.py 与 train.py。本轮不改变任何计算。

## 算法与采样

DDPG 使用确定性 Actor 和单评论家；TD3 使用双评论家、延迟 Actor 更新及目标策略平滑；TQC 使用随机 Actor、熵正则与分布评论家，丢弃目标分位数的上尾以抑制乐观偏差。三者仅调整 MPC 参数，均从自己的空经验池采集真实 CarSim transition，不共享经验、不使用固定控制标签。

当前 pilot 每算法 7900 decision，seed42，learning_starts=550、batch_size=256、每 decision 一次更新；Actor/Critic 及熵学习率独立实际应用。网络和全部算法参数见 rl_training_pilot.ini。TQC practical 为 3 个评论家×25 分位数，drop=2/critic，目标池保留 75−6=69 个分位数；Actor 使用全体估计均值，熵系数 auto、target_entropy=−动作维数。其 Actor 目标含 α log π−Q，与环境 reward、MPC Q/R 和安全失败惩罚不同，不能直接比较不同算法 loss 的绝对量级。

## 20维 road_preview_joint（严格顺序）

定义 C_b(z)=clip(z,−b,b)，非负量按下表对应范围 clip；预瞄计算来自真实 LTARG、路面表和当前车速，不能读取未来实际车辆状态。

|序号|量与归一化|范围|
|---:|---|---|
|1|hypot(vx,vy)/30|[0,2]|
|2|当前曲率×25|[−1,1]|
|3|预瞄曲率变化×25|[−1,1]|
|4|曲率变化率×100|[0,1]|
|5|预瞄航向变化/0.30|[−1,1]|
|6|当前 μ/1.5|[0,1]|
|7|峰值附着需求/grip_demand_clip|[0,1]|
|8|需求到达相对位置|[0,1]|
|9|asinh(ey/ey_scale_m)|[−4,4]|
|10|asinh(e_course/epsi_scale_rad)|[−4,4]|
|11|上一转角/max_steer|[−1,1]|
|12|2(Np−min_np)/(max_np−min_np)−1|[−1,1]|
|13|预瞄最小 μ/1.5|[0,1]|
|14|近端附着需求/grip_demand_clip|[0,1]|
|15|车身航向误差/heading_scale_rad|[−4,4]|
|16|beta/sideslip_threshold_rad|[−4,4]|
|17|(r−vx·curvature)/0.25|[−4,4]|
|18–20|当前 Qy、Qpsi、Rdelta 的 log(weight/center)/最大单侧 log span|[−1,1]|

旧12/14维策略保留原版本契约，不加载为20维。精确预瞄、峰值需求定义见 makeRoadPreviewObservation，不能将表中量推断为附加传感器。

## 动作 profile

|维数|配置|策略输出|
|---:|---|---|
|1|--action-profile qy1|Qy|
|2|--action-profile weights2|Qy,Qpsi|
|3 主线|strong_weights3.ini|Qy,Qpsi,Rdelta|
|4 完整|strong_full4.ini|Np,Qy,Qpsi,Rdelta|

strong_actions3.ini 的历史3维是 Np,Qy,Qpsi，不能与主线3维混用。桥接固定四维 [Np,Qy,Qpsi,Rdelta]；未学习量填0。Python action_mapping.py 按名称映射，模型 manifest 校验名称、维数、版本和配置签名。

连续权重映射为 w*(wmax/w*)^a（a≥0），w*(w*/wmin)^a（a<0），中性动作严格对应强 MPC：36/18400/8960/10800。Qy 范围11500–23920，Qpsi4480–13440，Rdelta7560–14040；每决策权重最多乘/除1.08。Np 离散到28–44，full4每决策最多改变2；Nc=13固定。legacy五档保持旧义，不能复用其模型为连续动作模型。

## 完整 joint_tracking 环境奖励

令 ey=Y−Yref(X)，h=wrap(psi−psi_ref)，c=wrap(h+beta)，er=r−vx·curvature；角度均为 rad。ey 是全局 Y 差，不是道路法向距离。

记 s_y=0.2 m，s_h=0.035 rad，p=0.25 m，τ=0.25 s；控制周期 dt=0.04 s。各控制步损失为：

\[
L_y=w_y C_{20}(e_y/s_y)^2,\quad L_h=w_h C_{20}(h/s_h)^2,\quad L_c=w_c C_{20}(c/s_h)^2,
\]
\[
L_p=w_p[\max(0,|e_y|-p)/p]^2,\quad L_g=w_g[\max(0,|e_y+\tau\dot e_y|-|e_y|)/s_y]^2,
\]
\[
\dot e_y=v_x\sin\psi+v_y\cos\psi-y'_{ref}(v_x\cos\psi-v_y\sin\psi),
\]
\[
L_\delta=w_\delta C_1(\delta/\delta_{max})^2+w_{\dot\delta}C_5((\delta-\delta_{prev})/(s_{\dot\delta}dt))^2,
\]
\[
L_{safe}=w_\beta[\max(0,|\beta|/\beta_t-1)]^2+w_r C_3(e_r/0.25)^2.
\]

对三个权重 j，定义 bj=0.5 log(wmax,j/wmin,j)，dj=C2(log(wj/wprev,j)/bj)，zj=C1(log(wj/w*,j)/bj)，n=C1(|Np−Np_prev|/(Nmax−Nmin))（首次无有效前值取0）。

\[
L_{switch}=w_{np}n^2+\frac{w_{change}}3\sum_j d_j^2,
\quad L_{other}=\frac{w_{dev}}3\sum_j z_j^2+w_N clip((Np-Nmin)/(Nmax-Nmin),0,1)+w_{vy}C_3(v_y/2)^2.
\]

L 为以上全部项之和，控制步 reward=−reward_scale·L。每个 RL 决策包含最多2个控制步：

\[
R=\frac{\sum reward_{step}}{\max(1,steps-1_{failed})\,reward\_scale}-25\,N_{QP}-100\,1_{failed}-100\,1_{incomplete\land\neg failed}.
\]

QP 项由 penalize_qp_failure 开关决定，失败优先于未完赛，两者不叠加。当前 scale=10；wy=wh=1，wc=.05，wp=3，wg=.6，wδ=.03，wδdot=.05，sδdot=100°/s（内部转rad），wβ=.2，βt=5°，wnp=.002，wchange=.001，wdev=wN=wvy=wr=0。w_yaw_error 对应横摆角速度误差，不是航向误差；w_psi 属于 legacy，不替代 joint 的 w_heading。控制步损失在 bridge_trace.csv 分项记录，决策级 QP/终止项在 decision_reward.csv。

legacy 奖励保留原函数分支，不能按上式解释旧模型。MPC 预测代价使用 Qy/Qpsi/Rdelta 等优化预测控制序列；上述 reward 评价真实闭环；TQC 熵正则是学习算法内部目标；四者必须分开。

## 验证选模

validation 不进入梯度更新，test 不参与选模。合法需完整结束、QP=0、Ay≤0.85g、|Beta|≤5°；逐工况同时计算相对原始和强 MPC 的改善。只有全部验证工况均双RMSE改善才生成 best_joint；best_legal 允许未达标，按最差工况 min(Iy,Iheading) 最大化。保留全部周期候选和 Pareto，final 只诊断。严格全部达标、平均改善、最差改善和达标工况比例分别报告。


## 生成式消融与旧契约

当前1D/2D由 train.py 在原 strong_weights3 基础上生成run私有 delta，4种profile均已与对应历史有效配置逐项比较，控制参数、观测、奖励、映射和限速等价。为公平比较，变化只限所声明动作维度；旧策略不能按新名字加载。历史 Np前缀配置从模型冻结目录或精确恢复机制读取。

不再增加 wy2/wy4、Actor学习率或噪声专用INI；使用显式单因素覆盖，并保留实际 source_training.ini、run_delta.ini 与 manifest。正文所有奖励公式、系数和MPC物理含义不因结构精简改变。TQC分位数审计函数在 algorithms.quantile_summary；旧控制/奖励审计纯函数在 evaluate；只有正式 joint_validation 实施当前全工况安全选模规则。
