# 增量式三自由度 LTV-MPC 控制器

本文描述当前代码实现。结构借鉴《基于强化学习的轻型货车自适应MPC路径跟踪控制》中增量控制、扩展状态及双线性离散化的思路；参考论文采用二维横向模型和 TD3，本项目按 CarSim 接口保留三自由度动力学，并支持 DDPG 与可选 TD3。两者均使用本文的三自由度 MPC，不能将论文控制器与本实现视为同一个控制器。

## 1. 状态、轮胎和连续动力学

以 CarSim 坐标定义状态与前轮路轮转角：

\[
z=[v_y,v_x,\psi,r,Y,X]^\mathsf T,\qquad u=\delta_f.
\]

轮胎表按静态单轮轴荷 \(F_{zf}=mgb/[2(a+b)]\)、\(F_{zr}=mga/[2(a+b)]\) 求近零侧偏斜率，再乘双轮和有效路面附着系数得到 \(C_f,C_r\)。当前工作点的线性侧向力为

\[
F_{yf}=C_f\left(u-\frac{v_y+ar}{v_x}\right),\qquad
F_{yr}=C_r\left(\frac{br-v_y}{v_x}\right).
\]

实验配置 `configs/carsim_nonlinear_experimental.ini` 还提供轮胎表局部模型。令上式括号分别为 \(\alpha_f,\alpha_r\)，从 CarSim 回显的同一轮胎表插值得到单轮侧向力 \(f_y(\alpha,F_z)\)，并令

\[
F_{yf}=2\mu f_y(\alpha_f,F_{zf}),\qquad
F_{yr}=2\mu f_y(\alpha_r,F_{zr}).
\]

各预测步在名义侧偏处估算局部斜率。为避免饱和区或表外延伸出现零、负切线导致 QP 利用局部模型反向操纵，实验配置采用
\(\widetilde C_\alpha=\max(\partial f_y/\partial\alpha,\,0.15C_{\alpha,0})\)。仿射项仍保持该工作点的表格力，故该近似在工作点处匹配力，但下限生效时 **并非真实雅可比**；静态轴荷、相同轮胎和缩放假设仍然存在。该功能是实验选项，不代表真实轮胎饱和或闭环稳定性已得到完整验证。

模型包含车体纵向、侧向和横摆三个动力学自由度：

\[
\begin{aligned}
\dot v_y&=(F_{yf}+F_{yr})/m-v_xr,\\
\dot v_x&=v_yr+(F_x-F_{yf}u)/m,\\
\dot\psi&=r,\qquad \dot r=(aF_{yf}-bF_{yr})/I_z,\\
\dot Y&=v_x\sin\psi+v_y\cos\psi,\\
\dot X&=v_x\cos\psi-v_y\sin\psi.
\end{aligned}
\]

代码没有编造无法从 CarSim 读取的纵向轮胎驱动力 \(F_x\)。在实测工作点以相邻状态差得到 \(a_x=\Delta v_x/T_s\)，并把 \(\dot v_x=a_x\) 写入该工作点的函数值。预测名义轨迹上，此短时扰动按 \(a_{x,k}=a_x\exp(-kT_s/0.5\,\mathrm s)\) 衰减，避免将瞬时制动加速度恒定外推至整个预测时域。

## 2. 每个预测步的线性化与 Tustin 离散化

在第 \(t\) 个控制周期，以上次最优转角序列左移一位，并按角度与速率限制投影，得到名义转角 \(\bar u_{0:N_c-1}\)；超过 \(N_c\) 后保持末项。以 CarSim 实测状态为 \(\bar z_0\)，沿此转角序列生成名义状态 \(\bar z_k\)。**每个预测阶段都在自己的 \((\bar z_k,\bar u_k)\) 上线性化**，所以 \(A_k,B_k,c_k\) 随 \(k\) 变化。

连续局部模型为

\[
\dot z\approx J_k z+G_k u+d_k,
\quad J_k=\left.\frac{\partial f}{\partial z}\right|_{\bar z_k,\bar u_k},
\quad G_k=\left.\frac{\partial f}{\partial u}\right|_{\bar z_k,\bar u_k},
\quad d_k=f(\bar z_k,\bar u_k)-J_k\bar z_k-G_k\bar u_k.
\]

这里 \(f\) 的纵向分量取当前阶段的 \(a_{x,k}\)。代码对上述仿射模型使用双线性离散化：

\[
M_k=I-\tfrac{T_s}{2}J_k,\qquad
A_k=M_k^{-1}(I+\tfrac{T_s}{2}J_k),\quad
B_k=M_k^{-1}T_sG_k,\quad c_k=M_k^{-1}T_sd_k.
\]

一次带部分选主元的消元同时求出 \(A_k,B_k,c_k\)。预测式为

\[
z_{k+1}=A_k z_k+B_k u_k+c_k.
\]

相较于每周期只使用一组矩阵的模型，这里沿整条名义预测轨迹更新矩阵；相较于前向 Euler，Tustin 离散化在给定采样周期下对局部线性动力学更稳健。实际车辆始终由 CarSim Solver 积分。

## 3. 增量控制与扩展状态

真正的决策变量是 \(N_c\) 个前轮转角增量

\[
U=[\Delta u_0,\ldots,\Delta u_{N_c-1}]^\mathsf T,
\qquad u_k=u_{t-1}+\sum_{j=0}^{\min(k,N_c-1)}\Delta u_j.
\]

由 \(\xi_k=[z_k^\mathsf T,u_{k-1}]^\mathsf T\) 可写出显式扩展状态方程：

\[
\xi_{k+1}=
\begin{bmatrix}A_k&B_k\\0&1\end{bmatrix}\xi_k+
\begin{bmatrix}B_k\\1\end{bmatrix}\Delta u_k+
\begin{bmatrix}c_k\\0\end{bmatrix}.
\]

实现为了避免重复存储第七维，用累计增量直接重构 \(u_k\)，与该扩展方程等价。设 \(S_{k,j}=\partial z_k/\partial\Delta u_j\)，则

\[
S_{0,j}=0,\qquad
S_{k+1,j}=A_kS_{k,j}+B_k\mathbf 1_{\{j\leq\min(k,N_c-1)\}}.
\]

所有 \(S_{k,j}\) 在当前周期用阶段相关的 \(A_k,B_k\) 递推。若 \(\bar U\) 是名义转角对应的增量，则优化中预测状态为

\[
z_k(U)\approx\bar z_k+\sum_j S_{k,j}(\Delta u_j-\Delta\bar u_j).
\]

## 4. 跟踪误差、目标函数和硬约束

CarSim 的 `LTARG_TABLE` 在当前支持的 Straight East 道路上给出 \(Y_{\rm ref}(X)\)。自然三次样条产生 \(Y'_{\rm ref}\)、\(\psi_{\rm ref}=\arctan Y'_{\rm ref}\) 和 \(\kappa_{\rm ref}=Y''_{\rm ref}/[1+(Y'_{\rm ref})^2]^{3/2}\)。在名义预测点定义

\[
e_k=\begin{bmatrix}
Y_k-Y_{\rm ref}(X_k)\\
\operatorname{wrap}(\psi_k-\psi_{\rm ref}(X_k))\\
v_{y,k}\\r_k-v_{x,k}\kappa_k
\end{bmatrix},
\qquad
\kappa_k=\operatorname{clip}\left(\kappa_{\rm ref}(\bar X_k),
\pm\frac{a_{y,\max}}{\max(1,\bar v_{x,k}^{,2})}\right).
\]

曲率上限由本次轮胎表估计。轨迹误差在名义预测点线性化为 \(e_k(U)\approx\bar e_k+E_k(U-\bar U)\)。例如

\[
\frac{\partial e_{y,k}}{\partial\Delta u_j}=
S_{Y,k,j}-Y'_{\rm ref}(\bar X_k)S_{X,k,j}.
\]

令 \(Q=\operatorname{diag}(q_y,q_\psi,q_{v_y},q_r)\)，\(\gamma_k=1\)（末端步为 `terminal_scale`）。根据名义速度及曲率求受限的前馈转角 \(u_{\rm ff,k}\)。QP 目标为

\[
\min_U\;\sum_{k=1}^{N_p}\gamma_k e_k(U)^\mathsf TQe_k(U)
+r_{\rm steer}\sum_{k=1}^{N_p}(u_k(U)-u_{\rm ff,k})^2
+r_\Delta\sum_{j=0}^{N_c-1}(\Delta u_j)^2.
\]

相较于旧控制器，`r_delta` 现在**直接作用于优化变量**；`r_steer` 则作用于由增量前缀和重构的转角。优化展开后为 \(\tfrac12 U^\mathsf THU+g^\mathsf TU\)。硬约束为

\[
|u_{t-1}+\sum_{j=0}^{i}\Delta u_j|\leq u_{\max},\qquad
|\Delta u_i|\leq T_s\dot u_{\max}.
\]

因此角度约束的矩阵块是下三角全 1 矩阵，速率约束块是单位矩阵。ADMM 分裂 \(v=GU\) 后交替求解二次子问题、投影到约束区间并更新对偶变量。收敛后只执行第一项 \(u_t=u_{t-1}+\Delta u_0\)，下一周期重新读取 CarSim 状态并滚动求解。

## 5. DDPG 与安全分支

DDPG 或可选 TD3 每 `decision_steps` 周期输出整数 \(N_p\) 及六个绝对 Q/R 数值；\(N_c,T_s\)、转角幅值、速率约束和求解器设置继续由共享 MPC 配置给出。旧低速运动学模型已删除，所以策略动作不再按旧模型融合速度门控。DDPG 策略文件标记为 `DDPG_MPC_V16_INCREMENTAL_LTV`，TD3 标记为 `TD3_MPC_V17_INCREMENTAL_LTV`；旧 V14/V15 网络针对不同控制问题训练，被加载器拒绝。

若当前参考曲率超出估计附着能力、跟踪偏差与侧向速度同时较大、名义预测失效或 QP 未收敛，则暂用受附着和转角约束的前视跟踪控制。车速低于 1 m/s 时三自由度小侧偏线性模型不再适用，也走此分支；该分支是应急控制，不是第二套 MPC 预测模型。CarSim 车辆动力学从未被替换。

实验配置选择按实测误差和侧滑触发保护，而不是仅凭参考曲率触发；仍保留求解异常时的保护分支。由于预测模型与保护逻辑改变，旧 DDPG/TD3 策略仅可作为再训练的 Actor 初始值，不能将旧工况验证结论直接转用于实验模型。

主要实现：`src/vehicle.cpp::discretizeErrorModel()`、`src/mpc.cpp::LateralMpc::precompute()` / `control()`、`src/rl.cpp::DdpgPolicy`、`src/main.cpp::runConfiguredSimulation()`。
