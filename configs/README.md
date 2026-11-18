# 配置入口与兼容约定

所有既有 INI 保留原字节、路径和继承链，原始/强 MPC 不调参。未学习维度填中性动作；同维不同动作名称不兼容。训练设置不是控制器设置。

|配置|继承|动作/用途|算法与使用场景|
|---|---|---|---|
|carsim_curvature_schedule_40ms.ini|default.ini|固定控制器（历史目标/调度消融）|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl.ini|default.ini|np,q_y,q_psi,r_delta|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint.ini|default.ini|np,q_y,q_psi,r_delta|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint_actions2.ini|default.ini|np,q_y|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint_actions3.ini|default.ini|np,q_y,q_psi|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint_actions4.ini|default.ini|np,q_y,q_psi,r_delta|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint_continuous14.ini|default.ini|np,q_y,q_psi,r_delta|保留历史消融/旧CLI；按原模型使用|
|carsim_python_rl_joint_legacy14.ini|default.ini|np,q_y,q_psi,r_delta|保留历史消融/旧CLI；按原模型使用|
|default.ini|—|固定控制器|当前入口/共同基础|
|mpc_heading_original.ini|default.ini|固定控制器（历史目标/调度消融）|保留历史消融/旧CLI；按原模型使用|
|mpc_heading_strong.ini|mpc_strong.ini|固定控制器（历史目标/调度消融）|保留历史消融/旧CLI；按原模型使用|
|mpc_strong.ini|default.ini|固定控制器|当前入口/共同基础|
|rl_training.ini|—|学习率/网络/预算；非车辆控制器|当前入口/共同基础|
|rl_training_pilot.ini|—|学习率/网络/预算；非车辆控制器|当前入口/共同基础|
|strong_actions2.ini|mpc_strong.ini|np,q_y|保留历史消融/旧CLI；按原模型使用|
|strong_actions3.ini|mpc_strong.ini|np,q_y,q_psi|保留历史消融/旧CLI；按原模型使用|
|strong_actions3_wy2.ini|mpc_strong.ini|np,q_y,q_psi|保留历史消融/旧CLI；按原模型使用|
|strong_actions3_wy4.ini|mpc_strong.ini|np,q_y,q_psi|保留历史消融/旧CLI；按原模型使用|
|strong_actions4.ini|mpc_strong.ini|np,q_y,q_psi,r_delta|当前入口/共同基础|
|strong_full4.ini|strong_actions4.ini|np,q_y,q_psi,r_delta|当前入口/共同基础|
|strong_qy_only.ini|strong_actions3.ini|q_y|当前入口/共同基础|
|strong_qy_qpsi.ini|strong_actions3.ini|q_y,q_psi|当前入口/共同基础|
|strong_weights3.ini|strong_actions4.ini|q_y,q_psi,r_delta|当前入口/共同基础|
|td3_qy_actor_lr1e4.ini|—|学习率/网络/预算；非车辆控制器|保留历史消融/旧CLI；按原模型使用|
|td3_qy_sigma010.ini|—|学习率/网络/预算；非车辆控制器|保留历史消融/旧CLI；按原模型使用|

magic_tire.csv 为保留轮胎表；real_scenario_sets.local.json 为旧路径列表兼容入口；新训练优先 datasets/carsim/manifests/dataset_v1.json。所有 DRL profile 支持三算法，主线3D是 Qy/Qpsi/Rdelta；strong_actions3 的 Np/Qy/Qpsi 只属历史消融。
未把仍被继承的 strong_actions4.ini 移走。四份过时规划 JSON 已删除，可在父提交 e83f7fb 恢复；不修改模型冻结配置链哈希。
