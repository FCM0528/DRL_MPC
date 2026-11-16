# 紧凑控制器数值参考

controller_reference.mat只包含四组已保存的state/control/reference/lambda/time数值数组，共632个状态点，没有C++程序、MEX或训练Agent。历史原始模型、数据与迁移工具集中存于archives压缩包和隔离区。

这些完整记录包含必要的启动历史和末端记录，用于逐步算法一致性检查；跟踪评价指标另行严格筛选0≤X≤120。

运行`validate_matlab_replay`执行原生MATLAB回放，运行`review_matlab_native`进行静态分析和边界检查。真实闭环回归使用当前原生Agent及results/matlab_comparison保存的输出，不依赖历史Agent。
