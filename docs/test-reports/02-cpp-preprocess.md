# C++ 预处理与全 C++ 地表源测试

## 测试目的

验证 C++ 预处理器能独立完成 GLL 几何、材料、C-PML、震源定位、METIS 分区、recording
映射及 HDF5 写入，并确认 Python 配置仍是 Python 流程的参数权威来源。

## 测试方法

测试分为三层：Python/C++ 配置一致性单元测试；可选的小网格 C++ 烟雾测试；
`layer-surface-source-cpp` 从 C++ 网格/配置开始，依次运行预处理、CUDA 三方向正演、
MPI 后处理和 PyFK 对比。

## 参数选择

- 烟雾测试：小规则网格，检查求解器可读的 partition/config schema。
- 全流程：22×22×11、N=4、5324 单元、356445 个全局 GLL 节点。
- 地表源：`source_z_m=None`，定位到单元 253、参考坐标 `(0.2232,0.2232,-1)`。
- 时间：`solver_dt=0.005 s`、1000 步、每方向 500 帧。
- 硬件：RTX 5060 Ti；后处理 MPI-4；共享主机内存上限 60 GiB。

## 测试结论

- CUDA x/y/z 分别为 36.85/36.87/36.84 s，MPI-4 后处理 66.2 s，16 个 tile 全部有限且
  非零。
- 0–2 s 与 PyFK：相关系数 0.992286、SEM/参考 scale 0.952191、拟合 L2 0.123995。
- 1/2/5/10 m 理论源深度近似的结果稳定，支持严格地表源实现正确。
- 当前自动套件中烟雾测试默认跳过，因此发布前若修改 C++ 预处理，应显式设置
  `GF_RUN_CPP_PREPROCESS_SMOKE=1`。

原始记录见
[`examples/layer-surface-source-cpp/VERIFICATION.md`](../../examples/layer-surface-source-cpp/VERIFICATION.md)。
