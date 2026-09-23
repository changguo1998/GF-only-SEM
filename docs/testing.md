# 测试与验证方案

本文档是项目测试的统一入口。目标不是穷举“模型 × 求解器 × 后端 × 参数”的笛卡尔积，
而是让每个结论至少有一个独立、可重复的证据，并让同一个端到端算例承担多个相容的验证任务。

## 测试分层

| 层级 | 内容 | 运行时机 | 判定方式 |
| --- | --- | --- | --- |
| L0 | Python/C++ 单元测试 | 每次修改 | 断言全部通过 |
| L1 | 小规模组件集成 | 每次构建或 CI | HDF5 往返、单元正演、MPI halo、CUDA 单元残差 |
| L2 | 三个主算例 | 定期或发布前 | 解析解/半解析解门限与后端一致性 |
| L3 | 浅源、地表源、扩域、网格研究 | 相关改动或专项研究 | 固定参数对照，不作为日常回归 |

## 自动化测试清单

### Python：235 项（另有 1 项环境相关跳过）

| 文件 | 数量 | 覆盖内容 |
| --- | ---: | --- |
| `tests/greenfun/test_cli.py` | 7 | 查询 CLI、输出格式、错误参数 |
| `tests/greenfun/test_index_cache.py` | 22 | tile 扫描、缓存命中/失效/损坏恢复 |
| `tests/greenfun/test_interpolator.py` | 9 | 三线性插值、边界、形状校验 |
| `tests/greenfun/test_library.py` | 24 | 单点/批量查询、源选择、异常输入 |
| `tests/greenfun/test_source_run.py` | 16 | tile 加载、懒加载、重叠节点去重 |
| `tests/preprocess/test_boundary_detector.py` | 3 | 外边界、自由表面、共享面 |
| `tests/preprocess/test_cfl_validator.py` | 12 | CFL、输出步长、stride 与非法参数 |
| `tests/preprocess/test_cli_integration.py` | 1 | Python 配置对 C++ 加速器保持权威 |
| `tests/preprocess/test_config_loader.py` | 6 | 配置加载与字段校验 |
| `tests/preprocess/test_config_writer.py` | 6 | `config.h5` 分组、类型与目录创建 |
| `tests/preprocess/test_cpp_preprocess_smoke.py` | 1 | C++ 预处理生成求解器可读分区；默认跳过 |
| `tests/preprocess/test_gll_geometry.py` | 13 | GLL 点/权重、Jacobian、质量与多单元几何 |
| `tests/preprocess/test_model_loader.py` | 5 | 材料默认值、callable 与类型 |
| `tests/preprocess/test_model_writer.py` | 7 | `model.h5`、PML 标记、SLS 元数据与多分区写入 |
| `tests/preprocess/test_partition.py` | 11 | METIS 分区、全局 DOF 与交换模式 |
| `tests/preprocess/test_pml.py` | 6 | PML 区域和阻尼单调性 |
| `tests/preprocess/test_pml_cpml.py` | 2 | C-PML 面剖面与参数尺度 |
| `tests/preprocess/test_recording_map.py` | 3 | 记录节点映射与 PML 排除 |
| `tests/preprocess/test_source_locator.py` | 14 | 地表/埋藏源、插值权重与 PML 排除 |
| `tests/preprocess/test_stf_evaluator.py` | 8 | STF 时间轴、Ricker 与步进函数 |
| `tests/preprocess/test_topology_reader.py` | 7 | 拓扑 schema、方向约定与错误文件 |
| `tests/test_analytical_compare.py` | 5 | 解析比较、尺度拟合、接收点选择 |
| `tests/test_attenuation_injection.py` | 15 | Qμ/Qκ 独立拟合、SLS 注入、模量校正与弹性极限 |
| `tests/test_finite_q_propagation.py` | 4 | SLS 解析传递函数、门限与 record HDF5 读取 |
| `tests/test_sls_memory.py` | 5 | 应变驱动记忆变量、稳态、体积/偏量独立性 |
| `tests/test_waveform_validation.py` | 2 | 解析算例门限和三倍幅值错误回归 |
| `tests/tools/test_gmsh_to_hdf5.py` | 18 | GMSH 拓扑转换、方向、CSR 与 HDF5 |
| `tests/tools/test_gmsh_to_hdf5_integration.py` | 4 | 四单元完整转换和文件往返 |

运行命令：

```bash
.venv/bin/python -m pytest tests -q
GF_RUN_CPP_PREPROCESS_SMOKE=1 .venv/bin/python -m pytest \
  tests/preprocess/test_cpp_preprocess_smoke.py -q
```

### C++/CUDA：有限 Q 测试已纳入常规 CTest

| 文件 | 数量 | 覆盖内容 |
| --- | ---: | --- |
| `tests/test_gll.cpp` | 11 | GLL 点、权重、导数与插值 |
| `tests/test_element.cpp` | 3 | CPU 单元残差、刚体运动与均匀应变 |
| `tests/test_element_cuda.cu` | 4 | N=3/N=5 CUDA–CPU 残差、刚体平移和 C-PML 单步状态对照 |
| `tests/test_newmark.cpp` | 3 | Newmark 预测、校正与能量守恒 |
| `tests/test_pml.cpp` | 11 | 阻尼、C-PML 索引、记忆变量更新 |
| `tests/test_source.cpp` | 3 | 点力定位、力守恒、多单元分配 |
| `tests/test_sls.cpp` | 6 | SLS 常量、索引、系数与无衰减极限 |
| `tests/test_sls_finite_q.cpp` | 3 | SPECFEM Qμ=20 剪切、Qκ=10 体积与时间步细化 |
| `tests/test_io.cpp` | 2 | 分区与配置 HDF5 往返 |
| `tests/test_restart.cpp` | 1 | C-PML 运行态 HDF5 重启往返 |
| `tests/test_record.cpp` | 2 | 全域四类动态字段、无压缩 schema 和 float32 写入 |
| `tests/test_assembly.cpp` | 8 | 全局装配、震源 RHS、scatter/gather |
| `tests/test_exchange.cpp` | 3 | MPI halo、累加与空模式 |
| `tests/test_integration.cpp` | 3 | 单单元正演、刚体残差与 PML 阻尼 |
| `tests/test_postprocess_tile.cpp` | 8 | tensor 布局、独立计数与 tile schema |

CTest 当前注册 65 项：postprocess 的 8 个 Catch2 case 由一个稳定入口运行，record 测试另有一个
fixture 准备项，因此注册数与逻辑 case 数不同。有限 Q 的 3 个本构用例已注册。

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target test_sls_finite_q
build/tests/test_sls_finite_q --reporter compact
```

## 代表性参数组合

三个主算例固定使用 `N=4`、`output_dt_s=0.01`、三个力方向和不压缩 HDF5。其余参数按验证
目标选择，不在每个模型上重复所有后端。

| 算例 | 代表参数 | 默认求解器 | 证明内容 | 自动门限 |
| --- | --- | --- | --- | --- |
| `halfspace` | 22×22×11；2 Hz；278 m 埋源；Q=1e9；自由表面 | SLS CPU+MPI，16 ranks | SLS 弹性极限下的自由表面主体波、幅值、三方向张量 | 0–2 s：corr≥0.98，拟合 L2≤0.10，scale∈[0.8,1.2] |
| `layer` | 22×22×11；1 Hz；500 m 界面对齐；278 m 埋源；Q=1e9 | elastic CPU+MPI，16 ranks | 层状材料、界面、MPI 正演与 PyFK 一致性 | 0–2 s：corr≥0.98，拟合 L2≤0.20，scale∈[0.8,1.2] |
| `fullspace-cubic` | 24³；1 Hz；全六面 PML；8 s；Q=1e9 | elastic CUDA | 全空间传播、PML、CUDA 和 Stokes 解析解 | corr≥0.80，scale∈[0.8,1.2]；0.80–0.95 记为已知边界误差警告 |

选择理由：

- 物理后端不是新的物理模型。CPU–CUDA 只需在同一个 `fullspace-cubic` 输入上比较，避免在
  半空间和层状模型上重复六种 solver 组合。
- 三个力方向用于生成完整 Green 张量；CPU–CUDA 数值一致性只比较 x 方向的 step 400/700，
  因为另外两个方向已经由解析解算例覆盖。
- Q=1e9 用于严格弹性极限；有限 Q 使用 SPECFEM 参数独立验证 Qμ=20 剪切和 Qκ=10 体积响应。
  `finite-q-propagation` 再用 Qμ=20/Qκ→∞ 验证 1–2 个 S 波长传播后的解析振幅衰减与相位
  色散；弹性对照抵消共同的网格/PML 误差。
- `float32` 是生产记录格式；float64 的格式与往返由单元测试覆盖，不重复整套大算例。

主算例入口：

```bash
bash scripts/run_all_examples.sh
bash scripts/run_all_examples.sh --case halfspace
bash scripts/run_all_examples.sh --case layer
bash scripts/run_all_examples.sh --case fullspace
```

`halfspace`、`layer` 和 `layer-shallow-source` 的 `compare.sh` 都复用
`examples/_shared/verify_waveform.py`。该检查读取比较结果 NPZ，并以退出码同时约束相关系数、
尺度不变 L2 和绝对幅值；其中 `scale` 统一定义为 `SEM = scale × reference`。因此主脚本的
PASS 不再只表示流程运行成功。

## 专项算例

| 算例 | 何时运行 | 与主算例的差异 | 支持的结论 |
| --- | --- | --- | --- |
| `layer-shallow-source` | 修改震源定位/插值时 | 仅把层状模型震源改为 100 m | 浅部非 GLL 点埋源有效；主体波与 PyFK 一致 |
| `layer-surface-source-cpp` | 修改 C++ 预处理或地表源时 | `source_z_m=None`，全 C++ 流程 | 地表源定位及 C++ mesh→postprocess 链路有效 |
| `fullspace-expanded` | 修改 PML或边界时 | 28 km 域、22³、约 6.36 km PML | 源区 corr=0.9672 且 scale≈1，说明主要残差来自有限边界/PML |
| `finite-q-propagation` | 修改 SLS、衰减预处理或 record schema 时 | 20×12×12；Qμ=20/Qκ→∞；y 力沿 x 传播 | 1.5–2.0 Hz 振幅衰减误差≤12%，相位色散误差≤0.11 rad |
| `meshsize/fullspace*` | 修改离散或开展收敛研究时 | 18³/20³/22³/24³/28³，共用 64 固定物理点 | 分辨率变化与边界误差分离；28³ 为内存上限 |

### 20³ 全域 record 回归（2026-09-24）

使用 `meshsize/fullspace20` 的 8000 单元模型，将持续时间缩短为 8 步并保留每步输出，运行
CUDA 三个力方向以及串行/MPI-4 后处理。该测试专门验证全域 record 改造，不用于波形精度
判断。

| 检查项 | 结果 |
| --- | --- |
| Record 范围 | 24/24 文件均为 `cell_scope=all_local_cells`，每个包含 8000 单元 |
| 动态字段 | strain/displacement/velocity/acceleration 形状正确，全部有限且无 HDF5 压缩 |
| 磁盘占用 | 每个 record 60,017,256 B；三方向 8 步共 1.341 GiB |
| CUDA 正演 | 每方向墙钟 2.40–3.07 s，峰值主机 RSS 1.64 GiB |
| 串行后处理 | 内部计时 1.40 s，峰值 RSS 315 MiB；24 次文件打开、96 次数据集读取 |
| MPI-4 后处理 | 内部计时 1.20 s，外部墙钟 2.16 s；每个 worker 均打开 24 个文件 |
| 数值一致性 | 16 个 tile、160 个数据集和 256 个属性逐元素完全一致，所有浮点输出有限 |

单 GPU 会把 16 个输入 partition 合并为一个输出 record rank，因此该模型的每个 MPI
worker 都需要读取同一组 record；8 步规模下 MPI 启动成本高于其并行收益。按实测文件大小
线性外推，原 800 步配置的三方向全域 record 约为 134.15 GiB，明显超过配置中的 20 GiB
存储预算；完整生产运行前必须提高预算或降低快照频率。

专项命令：

```bash
bash examples/layer-shallow-source/compare.sh
bash examples/layer-surface-source-cpp/run.sh
bash examples/layer-surface-source-cpp/compare_theory.sh
bash examples/fullspace-expanded/compare.sh
bash examples/finite-q-propagation/run.sh auto
bash examples/meshsize/run_study.sh
```

## 结论与证据

| 结论 | 首要证据 | 交叉证据 | 不能外推的范围 |
| --- | --- | --- | --- |
| 自由表面主体波与幅值正确 | `halfspace` 0–2 s Lamb 门限 | 10 个固定地表点 mean corr=0.9995 | 2 s 后受有限边界/PML 回波影响 |
| 层状界面实现正确 | `layer` 0–2 s PyFK 门限 | 278 m/100 m 两个源深度 corr=0.9914/0.9930 | 5 s 全时段不能用于判断主体波误差 |
| CPU-MPI 与 CUDA 一致 | `fullspace-cubic/compare_solvers.py` | step 400/700 corr>0.9999998，rel_l2≤5.1e-4 | 当前脚本只比较弹性 x 方向 |
| SLS 在 Q→∞ 回到弹性 | SLS 单元测试和 `halfspace` SLS 解析验证 | 历史端到端 rel_l2=0.0 | 不替代有限 Q 验证 |
| SLS 有限 Q 本构正确 | SPECFEM Qμ=20 剪切、Qκ=10 体积复模量 | 时间步减半稳定 | 单元本构不替代传播验证 |
| SLS 有限 Q 传播正确 | `finite-q-propagation` 解析复波数传递函数 | CUDA 振幅/相位最大误差 9.67%/0.020 rad；2-rank CPU 为 8.54%/0.092 rad | 当前只覆盖均匀介质横向 S 波和 Qκ→∞ |
| 震源绝对幅值正确 | 三个解析算例的 scale 门限 | 扩大全空间源区 scale=0.999 | 不代表晚期反射波形正确 |
| 主要剩余误差来自边界/PML | `fullspace-expanded` 源区 corr=0.9672、拟合 L2=0.0882 | 18³–28³紧凑域拟合 L2 稳定在 0.3414–0.3468 | 近场震源与解析离散仍未单独分离 |
| 串行/MPI 后处理一致 | halfspace 9 tiles 历史逐位对比 | 20³全域record：16 tiles、160 datasets逐元素一致 | 大模型内存峰值仍需单独监控 |

## 推荐运行节奏

```bash
# 每次修改：约束局部逻辑
.venv/bin/python -m pytest tests -q
ctest --test-dir build --output-on-failure

# 定期：三个互补主算例
bash scripts/run_all_examples.sh

# 发布前：按改动选择专项，不机械运行全部大模型
bash examples/layer-shallow-source/compare.sh
bash examples/layer-surface-source-cpp/run.sh
bash examples/fullspace-expanded/compare.sh
bash examples/finite-q-propagation/run.sh auto
```

网格研究耗时和存储量最大，只在离散、PML 或误差解释发生变化时运行。普通功能改动不应重跑
五套网格。
