# Handoff — CG-SEM Assembly Fix Implementation

> 创建时间: 2026-07-12
> 状态: 修复方案评审和算法验证已完成，可开始实现

## 项目背景

gf-calculation 是一个 3D 粘弹性 SEM 正向求解器。半空间示例运行后对比 Lamb 解析解，rel_l2 误差 ~100%（CUDA 输出全零，MPI+CPU 输出 inf）。

## 根因

求解器使用**单元局部 DOF 编号**，无全局装配机制。每个单元独立演化，波无法跨单元界面传播。

## 修复方案

文件: **`docs/design/fix-plan.md`**

五阶段修复链路:

| Phase | 内容 | Tasks |
|-------|------|-------|
| 0 | **预处理器** — 计算 ibool 映射，转换交换模式为全局 DOF | 0.1-0.3 |
| 1 | **求解器基础设施** — 全局数组、scatter/gather 例程、全局 mass/damping 装配 | 1.1-1.4 |
| 2 | **CPU 求解器循环** — 重写 Newmark + 时间循环，使用全局编号 | 2.1-2.4 |
| 3 | **CUDA 求解器循环** — 并行实现，带 atomic scatter | 3.1-3.2 |
| 4+5 | **I/O 和测试** — 录制、重启、交换验证 + 单元/集成测试 | 4.1-4.3, 5.1-5.5 |

方案已评审并修正（10 处问题已解决），已与 SPECFEM3D 参考代码验证算法一致性。

### 方案修订记录

上次评审发现并修复了 10 个问题:

- **严重**: ibool 差一错误(`iglob=1→0`) / 方案 B 不可行(ghost 无 ibool) / Task 2.1 目标函数错误
- **严重程度高**: ibool 覆盖范围说明 / CudaDeviceState 缺少数组 / corner_node 未定义
- **严重程度中**: ibool 展平步骤缺失 / ghost 元素架构文档缺失

## 算法验证

文件: **`docs/design/algorithm-verification.md`**

全部 8 个算法维度与 SPECFEM3D 一致:

- ibool 计算: ✓ (坐标排序, 与 `get_global.f90` 一致)
- Newmark 格式: ✓ (β=0,γ=0.5, 代数等价)
- 单元核 + scatter/gather: ✓ (等价于直接 ibool 全局写入)
- 质量除法: ✓ (与 `accel *= rmass` 时机一致)
- MPI 装配: ✓ (send/recv 累加模式一致)
- 源加载: ✓ (通过 scatter 分发 = 直接全局注入)
- 应变/录制: ✓ (相同 GLL 导数与 ibool 查找)
- PML: ⚠ 简化模型(速度阻尼) vs C-PML

## 前置文件准备

| 文件 | 状态 | 说明 |
|------|------|------|
| `docs/workflow-comparison.md` | ✅ | 10 个类别、38 项差异的 SPECFEM3D 对比 |
| `docs/design/fix-plan.md` | ✅ | 经评审修正，共 17 个 tasks 跨 5 个 phase |
| `docs/design/algorithm-verification.md` | ✅ | fix-plan vs SPECFEM3D 的 8 维度算法检查 |

## 下一步

开始 Phase 0 实现:

1. **Task 0.1**: 在 `preprocess/partition.py` 中实现 `compute_ibool()` — 对全部元素的 GLL 坐标排序去重，分配 iglob
1. **Task 0.2**: 将 ibool（本地切片展平）写入分区文件 `/field/ibool`，将 nglob 写入 `/field/nglob` 属性
1. **Task 0.3**: 转换交换模式中 send_dof/recv_dof 的索引，从 element-local `(elem*N*3+dir)` 转换为 per-rank 全局 `iglob*3+dir`

## 关键架构决策

- **ibool 为 per-rank**: 每个秩计算自己的 ibool; 不同秩对同一物理节点有不同的 iglob 值。跨秩装配纯靠 MPI 交换模式。
- **只在预处理器中转换交换模式 (方案 A)**: Ghost 元素数据不在分区文件中存储 → 求解器端转换会越界。预处理器有完整 ibool 才能安全转换。
- **拉格朗日乘子法**: 单分量点力通过拉格朗日插值权重施加到源所在单元的所有 GLL 节点上。
- **质量矩阵为节点级标量**: `global_mass[iglob]` — 对弹性各向同性充分，各分量相同。如果将来扩展各向异性则需各分量独立。
- **显式中差 Newmark** (β=0,γ=0.5): CFL 稳定条件限制时间步长。将来可考虑隐式 (β≥0.25)。
- **单元局部中间数组**: 预处理器保持单元局部临时数组（elem_displacement, elem_residual），通过显式 scatter/gather 连接全局状态。这是与 SPECFEM3D 直接全局写入方式的主要架构差异。优势: CUDA 需要 atomic scatter 时接口更清晰。

## 环境

```bash
# 激活 Spack
source $HOME/.spack/share/spack/setup-env.sh
spack load cuda         # CUDA 13.2
spack load /zkrqzmds    # OpenMPI 5.0.10 (有 mpirun)

# 完全设置
source examples/halfspace/env_setup.sh
# 或
source env_setup.sh

# 构建 CPU
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build

# 构建 CUDA
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build

# 测试
cd tests && python -m pytest -v  # Python 测试
cd forward/build && ctest        # C++ Catch2 测试

# 格式化 (commit 前)
bash format.sh
```

## 参考代码

- `external_reference_codes/specfem3d/` — SPECFEM3D Cartesian (read-only, git untracked)
  - `src/shared/get_global.f90` — ibool 算法
  - `src/specfem3D/update_displacement_scheme.f90` — Newmark
  - `src/specfem3D/compute_forces_viscoelastic_calling_routine.F90` — 完整计算流程 + 质量除法
  - `src/specfem3D/assemble_MPI_vector.f90` — MPI 装配
  - `src/specfem3D/compute_add_sources_viscoelastic.F90` — 源加载
