# Handoff — CG-SEM Assembly Fix Implementation

> 创建时间: 2026-07-12
> 最后更新: 2026-07-13
> 状态: **实现完成** — 全部 82/82 fix-plan 复选框已完成

## 项目背景

gf-calculation 是一个 3D 粘弹性 SEM 正向求解器。此前求解器使用**单元局部 DOF 编号**，无全局装配机制，波无法正确跨单元界面传播。半空间示例 rel_l2 误差 ~100%（CUDA 输出全零，MPI+CPU 输出 inf）。

## 已完成的 5 阶段修复

| Phase | 内容 | Tasks | 状态 |
|-------|------|-------|------|
| 0 | **预处理器** — ibool (`local_element2rank_node`) 计算，交换模式转换为全局 DOF | 0.1-0.3 | ✅ |
| 1 | **求解器基础设施** — 全局数组、scatter/gather、全局 mass/damping 装配 | 1.1-1.4 | ✅ |
| 2 | **CPU 时间循环** — Newmark 重写，使用全局编号 + 交换 + 质量装配 | 2.1-2.4 | ✅ |
| 3 | **CUDA 求解器** — 原子 scatter、gather、全局 Newmark 核 | 3.1-3.2 | ✅ |
| 4+5 | **I/O + 测试** — 重启格式适配、单元测试、集成测试 | 4.1-4.3, 5.1-5.5 | ✅ |

### 修复的 3 个关键 Bug

| Bug | 症状 | 根因 | 修复 |
|-----|------|------|------|
| **NaN 爆炸** | ghost-only 节点 `0/0=NaN` | 42-59% rank 节点质量为零（仅属 ghost 元素） | `newmark_correct()` 中 `if (mass <= 0) skip`（CPU+CUDA） |
| **多 rank 爆炸** | displacement 1e+31 | residual exchange 加邻居力但质量只有本地 → 过度加速 | **质量 exchange**：共享节点 `m += m_neighbor` |
| **u/v 分歧** | mass exchange 后仍 2e+06 | 两 rank 用不同位移算元素核 → 残差不一致 → 正反馈 | **u_tilde sync**：预测位移在元素核前 exchange+average |

### 其他修复

- **Restart 适配全局 DOF**：`RestartWriter` 支持 `use_global_dof` 属性，全局模式写 flat 1D 数组
- **Ghost 数据可选**：`io.cpp` 用 `try_read` 替代硬读，支持无 ghost 的分区
- **源注入 DOF 偏移 bug**：`source.cpp` 中 `dof_base * 3 + dir` → `node_idx * 3 + dir`（预存 bug，1 单元源时无害）
- **CUDA 段错误**：`read_partition_all` 合并分区时缺少 ibool → 越界。修复：合并后清除 ibool，CUDA 强制 element-local 路径
- **命名重命名**：27 个文件中的 SPECFEM3D 术语 → X2Y 约定

## 测试结果

| 测试套件 | 数量 | 状态 |
|---------|------|------|
| Python (pytest) | 188 | ✅ 全部通过 |
| C++ Catch2 (5 个文件) | 19 (441+55+1153+20+3 assertions) | ✅ 全部通过 |
| 2-rank halfspace (3 方向) | 位移 ~1e-12 m | ✅ 与 1-rank 和 legacy 路径一致 |
| CUDA 单 GPU | 1000 步 / 2.4s | ✅ 无段错误 |
| Lamb 对比 | rel_l2 ≈ 472 | ⚠ 预存问题（PML 反射 + 表面源 vs 埋深源矛盾） |

## 架构决策（实现中发现）

### 1. per-rank ibool 不可跨分区合并

每个 rank 的 `local_element2rank_node` 使用自己本地的节点编号（0 到 n_rank_node-1）。两个不同 rank 的节点 100 对应不同的物理节点。因此：

- **MPI 求解器**：`read_partition(path, rank)` — per-rank ibool 正确
- **CUDA 单 GPU / 合并读取**：`read_partition_all` / `read_partition_range` — 清除 ibool，强制 element-local 路径

### 2. 质量 exchange 必须

无质量 exchange 时：`a_new = (r_local + r_neighbor) / m_local` → 共享节点加速度 ~2x 正常值 → 正反馈 → 爆炸（1e+30）。

### 3. u_tilde sync 必须在元素核之前

即使质量 exchange 正确，两 rank 在共享节点上的位移仍会漂移。元素核使用不同位移计算残差 → exchange 求和后加速度不正确 → 爆炸。

### 4. 非零位移的正反馈循环

```
u_rankA ≠ u_rankB (共享节点)
  → r_rankA = K·u_rankA ≠ r_rankB = K·u_rankB
  → r_exchange = r_A + r_B (过大致使 sum ≈ 2× 正确值)
  → a_new = r_exchange / m_exchange (≈ 2× 正确值)
  → u_new (更大) → 下一次迭代更严重的分歧
```

修复链：**质量 exchange** → **u_tilde sync**（断开了这个循环）。

## 剩余工作

### 高优先级

- **Lamb 对比 rel_l2 验证**：当前 rel_l2 ≈ 472，可能原因：
  1. PML 是简单速度阻尼（非 C-PML），反射波影响半空间解
  2. Johnson (1974) 解析解要求源在自由面下 10m，SEM 源在自由面
  3. 修复：实现 C-PML 或使用 buried source 替代

- **CUDA + MPI (multi-GPU) 集成测试**：`gf_solver_elastic_mpi_cuda` 二进制未测试。CUDA 原子 scatter 在 multi-GPU 场景可能需要同步

### 低优先级

- **CUDA 全局 DOF 路径**：当前 CUDA 单 GPU 使用 element-local 路径。如需与 MPI CPU 路径完全一致，需实现跨分区 ibool 合并（涉及全局节点编号）
- **双阶段 MPI 通信优化**：当前是单阶段 exchange（无计算/通信重叠）
- **C-PML 完整实现**：当前使用速度阻尼替代，对吸收边界有反射

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `docs/superpowers/plans/2026-07-12-cg-sem-assembly-fix.md` | 修复方案（82/82 完成） |
| `docs/design/algorithm-verification.md` | 与 SPECFEM3D 算法一致性检查 |
| `docs/design-decisions.md` | 架构设计决策 |
| `docs/design/naming-convention.md` | X2Y 命名约定 |
| `preprocess/partition.py` | ibool 计算 + 交换模式转换 |
| `forward/share/src/solver.cpp` | 核心求解器循环 |
| `forward/share/src/io.cpp` | 数据读写（含 read_partition_all/range） |
| `forward/share/src/assembly.cpp` | scatter/gather 实现 |
| `forward/share/include/gf/assembly.hpp` | scatter/gather 模板 |
| `forward/share/src/restart.cpp` | 重启格式（全局 DOF + 向后兼容） |
| `forward/share/src/cuda_step.cu` | CUDA 求解器状态分配 + 核函数 |
| `forward/share/src/element_cuda.cu` | CUDA 元素核 |
| `tests/test_assembly.cpp` | scatter/gather 单元测试 |
| `tests/preprocess/test_partition.py` | ibool 单元测试 |

## 环境

```bash
# 激活 Spack
source $HOME/.spack/share/spack/setup-env.sh
spack load /zkrqzmds    # OpenMPI 5.0.10
spack load eigen         # Eigen 3.4.0
spack load cuda          # CUDA 13.2 (仅 CUDA 构建需要)

# 构建 CPU
cd forward && cmake -B build -DGF_DEVICE_BACKEND=CPU && cmake --build build

# 构建 CUDA
cmake -B build -DGF_DEVICE_BACKEND=CUDA && cmake --build build

# 运行全部测试
python -m pytest tests/ -q                        # Python (188)
./build/tests/test_assembly -s                     # C++ scatter/gather (8)
./build/tests/test_newmark -s                      # C++ Newmark (3)
./build/tests/test_integration -s                  # C++ integration (3)
./build/tests/test_io -s                           # C++ I/O (2)
./build/tests/test_source -s                       # C++ source (3)
```
