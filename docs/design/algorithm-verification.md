# fix-plan 与 SPECFEM3D 算法一致性检查报告

> 检查日期: 2026-07-12
> 审查范围: `docs/design/fix-plan.md` 全部 5 Phase, 17 Tasks
> 参考代码: `external_reference_codes/specfem3d/`

______________________________________________________________________

## 1. ibool 算法 (Task 0.1)

| 维度 | SPECFEM3D (`get_global.f90`) | fix-plan | ✓/✗ |
|------|----------|--------|-----|
| 方法 | 坐标排序: 按 (x,y,z) 排序, 邻点距离检查容差 | 相同: 按 (x,y,z) 排序, 距离检查容差 | ✓ |
| 容差 | `SMALLVALTOL = SMALLVAL_TOL * abs(UTM_X_MAX - UTM_X_MIN)` | `tol = 1e-12 * domain_extent` | ✓ |
| 数据结构 | `ibool(NGLLX, NGLLY, NGLLZ, NSPEC_AB)` 4D Fortran | `ibool[n_cell, NGLL, NGLL, NGLL]` 4D, 展平为 `[n_local * n_node]` 存储 | ✓ |
| 编号 | Fortran 1-based | C++ 0-based `iglob = 0` | ✓ |
| Per-rank 范围 | 仅本地元素 (`NSPEC_AB`), ghost 通过 `ibool_interfaces_ext_mesh` | 预处理器中为全部元素(含 ghost)计算, 只存 `[0:n_local]` 切片 | ✓ |

______________________________________________________________________

## 2. Newmark 格式 (Task 2.1)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| 参数 | beta=0, gamma=0.5 (显式中差) | beta=0.0, gamma=0.5 | ✓ |
| 预测步 | `u_new = u + dt*v + dt²/2*a` (原地覆盖), `v_half = v + dt/2*a`, `a = 0` | `u_tilde = u + dt*v + dt²/2*a` (独立输出), u 和 v 保持不变 | ✓ 等价 |
| 校步 | `v_new = v_half + dt/2*a_new`, 不更新 u (已由预测步完成) | `u += dt*v + dt²/2*a_old`, `v += dt/2*(a_old+a_new)`, `a = a_new` | ✓ 等价 |
| 最终 u | u_old + dt*v_old + dt²/2*a_old | u_old + dt*v_old + dt²/2*a_old | ✓ |
| 最终 v | v_old + dt/2\*(a_old + a_new) | v_old + dt/2\*(a_old + a_new) | ✓ |
| 目标函数 | `update_displ_elastic`, `update_veloc_elastic` | `solver.cpp` 内联 `newmark_predict`/`newmark_correct` | ✓ |

______________________________________________________________________

## 3. 单元残差 + scatter/gather (Task 1.3, 2.2)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| 读取全局位移 | `displ(:, iglob)` where `iglob = ibool(i,j,k,ispec)` | `gather_from_rank(displacement, ibool) → local_element_displacement` | ✓ 等价 |
| 单元核 | `-K*displ` 使用 GLL 导数矩阵 `hprime_xx`, `hprimewgll_xx` | `K_e * u_tilde` 使用相同 GLL 导数 `dxi_dx`, `D_mat` | ✓ |
| 写入全局 | 直接 `accel(:,iglob) += contribution` — 无独立 scatter 函数 | `scatter_to_rank(local_element_residual, ibool) → rank_node_residual` — 显式 scatter | ✓ 等价 |
| 质量除法时机 | **单元核后、MPI 装配后**: `accel(iglob) *= rmass(iglob)` (compute_forces_viscoelastic_calling.F90:367-378) | **校步内**: `a_new[i] = residual[i] / mass[i/3]` | ✓ 等价 |
| 质量数组 | `rmass(iglob)` — 预求逆 (× 而非 ÷) | `rank_node_mass[iglob]` — 直接除 | ✓ 等价 |

______________________________________________________________________

## 4. MPI 装配 (Task 0.3, 4.2)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| Send buffer | `buffer_send(:,ipoin) = accel(:,ibool_interfaces_ext_mesh(ipoin,:))` | `exchange_halo` 使用 Task 0.3 转换的 per-rank 全局 DOF send_dof | ✓ |
| Recv/累加 | `accel(:,iglob) += buffer_recv(:,ipoin)` — 加性累加 | `residual[iglob*3+d] += recv_buf` — 加性累加 | ✓ |
| 索引约定 | `ibool_interfaces_ext_mesh` 包含 recv rank 上的 per-rank 全局 DOF | Task 0.3 将 send 和 recv 都转换为 per-rank 全局 DOF | ✓ |
| 双阶段优化 | 外要素(iphase=1)→发送→接收+内要素(iphase=2) 计算/通信重叠 | 单阶段 + exchange, 无异步重叠 | ✓ 正确性一致, 无优化 |

______________________________________________________________________

## 5. PML 阻尼 (Task 2.4)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| 模型 | **C-PML** — 分裂场卷积完美匹配层, 作用于 accel (力) | **Kelvin-Voigt 速度阻尼**: `v -= d*v`, 作用于 global velocity | **不同模型** — 有意简化 |
| 应用时机 | MPI 装配后、质量除法前, 对 accel 作用 | 在 scatter/exchange 前对 global velocity 作用 | ⚠ 不同物理 |
| 效果 | 完美吸收 (理论上无反射) | 基本吸收边界条件, 对垂直入射波效果较好 | 简化 |

> C-PML 需要每个 PML 节点至少 6 个辅助分裂场变量 + 卷积记忆变量, 架构上根本不同。简单速度阻尼对半空间示例足够。

______________________________________________________________________

## 6. 源加载 (Task 2.3)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| 定位方式 | 网格定位器找 1 个主包含单元 | Newton 迭代找**所有**自由面包含单元, 权重归一化 | 不同策略, 均有效 |
| 权重类型 | `sourcearrays(isource, component, i, j, k)` — 多分量 | `src_weights[flat]` — 标量单分量 | 不同, 均有效 |
| 注入方式 | `accel(iglob) += sourcearrays * stf` — 通过 ibool 直接写入全局 | 写入 `local_element_residual` → `scatter_to_rank` 分发 | ✓ 等价 |
| 修改需求 | — | **无需修改 source.cpp** — 注入 local_element_residual 然后 scatter | — |

______________________________________________________________________

## 7. 应变计算 (Task 4.1)

| 维度 | SPECFEM3D | fix-plan | ✓/✗ |
|------|-----------|--------|-----|
| 位移读取 | `displ(:, ibool(i,j,k,ispec))` — 通过 ibool 从全局读取 | `gather_from_rank(displacement, ibool) → local_element_displacement` | ✓ 等价 |
| 应变公式 | GLL 导数 × 单元局部位移 | 相同: `dxi_dx`, `D_mat` | ✓ |
| 顶点录制 | 角点位解码: `corner_i = (corner & 1) ? ...` | 相同解码模式 (已修复 corner_node 未定义问题) | ✓ |

______________________________________________________________________

## 8. 整体时间步结构对比

```
SPECFEM3D (elastic forward):               fix-plan Task 2.2:

update_displ_Newmark                        newmark_predict
  ├─ u = u + dt*v + dt²/2*a  [预测]          └─ u_tilde  [独立输出]
  ├─ v = v + dt/2*a          [半步]
  └─ a = 0                   [清零]

compute_forces_viscoelastic_calling         gather_from_rank(u_tilde)
  ├─ [phase 1] 计算外力 → accel(iglob) +=       → local_element_displacement
  ├─ 加载源 → accel(iglob) +=
  ├─ MPI 异步发送 accel 边界                  compute_element_residual
  ├─ [phase 2] MPI 接收 → accel += 接收缓存      → local_element_residual
  ├─ 计算外力 → accel(iglob) +=
  ├─ PML → accel (C-PML)                      PML on global velocity [简化]
  └─ accel(iglob) *= rmass(iglob) [质量除法]   加载源 → local_element_residual

update_veloc_elastic                         scatter_to_rank(local_element_residual)
  └─ v = v + dt/2*a         [校步]            → rank_node_residual

                                              exchange_halo(rank_node_residual)

                                              newmark_correct
                                                ├─ a_new = residual/mass  [质量除法]
                                                ├─ u += dt*v + dt²/2*a_old
                                                ├─ v += dt/2*(a_old+a_new)
                                                └─ a = a_new
```

**代数等价性确认**：β=0、γ=0.5 时，两个流程得到的 `u_new`、`v_new`、`a_new` 值相同。差异仅在于：

- 内存布局（独立元素局部数组 vs 直接全局写入）
- 质量除法位置（校步内 vs 独立乘 1/mass 循环）
- 位移保存策略（独立输出 vs 原地覆盖）

______________________________________________________________________

## 9. 总体评估

| 类别 | 状态 |
|------|------|
| ibool 计算 | ✓ 与 `get_global.f90` 一致 |
| Newmark 格式 | ✓ 数学等价 (β=0 时可证明产生相同终态) |
| 单元核 + scatter/gather | ✓ 与直接 ibool 全局写入等价 |
| 质量除法 | ✓ 与 `accel *= rmass` 相同 |
| MPI 装配 | ✓ 相同: send/recv + 累加 |
| 源加载 | ✓ 等价 (通过 scatter 分发 vs 直接全局, 中间结果不同, 终态相同) |
| 应变/录制 | ✓ 相同 GLL 导数与 ibool 查找 |
| PML | ⚠ 不同模型 (速度阻尼 vs C-PML) — 有意简化 |
| 双阶段重叠 | — 未实现 (单阶段正确, 无计算/通信重叠优化) |

**结论**: fix-plan 中所有影响**正确性**的算法设计与 SPECFEM3D 参考实现一致。PML 简化(速度阻尼替代 C-PML)是建模层面的权衡, 非算法错误。无双阶段优化不影响结果正确性。
