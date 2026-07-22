# Buried Source Support — ✅ 实现完成

## Motivation

- **生产用途**: 格林函数计算，震源在自由表面（z=0）
- **调试用途**: 验证/对比需要将震源放在模型内部（如 Lamb 参考解 10m 埋深）
- **设计目标**: 生产路径零影响，调试只需改一行 `config.py`

## 改动总览

```
config.py         ← 新增可选 source_z_m
cli.py            ← 读取 source_z_m，传递给定位器和预检查
source_locator.py ← 双模式：表面模式（现有）+ 埋藏模式（新）
preflight.py      ← 更新 warning 文案
```

求解器侧（`source.cpp`、`solver.cpp`）**无需改动**——Newton 定位和 Lagrange 权重已支持任意位置。

______________________________________________________________________

## 1. config.py — 新增可选字段

```python
# ── Source ───
source_x_m = 5000.0
source_y_m = 5000.0
source_z_m = None      # None → 自由表面 (zmin); float → 指定 Z 坐标
```

向后兼容：旧配置无此字段 → `getattr(config, 'source_z_m', None)` 返回 None。

______________________________________________________________________

## 2. cli.py — 读取 Z 坐标

```python
# 之前:
source_z = float(domain_bounds["zmin"])

# 之后:
source_z = getattr(config, 'source_z_m', None)
if source_z is None:
    source_z = float(domain_bounds["zmin"])
```

日志输出标明模式：

```
"  Source at (5000.0, 5000.0, 0.0), on free surface, in 4 element(s)"
"  Source at (5000.0, 5000.0, 10.0), BURIED 10.0m below surface, in 1 element(s)"
```

______________________________________________________________________

## 3. source_locator.py — 双模式定位

### 当前流程（表面模式，不变）

```
1. 遍历所有单元，找 boundary_tag == 1 的 → free_surface_cells
2. 对每个候选：Newton 迭代求 (ξ, η, ζ)
3. 若收敛 & 权重和 > 0 → 收录
4. 跨元素归一化权重（震源在顶点/边上时涉及多个单元）
```

### 新增埋藏模式

```
1. 获取 is_pml 数组（从 caller 传入），排除 PML 单元
2. 对每个非 PML 单元：
   a. 粗筛：AABB bounding box 测试（GLL 坐标 min/max）
   b. 若通过：Newton 迭代求 (ξ, η, ζ)
3. 若收敛 → 收录
4. 若 0 个元素找到 → 报错；若 >1 个 → warning 提示落在面上/边上
5. **归一化**（与表面模式相同）：跨元素归一化权重（数学上总是需要，
   确保装配后该物理节点的总振幅 = 1.0）
```

**数学说明**：归一化在两种模式下都必要。Lagrange 基函数在单元内
是单位分解（Σ w_ijk ≈ 1.0），但装配时 `scatter_to_rank` 做加法——
共享同一 `iglob` 的多个单元各自加上自己的权重。若不归一化，
共享节点上的振幅会被放大 N 倍（N = 共享单元数）。归一化用总权重和
除以各单元权重，使装配后的总振幅 = 1.0。

埋藏源在单元内部时总权重 ≈ 1.0，归一化是 no-op。若在面/边上则

> 1 个单元，归一化正确分配振幅（与表面模式一致）。

**另注**：埋藏源出现在 >1 个单元中是一个**非致命 warning**——
可能暗示源恰好在单元面/边上（几何巧合），或者源位置选在了
PML 边界附近等。用户应确认位置是否符合预期。

### API 变更

```python
def locate_source(
    topology: TopologyData,
    source_xyz: npt.NDArray[np.float64],
    gll_coords: npt.NDArray[np.float64],
    boundary_tag: npt.NDArray[np.int64],
    N: int,
    is_pml: npt.NDArray[np.bool_] | None = None,  # NEW: 用于埋藏模式排除 PML
) -> dict:
```

返回 dict 增加 `"mode"` 字段：`"surface"` 或 `"buried"`。

### Bounding box 粗筛函数

```python
def _find_candidate_elements(
    source_xyz: npt.NDArray[np.float64],
    gll_coords: npt.NDArray[np.float64],
    is_pml: npt.NDArray[np.bool_] | None = None,
) -> list[int]:
    """返回可能包含 source_xyz 的候选单元列表（AABB 粗筛）."""
    # 对每个非 PML 单元，检查 source_xyz 是否在 GLL 坐标的 AABB 内
```

______________________________________________________________________

## 4. preflight.py — 更新 warning

```python
if not z_on_surface:
    result.add_warning(
        f"Source: z = {source_z} is not on free surface (z_min = {domain_bounds['zmin']}). "
        f"Buried source — only for debugging/validation."
    )
```

______________________________________________________________________

## 5. 测试计划

| 测试 | 文件 | 内容 |
|------|------|------|
| 表面模式不退化 | `tests/preprocess/` | 现有 surface 测试，确保 `source_z_m=None` 行为不变 |
| 埋藏模式定位 | `tests/preprocess/` | 在简单网格中放一个内部源，验证定位到正确单元 |
| PML 排除 | `tests/preprocess/` | 源在 PML 区域应报错 |
| 跨元素 warning | `tests/preprocess/` | 源恰好在面上时 warning |

______________________________________________________________________

## 6. 涉及文件清单

| 文件 | 改动类型 |
|------|---------|
| `examples/halfspace/config.py` | 新增 `source_z_m = None` |
| `preprocess/cli.py` | 读取 `source_z_m`，传递 `is_pml` 给定位器 |
| `preprocess/source_locator.py` | 双模式 + bounding box 粗筛 + PML 排除 |
| `preprocess/preflight.py` | warning 文案更新 |
| `tests/preprocess/test_source_locator.py` | 新增埋藏模式测试 |

**求解器无改动。**
