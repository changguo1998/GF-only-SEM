# CPU-MPI 与 CUDA 求解器一致性测试

## 测试目的

确认同一离散模型上 CPU-MPI 与单 GPU CUDA 的弹性推进、全局 DOF 装配、共享节点处理和
C-PML 更新给出相同数值结果。

## 测试方法

在 `fullspace-cubic` 上分别运行 16-rank CPU-MPI 与单 GPU CUDA 的 x 向点力，将记录按
坐标对齐，在 step 400 和 700 对整个应变数组计算相对 L2 与 Pearson 相关系数。

## 参数选择

- 均匀全空间、三维规则六面体、N=4、全六面 C-PML。
- 同一个 `config.h5`、STF、网格与 partition；比较 x 方向即可隔离后端差异。
- 门限：`rel_l2 < 0.01`、`corr > 0.999`。

## 测试结论

| 步 | 相对 L2 | 相关系数 | 结果 |
| ---: | ---: | ---: | --- |
| 400 | 1.737e-4 | 0.99999998 | 通过 |
| 700 | 5.113e-4 | 0.99999987 | 通过 |

测试过程中修复了三类真实问题：MPI 交换遗漏棱/角共节点、共享节点 PML 阻尼选择不一致、
`compute_full_strain` 的重复节点偏移。修复后速度/位移/应变均支持 CPU-MPI 与 CUDA
一致；当前证据仅覆盖弹性 x 方向，不外推为所有有限 Q 参数组合的逐位一致。

详细记录见
[`examples/fullspace-cubic/VERIFICATION.md`](../../examples/fullspace-cubic/VERIFICATION.md)。
