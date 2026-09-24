# 后处理数值一致性与并行性能测试

## 测试目的

验证集总质量 L2 应变投影、位移/速度/加速度独立计数、tile 索引、串行/MPI 统一流程、
每 tile 单写者约束和 record-rank 重用优化。

## 测试方法

先用单元测试覆盖 tensor 排列、独立计数和 tile schema；再在 halfspace、20³ fullspace
和有限 Q 小模型上分别运行串行与不同 MPI worker 数，逐 dataset/attribute 比较并记录
文件打开、数据集读取、墙钟和内存。

## 参数选择

- halfspace：500 帧、三方向、16 tiles，串行对 MPI-4。
- 20³：800 帧、62073 唯一节点、16 tiles，MPI-4；字段总预算默认 32 GiB。
- 小有限 Q：2880 单元、55 帧、4 tiles，串行对 MPI-2。
- tile 分配使用 record-rank 重叠优先，并在 10% 工作量窗口内平衡负载。

## 测试结论

- halfspace 串行 158.4 s、MPI-4 61.1 s，2.59×；160 datasets 和全部 attributes 一致。
- 20³ 旧逐 tile 方案 34.3 s，worker-local record 重用后 19.0 s；每 worker 文件打开
  9600→2400、dataset 读取 38400→9600，HDF5 读取占比 59.187%→17.693%。
- 20³ 完整输出的解析检查为 corr=0.9742、scale=0.999、拟合 L2=0.1551。
- 有限 Q 小模型串行/MPI-2 为 0.4/0.3 s，4 tiles、40 datasets 全部一致且有限。
- 旧“每 MPI rank 构建完整合并场”约需 331 GB，已移除；当前内存与 worker 所属 tile
  batch 成正比。`GF_POST_MEMORY_GB` 控制共享字段预算。
- 集总质量 L2 只用于单元应变到连续 GLL 节点的投影；CG-SEM 已连续的位移、速度和加速度
  使用各自独立的计数平均。旧质量归一化、共享计数器和 tensor 转置结果均已失效。

设计与完整证据见
[`docs/design/postprocess-tile-parallel.md`](../design/postprocess-tile-parallel.md)。
