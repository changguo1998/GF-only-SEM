# Debug/Release record 范围与数据流测试

## 测试目的

验证 Debug 构建记录所有本地单元及四类动态场，Release 仅保存生产所需的浅层记录单元
应变；同时确认后处理能读取新 schema，且 Debug 附加数据不改变共同输出。

## 测试方法

使用 20³ 全空间模型，把运行缩短到 8 步并每步输出，运行 CUDA x/y/z、串行后处理和
MPI-4 后处理。检查 HDF5 属性、shape、dtype、压缩过滤器、有限性及串/并输出。另用完整
halfspace Release/Debug 结果比较所有共同 dataset。

## 参数选择

- 20³、8000 单元、16 partitions、16 tiles、8 步、float32、不压缩。
- Debug 字段：strain/displacement/velocity/acceleration，全 8000 单元。
- Release 字段：记录深度内的应变；无测试/性能日志代码。

## 测试结论

- 24/24 Debug record 均为 `cell_scope=all_local_cells`，四字段 shape 正确且全部有限。
- 每个 record 60,017,256 B；三方向 8 步共 1.341 GiB。线性外推 800 步约 134.15 GiB，
  完整 Debug 生产运行必须调整输出间隔或存储预算。
- 串行/MPI-4 后处理的 16 tiles、160 datasets、256 attributes 逐元素完全一致。
- halfspace Release/Debug 共同输出比较覆盖 874,508,416 个值，最大绝对差为 0。
- 新有限 Q 分析器已增加 `all_local_cells` 读取回归，避免再用旧 recording map 解释全域场。
