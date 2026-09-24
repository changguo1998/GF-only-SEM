# 自动化单元与组件测试

## 测试目的

验证几何、预处理、SLS、C-PML、MPI 交换、CUDA 单元核、HDF5 I/O、record、后处理和
Green 函数查询的局部不变量，为端到端测试提供快速回归基线。

## 测试方法

2026-09-24 在当前工作树运行：

```bash
.venv/bin/python -m pytest tests -q
ctest --test-dir build-debug -q
```

CUDA/MPI CTest 需要在可访问 GPU 和允许 OpenMPI 启动的会话中运行。受限会话中的 4 个
CUDA、2 个 MPI 用例会因环境退出，因此最终结果取提升权限后的退出码。

## 参数选择

- Python：整个 `tests/`，包括新的 Debug 全域有限 Q record 读取回归。
- C++：`DEBUG=ON`、CUDA 后端、当前 69 个 CTest 注册项。
- CUDA 单元核覆盖 N=3/N=5、刚体平移和 C-PML 单步 CPU 对照。
- MPI 覆盖 halo 传输与 CG-SEM 累加。

## 测试结论

- Python：248 项通过，1 项 C++ 预处理烟雾测试按设计默认跳过。
- CTest：69/69 通过，最终退出码为 0。
- 测试证明当前局部逻辑和已注册组件回归通过；不替代大模型解析解和性能测试。

详细用例清单见 [`docs/testing.md`](../testing.md)。
