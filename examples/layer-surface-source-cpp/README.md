# 层状介质地表震源（全 C++）

该用例是 `examples/layer/` 的独立 C++ 版本，不读取 `config.py`，也不调用任何
Python 工具。网格生成、预处理、CUDA 正演和 MPI 后处理均由 C++ 程序完成。

主要参数：22×22×11 个谱元，第一层厚 500 m，1 Hz Ricker 地表点力，侧面和底面
各 3 层 PML，地表为自由表面。

```bash
bash examples/layer-surface-source-cpp/run.sh
```

默认主机内存限制为 60 GiB，后处理使用 4 个 MPI rank；可通过
`GF_MEM_LIMIT_GB` 和 `POSTPROCESS_RANKS` 覆盖。

数值流程完成后，可用 PyFK 做独立理论对比：

```bash
bash examples/layer-surface-source-cpp/compare_theory.sh
```

PyFK 不允许震源严格位于真实界面，因此理论计算使用地表下 1 m 的近似震源；
SEM 计算仍使用严格位于自由表面的震源。
