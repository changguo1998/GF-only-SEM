# 50×50×25 大型半空间验证

该算例与40³全空间测试计算规模接近，但顶面为自由表面：

- 总网格50×50×25，62,500单元，1 km/单元；
- 四侧和底部各5层C-PML，物理区为40×40×20 km；
- 默认均匀弹性介质：Vp=5000 m/s、Vs=3000 m/s、密度2700 kg/m³，可替换配置中的速度和密度函数；
- 1 Hz Ricker震源位于 `(18.5, 25.5, 1) km`，x方向接近物理区1/3位置；
- 一次后处理长轴上的 `x1/x2/x3, y2` 三个tile，覆盖x=`[15,34] km`、
  y=`[23,28] km`；
- 保存2/5/8/11/14 km水平距离、0–2 km接收深度；当前Lamb解只验证地表接收点。

比较窗随距离取3.5/4.5/5.5/6.5/7.3 s，预计比最早PML返回早0.5–2.3 s，可覆盖
2–14 km的P/S主体波。解析Lamb计算使用24点Gauss-Legendre、800段。

2026-09-29 Debug CUDA全流程结果：2/5/8/11/14 km地表点的corr分别为
0.994963/0.995433/0.999008/0.999431/0.999345，scale为1.003104–1.054427，
拟合L2最大0.100242，五点均通过门限。比较窗均早于最早PML返回，并覆盖P/S主体波。

严格地表源z方向对照显示，14 km拟合L2从N=4、16.67 ms时的0.430252，降至N=4、10 ms
时的0.414918，再降至N=5、10 ms时的0.412111。当前测试中步长细化的影响大于谱阶提升；
匹配步长后谱阶提升仅带来小幅改善。详见
[`docs/test-reports/15-large-halfspace.md`](../../docs/test-reports/15-large-halfspace.md)。

## 纯 C++ 预处理

该算例提供 [`config_user.cpp`](config_user.cpp)，可通过 `GF_USER_CONFIG` 编译进
`gf_preprocess`，用 `gf_preprocess run` 完成包括 C-PML 在内的全 C++ 预处理。默认震源深度
为1 km；测试其他深度时可设置 `GF_SOURCE_DEPTH_M`。例如：

```bash
cmake -S preprocess/cpp -B build/preprocess-halfspace \
  -DGF_USER_CONFIG="$PWD/examples/halfspace-50x50x25/config_user.cpp" \
  -DGF_RUNTIME_OUTPUT_DIRECTORY="$PWD/bin-cpp-halfspace"
cmake --build build/preprocess-halfspace --target gf_preprocess -j8
cd examples/halfspace-50x50x25
# Regenerate topology-only model.h5 before each source-depth preprocessing run.
python mesh_gen.py
GF_SOURCE_DEPTH_M=1 "$PWD/../../bin-cpp-halfspace/gf_preprocess" run model.h5 \
  --N 4 --cfl-safety 0.5 --nx 50 --ny 50 \
  --pml-xmin 5 --pml-xmax 5 --pml-ymin 5 --pml-ymax 5 --pml-zmax 5 --n-ranks 16
```

`config.py` 和 C++ 配置均读取 `GF_SOURCE_DEPTH_M`（默认1000 m）。若接着运行 `run.sh`
的正演、后处理和 Lamb 对比，应保持相同的环境变量，并用 `GF_REUSE_PREPROCESSED=1`
保留 C++ 生成的预处理结果。
同深度的1 m/1 km对照结果见测试报告。

完成上述C++编译后，可运行五档埋深扫描（500/200/100/50/10 m）：

```bash
source scripts/env.sh
python examples/halfspace-50x50x25/run_depth_sweep.py
```

每档单独保留`depth_sweep/depth_XXXXm/`中的日志、指标、配置和三个tile，完成比较后
清理该档全域快照；已有指标和tile的档位会跳过。该脚本使用当前算例工作目录，不应
与其他运行并发执行；若修改测试参数，应先另行归档旧结果，避免复用旧档位。
命令包装工具的超时需足够长（本次设置12小时）。扫描结束后工作目录的预处理数据
属于最后的10 m档，复用时需设置`GF_SOURCE_DEPTH_M=10`，默认1 km需重新预处理。

本次扫描已完成：1000–50 m五个距离均通过，10/1 m仅14 km未通过；完整指标、曲线
和结论边界见[测试报告](../../docs/test-reports/15-large-halfspace.md)。

运行Debug CUDA全流程：

```bash
bash examples/halfspace-50x50x25/run.sh
```

仅当 `config.h5` 震源与 `config.py` 一致时，才可复用正演record：

```bash
GF_REUSE_WAVEFIELDS=1 bash examples/halfspace-50x50x25/run.sh
```

三方向Debug全域record预计约197 GiB；三个tile输出约386 MiB。选择tile只降低后处理
内存、计算和Green函数输出量，并不会降低Debug正演快照体积。复用检查会拒绝震源
坐标与当前配置不一致的旧数据。
