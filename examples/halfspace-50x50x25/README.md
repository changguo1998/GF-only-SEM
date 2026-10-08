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
2–14 km的P/S主体波。当前Lamb比较使用48点Gauss-Legendre、1600段，并独立细化
参考时间网格；下述早期报告使用的24点/800段/50 ms设置已作为历史对照保留。

### 参考时间网格与通过条件

`compare_grid.py`在细网格上计算参考阶跃响应、差分和卷积，再抽取到SEM输出时间。
默认从输出间隔的1/16开始逐次减半，连续两级参考波形在比较窗内的相对L2变化
不超过0.2%才计算验收指标；最大细化倍数1024，未收敛会报错，不会判为通过。
CSV记录`reference_dt_s`和`reference_relative_change`。最终退出码检查所有位置，
不再仅检查11 km基准点。可通过以下参数控制参考解精度：

```bash
python compare_grid.py --reference-initial-refinement 64 \
  --reference-max-refinement 1024 --reference-tolerance 0.002
```

该修复目前接入本算例比较入口；其他算例直接调用`examples/halfspace/reference.py`
旧CLI的流程尚未迁移。SEM时间步长、输出间隔和求解器没有改变。

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

严格地表源设置 `GF_SOURCE_DEPTH_M=-1`：C++用负值选择自由表面定位，Python对应
`source_z_m=None`，写入的实际源深度为0 m；不要将负值解释为模型外的点力。
按上面的预处理命令将深度设为-1后，可只做SEM正演和后处理，跳过昂贵的理论比较
（在算例目录执行，确保没有待保留的同名输出）：

```bash
for direction in x y z; do
  mkdir -p "wavefields/$direction"
  ../../bin-debug/gf_solver_elastic_cuda --direction "$direction"
done
../../bin-debug/gf_postprocess model.h5 config.h5 \
  --fx wavefields/x --fy wavefields/y --fz wavefields/z \
  --tile-x 1 --tile-y 2 --tile-x 2 --tile-y 2 --tile-x 3 --tile-y 2 \
  -o depth_sweep/depth_0000m/greenfun
cp config.h5 depth_sweep/depth_0000m/config.h5
```

### SEM源深度直接对比

在 `depth_sweep/depth_0000m/` 保留严格地表源的 `config.h5` 和 `greenfun/` 后运行：

```bash
python examples/halfspace-50x50x25/compare_sem_depths.py --depths 1 10 50 100 200 500
```

命令复用各档tile，不计算理论解；先核对归档中的域、模拟参数、源坐标和震源时程，
再比较五个固定地表接收点。输出到
`docs/test-reports/data/halfspace-sem-source-depth.csv`，包括九分量合并及逐力方向指标。
原始相对L2以严格地表SEM波形范数为分母，不校正幅度或到时；scale表示埋深源相对
地表源的最佳拟合幅度。源深度变化引起的差异不能直接解释为求解器误差。

1 m源在收敛参考下，2/5/8/11/14 km拟合L2为1.16%/2.31%/3.50%/4.33%/5.76%，
五点全部通过。对应参考内部步长为0.09765625 ms，SEM输出仍为50 ms。
最新地表源复核后，本地工作目录的预处理数据属于严格地表档，复用需设置
`GF_SOURCE_DEPTH_M=-1`；默认1 km必须重新预处理。1 m档保留在`depth_sweep/depth_0001m/`。

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

旧50 ms参考扫描中，1000–50 m五个距离均通过，10/1 m仅14 km未通过。2026-10-09
复用tile细化参考解时间采样后，500–10 m的25点全部通过，随后1 m补跑五点也通过。完整指标、
曲线和结论边界见[测试报告](../../docs/test-reports/15-large-halfspace.md)。

## 参考解收敛诊断

保留上述`depth_sweep`数据后，可独立检查积分精度和参考时间步，无需再运行正演：

```bash
python examples/halfspace-50x50x25/check_reference_convergence.py --validate-only
python examples/halfspace-50x50x25/check_reference_convergence.py \
  --depths 500 200 100 50 10 --offsets 2000 5000 8000 11000 14000 \
  --time-refinements 8 16
python examples/halfspace-50x50x25/check_reference_convergence.py \
  --depths 50 10 --offsets 5000 14000 --time-refinements 32 64
python examples/halfspace-50x50x25/check_reference_convergence.py \
  --depths 10 --offsets 5000 14000 --fine-points 96 --fine-segments 3200 \
  --time-refinements 16
# The 1 m case requires its retained tiles and substantially finer time sampling.
python examples/halfspace-50x50x25/check_reference_convergence.py \
  --depths 1 --offsets 2000 5000 8000 11000 14000 --time-refinements 64 128 256 512
python examples/halfspace-50x50x25/check_reference_convergence.py \
  --depths 1 --offsets 14000 --fine-points 96 --fine-segments 3200 --time-refinements 256
```

细化倍数相对50 ms输出步长。参考解在细时间网格上求导、卷积，再抽取至SEM时间点。
批量积分仅用于本算例沿x轴的诊断，执行前自动与原标量积分核对；原生产参考代码不变。
缓存为不压缩NPZ，保存在`depth_sweep/reference_convergence/`；参数或公式变化后须另行
归档旧缓存，不能跨配置复用。`metrics.csv`只记录当前调用，完整汇总脚本会扫描全部缓存。

使用具备NumPy、SciPy、h5py和Matplotlib的Python环境运行
`summarize_reference_convergence.py`，生成`docs/test-reports/data/`中的收敛、分量、时间窗
CSV及波形SVG。本机系统`/usr/bin/python3`已具备这些依赖，项目`.venv`尚无Matplotlib。
所有基准结果会与原扫描CSV自动核对；理论零分量不报告无意义的相对误差。

上述诊断最初未修改比较入口；现`compare_grid.py`已经接入共用细采样函数和参考
收敛门限。历史缓存采用的直接卷积与新FFT卷积已有数值一致性回归验证。

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
