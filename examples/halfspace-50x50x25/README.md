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
