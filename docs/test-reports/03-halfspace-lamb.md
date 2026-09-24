# 均匀半空间 Lamb 与 Debug/Release 测试

## 测试目的

验证自由表面、埋藏点力、三方向 Green 张量、主体波形和绝对幅值；同时确认 Debug 的
附加记录不会改变 Release 中共同的应变 Green 函数。

## 测试方法

完整流程为网格生成、预处理、三方向正演、MPI 后处理、Lamb 解析解生成和自动门限检查。
解析比较使用 0–2 s 主体波窗。另对现存 16 对 Release/Debug tile 的所有共同 dataset
逐值比较。

## 参数选择

- 网格/模型：22×22×11、10×10×5 km、N=4，`vp=5000 m/s`、`vs=3000 m/s`、
  `rho=2700 kg/m³`。
- 震源：深度 278 m，1 Hz Ricker、峰值时刻 1 s；Qμ=Qκ=1e9。
- 解析比较点：震中距 500 m；主波窗 `[0,2) s`。
- 多点检查：10 个固定地表点，距离 0.5–1.5 km，覆盖正负 x/y 和对角方向。

## 测试结论

- 当前 1 Hz Debug 产物对 Lamb：相关系数 0.999792、SEM/参考 scale 1.007655、拟合 L2
  0.020396，三个门限全部通过。
- 10 点当前结果：0–2 s 平均相关系数 0.9999、平均拟合 L2 0.0171；0–5 s 为
  0.9992/0.0320。最远 1.5 km 点全时段拟合 L2 为 0.0756，仍可观察到晚期边界影响。
- Release/Debug 的 16 个 tile、112 个共同 dataset、874,508,416 个值逐位一致，
  最大绝对差为 0；Debug 仅增加位移、速度、加速度张量。
- 旧 2 Hz 指标已由本次统一 1 Hz 全流程替代；旧 scale≈0.33 或“参考/SEM≈2.95”来自
  已修复的三倍计数错误，也不再使用。

入口：`bash examples/halfspace/compare.sh`；Debug 使用
`GF_BIN_DIR=$PWD/bin-debug bash examples/halfspace/compare.sh`。
