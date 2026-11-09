# Known Limitations

## 1. Full-Space Waveform Shape Error

以下指标使用旧的输出步长 Stokes 参考解，属于历史结果。2026-11-09 的 20³、5 s
CPU-MPI Debug 算例在 15 个选点发现旧/新参考解最大相差 31.4%；与新参考解比较，
64 个固定点平均相对 L2 为 0.0040，SEM/参考解 scale 为 1.001。这不能说明旧的 8 s
及 40³ 算例剩余误差的成因；所有旧全空间比较和边界/PML 归因均待复算。

The five-grid full-space study reports mean component-wise correlation of about 0.83–0.85.
All five grids were regenerated on 2026-09-20 after the postprocess averaging and fitted-L2
fixes. From 18³ to 28³ (3.0–4.7 elements per S wavelength), the corrected fitted relative L2
is flat at 0.3414–0.3468, while the SEM/analytical amplitude scale remains 1.013–1.038. Thus
neither the waveform residual nor the absolute amplitude improves with compact-domain grid
refinement.

扩域算例在约 3 单元/S 波长时，旧参考解下源区相关系数为 0.9672、拟合相对 L2 为
0.0882。历史数据提示边界/C-PML 可能有贡献，但参考解时间误差使主因无法确定；
应先统一复算，再分析点源表示与数值色散。

## Resolved: Systematic ~3× Amplitude Factor

**Status:** Fixed in postprocess (2026-09-17).

This was not an inherent SEM/GLL discretization effect. The serial and MPI postprocessors
used one shared `node_count` while independently accumulating displacement, velocity, and
acceleration. With all three quantities present, the count was incremented three times per
cell contribution, so each vector field was divided by three.

Each field now owns its accumulator and sample count. A regression test verifies that
enabling all three fields cannot change any field's average. The full-space analytical
comparison also rejects best-fit SEM/analytical amplitude scales outside [0.8, 1.2]; the old
correlation-only gate could not detect uniform amplitude errors.

Historical scales map exactly as follows because the faulty division was exactly three:

| Case | Historical scale | Corrected scale |
|------|------------------|-----------------|
| Full-space grids | 0.338–0.346 (SEM/reference) | 1.014–1.038 |
| Half-space | 2.95 (reference/SEM) | 0.983 |
| Layer | 2.60 (reference/SEM) | 0.867 |

The historical correlations are unchanged because they are invariant under uniform scaling.
The historical "scale-fitted shape L2" was not actually scale-invariant and is invalid. The
regenerated five-grid study gives SEM/reference scale 1.013–1.038 and corrected fitted relative
L2 0.3414–0.3468 at the 64 fixed receivers.
