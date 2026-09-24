# 测试记录索引

本目录按测试目的拆分会话期间完成的验证。每个文件独立记录测试目的、方法、参数选择、
结论、证据位置和结论边界。`docs/testing.md` 仍是运行入口与推荐测试组合，本目录是结果
档案。

## 当前结论总览

| 测试组 | 当前状态 | 记录 |
| --- | --- | --- |
| 自动化单元与组件测试 | 通过 | [01-automated-tests.md](01-automated-tests.md) |
| C++ 预处理与地表源全流程 | 通过 | [02-cpp-preprocess.md](02-cpp-preprocess.md) |
| 半空间 Lamb、Debug/Release | 通过 | [03-halfspace-lamb.md](03-halfspace-lamb.md) |
| 层状介质与浅源/地表源 | 通过 | [04-layered-pyfk.md](04-layered-pyfk.md) |
| 有限 Q 本构与传播 | 通过 | [05-finite-q.md](05-finite-q.md) |
| CPU-MPI/CUDA 一致性 | 通过 | [06-backend-consistency.md](06-backend-consistency.md) |
| 紧凑全空间网格研究 | 完成；无网格收敛趋势 | [07-compact-grid-study.md](07-compact-grid-study.md) |
| 扩域全空间研究 | 通过 | [08-expanded-fullspace.md](08-expanded-fullspace.md) |
| C-PML 实现、稳定性与厚度 | 实现/稳定性通过；厚度误差已量化 | [09-cpml.md](09-cpml.md) |
| Restart/Resume | 状态往返通过；缺少连续/恢复波场差分档案 | [10-restart-resume.md](10-restart-resume.md) |
| Debug/Release record schema | 通过 | [11-record-schema.md](11-record-schema.md) |
| 后处理数值一致性与性能 | 通过 | [12-postprocess.md](12-postprocess.md) |
| 阶段性能与容量 | 完成基线测量 | [13-performance-capacity.md](13-performance-capacity.md) |
| 40³ Debug 扩域与回波前远场 | 通过 | [14-fullspace-40cube.md](14-fullspace-40cube.md) |

## 历史数据处理规则

- 早期后处理曾把位移/速度/加速度错误地按质量归一化，造成约 `1e9` 量级幅值抑制；
  该阶段的幅值和 L2 无效。
- Green 张量的源方向/响应分量曾转置；修复前的非对角分量与辐射方向结论无效。
- 2026-09-17 以前位移、速度、加速度又共用一个计数器，输出被除以 3；旧绝对幅值无效。
- 旧“scale-fitted L2”以未缩放解析范数归一化，不具尺度不变性；旧值无效。
- 24³ 全空间早期接收区硬编码并混入 PML 节点；修正前相关系数无效。
- 有限 Q 传播旧结果使用旧 record schema。2026-09-24 用当前 1 Hz Debug schema 重算
  20×12×12 模型时，CPU/CUDA 的 1 Hz 振幅误差均为 13.67%；该模型的非 PML 横截面只有
  2 km。扩至 20×14×14 后两后端最大振幅/相位误差降至 6.25%/0.0084 rad，当前结论为通过。
- 相关系数对统一幅值缩放不敏感；只有同时报告 `scale` 与相对 L2 才能判断绝对幅值。

除明确标注为“历史”的内容外，本目录只用修复后的数据形成当前结论。
