# Restart/Resume 测试

## 测试目的

验证 `--resume` 不仅恢复位移、速度、加速度，还恢复 C-PML 和有限 Q SLS 的运行态，且
拒绝不完整或与当前模型不兼容的 restart 文件。

## 测试方法

Catch2 创建一个包含主状态和四组 C-PML 数组的 restart HDF5，写入后读取并逐数组比较；
随后删除一个 C-PML dataset，确认读取器抛出异常。CUDA 路径在写 restart 前执行设备到
主机的 C-PML/SLS 状态复制。

## 参数选择

- 小型合成状态：2 个节点的位移/速度/加速度。
- C-PML：`pml_displ_old/new`、`rmemory_displ`、`rmemory_strain` 均使用非零已知值。
- 当前验证入口：CTest 中 `Restart round-trip preserves C-PML state`。

## 测试结论

- 当前 Debug CTest 的 restart 用例通过，完整 C-PML 状态可无损往返；缺一项会被拒绝。
- `--resume` 从 `step+1` 继续，并校验 C-PML/SLS 是否启用及数组长度。
- 会话中曾进行小模型 resume 运行检查，但未保留“不中断连续运行 vs 中断恢复运行”的
  波场差分文件、参数和数值；因此当前证据只支持状态 I/O 与校验逻辑，不支持宣称端到端
  恢复轨迹逐位一致。若修改时间推进或 restart schema，应补一项连续/恢复差分测试。
