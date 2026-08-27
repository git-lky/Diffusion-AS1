# Diffusion-AS1 崩溃排查与 BIOS 更新/训练恢复指南

> 最后核对日期：2026-08-27
>
> 新会话请先阅读本文和 `PROJECT_HANDOVER.md`。在完成 BIOS 更新及验证前，不要继续长时间训练。

## 1. 当前结论

训练失败并不是 loss 爆炸或某个固定 Epoch 的数据错误。loss 一直正常下降，但训练会在不同轮次随机触发：

```text
d.is_cuda() INTERNAL ASSERT FAILED at ../c10/cuda/impl/CUDAGuardImpl.h
```

有时 Python 进程退出，有一次还发生过整机死机。软件侧的常见规避均未消除错误。当前最有力的根因线索是 Intel i9-13900KF 的已知 Vmin Shift 平台不稳定问题，而机器仍在使用修复发布前的旧 BIOS/CPU 微码。

机器实测信息：

- CPU：Intel Core i9-13900KF
- 主板：ASUS PRIME Z790-P D4（不是 WIFI 版）
- 当前 BIOS：1402，日期 2023-09-11
- 当前 CPU microcode：`0x11d`
- GPU：NVIDIA GeForce RTX 4090
- NVIDIA 驱动：535.230.02
- 系统：Ubuntu 20.04.6 LTS，内核 5.15.0-105
- Python：3.10.14（conda 环境 `dit`）
- PyTorch：2.5.1+cu121

Intel 当前要求受影响的 13/14 代桌面处理器使用包含 microcode `0x12F` 或更高版本的最新 BIOS，并使用 Intel Default Settings：

- Intel 官方说明：<https://www.intel.com/content/www/us/en/support/articles/000102331/processors.html>
- ASUS PRIME Z790-P D4 BIOS 页面：<https://www.asus.com/us/supportonly/prime%20z790-p%20d4/helpdesk_bios/>
- 高度相似的 PyTorch Issue：<https://github.com/pytorch/pytorch/issues/151486>
- 该 Issue 作者称更新 Z790 主板 BIOS 后解决：<https://github.com/pytorch/pytorch/issues/151486#issuecomment-3019550250>

ASUS 此主板的 BIOS 1820 首次明确包含 microcode `0x12F`；截至核对日期，最新版为 BIOS 1836（2026-05-14），并会同时更新对应的 Intel ME。

## 2. 训练故障与排查时间线

1. 最早一次训练曾到 Epoch 138，服务器维修断电中止，留下 `checkpoints/checkpoint_100.pt`。
2. 之后多次在 Epoch 8/12/16/26 等随机位置触发 `CUDAGuardImpl d.is_cuda()`。
3. 重建 conda 环境，并从 torch 2.4.1 升级为 torch 2.5.1+cu121；一次 35 Epoch 隔离测试成功，但正式训练仍复现。
4. DataLoader 改为 `num_workers=0`、`pin_memory=False`，仍复现。
5. 模型参数列表改为启动时一次性保存，避免重复遍历并稳定 EMA 参数映射，仍复现。
6. 临时关闭 autograd 多线程调度，仍在 Epoch 35 复现，因此该临时修改已撤销。
7. 最近一次训练完成日志中的 Epoch 96 后，在下一轮反向传播时再次崩溃。
8. 宿主机检查时 RTX 4090 温度正常，NVIDIA 驱动可通信；本次启动的内核日志没有 NVIDIA Xid 或 OOM。
9. 已核验 PyTorch 实际加载环境内自带的 CUDA 12.1/cuDNN 9 动态库，不存在训练时混用系统 CUDA 10.1/12.0 的证据。
10. 最后发现 CPU/BIOS 组合正处于 Intel 官方确认的不稳定范围：13900KF + BIOS 1402 + microcode 0x11d。

因此，现阶段不要继续通过换 Python 包或反复重训来碰运气；应先更新 BIOS。

## 3. 当前训练数据和断点

训练目前没有运行。`train.pid` 中的旧 PID 已失效。

最新滚动断点：

```text
checkpoints/checkpoint_latest.pt
epoch field: 94
saved loss: 0.008728204292073181
total_steps: 14915
文件大小约 171 MB
```

更新 BIOS 不会删除硬盘数据或这个 checkpoint。BIOS 问题解决后，直接运行 `train.py` 会自动读取该文件，并从日志中的 Epoch 95 继续；Epoch 95/96 可能被重新计算，这是预期行为。

代码目前具备：

- 每 5 轮原子覆盖保存 `checkpoints/checkpoint_latest.pt`；
- 启动时自动读取滚动断点；
- 每 100 轮永久保存 `checkpoint_*.pt`；
- `train_watchdog.sh` 自动重启脚本存在，但在系统稳定性问题解决前不要使用。

## 4. BIOS 更新前的风险说明

BIOS 更新是主板的标准维护操作，但不是零风险。最大风险是：

1. 下载了错误主板型号的 BIOS；
2. BIOS 写入过程中断电、强制关机或重启。

安全要求：

- 只从 ASUS 官方页面下载；
- 型号必须是 `PRIME Z790-P D4`，不能是 `PRIME Z790-P WIFI D4`，也不能是非 D4 型号；
- 使用 BIOS 自带的 ASUS EZ Flash 3，不在 Ubuntu 中强行刷写；
- 供电不稳定或雷雨天气不要更新，条件允许时使用 UPS；
- 更新期间不要拔 U 盘、按重启键或关闭电源；
- 如果任何页面显示的主板型号不匹配，立即取消，不要强行继续；
- 如果没有把握，带着本文让电脑维修人员完成更新是更稳妥的选择。

## 5. BIOS 更新详细流程

### 5.1 准备 U 盘

1. 准备一个可以清空或已有空闲空间的 U 盘。
2. 使用 Ubuntu 的“磁盘”图形工具确认/格式化为 FAT32。格式化会清空所选分区，务必核对是 U 盘而不是系统硬盘。
3. 打开 ASUS 官方 BIOS 页面：
   <https://www.asus.com/us/supportonly/prime%20z790-p%20d4/helpdesk_bios/>
4. 再次确认页面型号为 `PRIME Z790-P D4`。
5. 下载 BIOS Version 1836。
6. 在下载目录校验 ZIP 文件。把下面命令中的文件名替换成实际文件名：

   ```bash
   sha256sum 下载的BIOS压缩包.zip
   ```

   ASUS 公布的 BIOS 1836 SHA-256 应为：

   ```text
   B4A29639F0EA78035863CBB75E27FEFF5F08AD0D00498C1BAE9F4C372766043C
   ```

   如果不一致，不要使用该文件，重新从官方页面下载。
7. 解压 ZIP，将其中的 `.CAP` 文件复制到 U 盘根目录。
8. EZ Flash 3 可以直接选择 `.CAP` 文件，通常不需要在 Ubuntu 中运行压缩包内的 `BIOSRenamer.exe`。

### 5.2 记录原有 BIOS 设置

更新通常会恢复默认设置。进入旧 BIOS 后，使用手机拍下：

- Boot Priority/启动顺序；
- Ubuntu 所在硬盘；
- SATA/VMD/存储相关模式；
- XMP 状态；
- 风扇设置；
- 任何手动 CPU 电压、超频或 ASUS MultiCore Enhancement 设置。

更新后不要恢复 CPU 超频、电压或 MultiCore Enhancement。为了第一次稳定性验证，XMP 也暂时保持关闭。

### 5.3 使用 ASUS EZ Flash 3 更新

1. 保存工作并正常关机或重启，插好 U 盘。
2. 开机时反复按 `Delete` 或 `F2` 进入 BIOS。
3. 如果进入 EZ Mode，按 `F7` 进入 Advanced Mode。
4. 打开：

   ```text
   Tool → ASUS EZ Flash 3 Utility
   ```

5. 在存储设备列表中选择 U 盘，再选择 BIOS 1836 的 `.CAP` 文件。
6. 仔细核对界面显示的主板型号和目标 BIOS 版本。
7. 确认更新。
8. 等待更新完成。期间绝对不要断电、重启或拔 U 盘。
9. 更新后电脑可能黑屏一段时间、自动重启数次并进行内存训练，这是正常现象；耐心等待，不要因为首次启动较慢而强制关机。
10. 如果 EZ Flash 报文件不匹配或拒绝更新，停止操作，不要尝试绕过校验，重新确认主板型号和下载文件。

ASUS 官方说明 BIOS 1836 会同时更新 Intel ME 至相应版本，不需要另外运行旧版 Windows ME 更新工具。

### 5.4 更新后的 BIOS 设置

更新完成后再次进入 BIOS：

1. 选择 `Intel Default Settings`；如果只看到 `Load Optimized Defaults`，先加载它。
2. 保持以下功能关闭：

   ```text
   XMP
   AI Overclocking
   ASUS MultiCore Enhancement
   手动 CPU 电压/频率设置
   ```

3. 检查 Ubuntu 所在硬盘仍为第一启动项。
4. 按 `F10` 保存并重启。

如果更新后暂时找不到 Ubuntu，通常是启动顺序或原有存储模式恢复默认，并不代表硬盘数据被删除。不要格式化硬盘；回到 BIOS，对照更新前照片恢复正确的启动项/存储设置。

## 6. 更新完成后的验证

进入 Ubuntu 后执行：

```bash
cat /sys/class/dmi/id/board_name
cat /sys/class/dmi/id/bios_version
cat /sys/class/dmi/id/bios_date
cat /sys/devices/system/cpu/cpu0/microcode/version
```

期望看到：

```text
PRIME Z790-P D4
1836
新的 BIOS 日期
0x12f 或更高
```

把这四条输出发给助手。确认 BIOS 和微码正确前不要恢复训练，也暂时不要开启 XMP/超频。

下一阶段应先进行短时间、可控的 CPU/内存和 PyTorch CUDA 稳定性测试，再继续长训练。不要同时升级 NVIDIA 驱动，这样才能判断 BIOS 更新是否解决问题。

如果更新后仍然随机崩溃或整机冻结：

1. 保持 Intel Default Settings 和 XMP 关闭；
2. 考虑 CPU 已在旧微码下发生不可逆退化；
3. Intel 已为符合条件的受影响处理器延长质保，应联系 Intel、整机厂商或购买渠道申请检测/更换；
4. CPU/内存确认稳定后，再考虑将 NVIDIA 535 驱动升级到 Ubuntu 当前推荐的 580 分支。

## 7. 系统稳定后恢复训练

确认测试通过后，使用普通训练方式，不启用自动重启：

```bash
cd ~/Diffusion-AS1
conda activate dit
cp train.log train_before_bios_fix.log
nohup python -u train.py > train.log 2>&1 &
echo $! > train.pid
tail -f train.log
```

启动日志应包含：

```text
Using device: cuda
Resumed from ./checkpoints/checkpoint_latest.pt at epoch 95
```

查看日志时按 `Ctrl+C` 只会退出 `tail`，不会停止后台训练。

## 8. 给维修人员的简短说明

```text
主板：ASUS PRIME Z790-P D4
CPU：Intel Core i9-13900KF
当前 BIOS：1402（2023-09-11）
当前 CPU microcode：0x11d
目标：使用 ASUS EZ Flash 3 更新官方 BIOS 1836，更新后加载 Intel Default Settings，首次验证关闭 XMP、CPU 超频和 ASUS MultiCore Enhancement。
原因：机器在 PyTorch/CUDA 训练中随机崩溃并曾整机冻结，Intel 官方要求该处理器使用包含 microcode 0x12F 或更高版本的最新 BIOS。
```
