# Diffusion-AS1 项目交接文档

> **系统稳定性与 BIOS 更新的完整操作指南：** `BIOS_UPDATE_AND_TRAINING_RECOVERY.md`。在 BIOS 更新及验证前不要继续长时间训练。

> 给新会话的完整上下文，阅读此文档即可接续工作。最后更新：2026-08-27

## 1. 项目概况

基于扩散模型（DDPM/DiT）的自主飞艇轨迹规划：以起点、终点、动态风场为条件，迭代去噪生成轨迹。

- 地图：210×120 像素（525×300 km），轨迹 256 步 × 7.5 min = **32h**（>24h，覆盖完整昼夜循环）
- 飞艇：质量 7001kg，太阳能 + 电池能量模型（峰值 18kW，电池 80kWh）
- 风场：基础风（6-10 m/s 随机方向）+ 8 个阵风团，每小时更新，插值后每 7.5 min 一帧
- 模型：DiT 主干（adaLN-Zero，6 层，256 维，8 头，7.4M 参数）+ TemporalWindEncoder（32 帧风场时序注意力，3.8M 参数）

## 2. 环境（重要）

- conda 环境：**`dit`**（Python 3.10.14），所有操作必须在该环境执行
- **torch 2.5.1+cu121**（2.4.1 在本机训练会随机触发 `CUDAGuardImpl d.is_cuda()` 内部断言；不要换 cu130，驱动 535 只支持 CUDA 12.2）
- 其余依赖：numpy 1.26.4、matplotlib 3.10.9、imageio、tensorboard 2.21、tensorboardX 2.6.5
- GPU：RTX 4090 24GB，驱动 535.230.02（CUDA 12.2）
- Codex 沙箱内看不到 `/dev/nvidia*`，跑 GPU 命令必须请求 escalation（沙箱外执行）

## 3. 当前配置（config.py 核心值）

| 参数 | 值 | 说明 |
|---|---|---|
| `TRAJ_LEN` | 256 | 每步 7.5 min，总 32h |
| `WIND_NUM_FRAMES` | 32 | 风场采样帧数 |
| `WIND_FRAME_OFFSETS` | `[0,8,16,...,248]` | 32 帧间隔 8 步=1h，覆盖 0-31h |
| `WIND_HOURS` | 120 | 5 天，插值后 953 帧 |
| `WIND_BUFFER_SIZE` | 200 | 风场组数 |
| `NUM_EPOCHS` | 1000 | 约 157k 训练步 |
| `NUM_TRAJECTORIES` | 10000 | 训练数据量 |
| `SAVE_EVERY` | 100 | 1000 epoch 共 10 个 checkpoint ≈1.3GB |
| `BACKBONE` | `"dit"` | 可选 `"unet"` |

## 4. 已完成的代码改动

### 4.1 风场随机起始
- `data/dataset.py`：`generate_trajectory(..., wind_start=...)`，轨迹模拟与保存的 32 帧风场均从随机位置切片，与推理（batch_test）一致
- wind_start 上限统一公式：`len(wx) - 1 - max(TRAJ_LEN-1, max(WIND_FRAME_OFFSETS))`

### 4.2 飞行时间跨昼夜
- `start_time` 0-24h 随机写入条件向量
- 修复：软着陆插值 time 不回绕 bug（时间曾涨到 52h，改为 `% 1.0`）
- 修复：`solar_energy` 17-18 点抛物线过零返回负太阳能（clamp ≥0）
- 修复：`max_steps` 公式少算 1 步（`len - wind_start`）

### 4.3 风场可视化 + 多轨迹同图
- `test.py` 重写：默认 5 条候选半透明叠加 + 最优粗线 + 风场 quiver 叠加
- `batch_test.py` 原本已支持

### 4.4 真 DDIM 跳步去噪
- `models/diffusion.py`：重写 `sample_ddim` / `sample_ddim_with_guidance`（原实现是伪 DDIM=跳步 DDPM），新增 `_ddim_timesteps` / `_ddim_step`，支持 `eta`
- `sample_multi` 可切换 `sampler="ddpm"|"ddim"`、`ddim_steps`、`eta`
- 实测：DDIM 50 步 ≈ 0.6s vs DDPM 256 步 ≈ 4s（CPU）

### 4.5 分组测试 + 多指标
- `batch_test.py` 新增 argparse：`--tests / --sampler / --steps / --cfg / --candidates`
- 分组统计：按航程距离（<150 / 150-300 / >300km）、按风向（顺/侧/逆风）
- 补打印路径总长指标
- **修复潜伏语法错误** `ckpt[\"dit\"]`（原文件根本无法运行）

### 4.6 对比/消融实验脚本（已备好，暂不执行）
- `experiments/run_ablation.py`：固定场景扫描 sampler / CFG / 候选数，输出 CSV
- `experiments/baseline_ppo.py`：PPO baseline 同场景评估，输出 CSV

### 4.7 数据生成控制器（重要决策）
- 用户提供的 PPO 策略（`./ppo_policy/PPO_policy.pth`）训练于 **8h 任务**，在 32h 任务上**成功率仅 6%**（44% 能量耗尽、飞出地图 7 倍远），不可用于生成数据
- `_simple_action` 升级为 P 控制器：目标速度 0.6（归一化）、增益 k=0.7、边界规避（margin 8px）、风补偿
- 实测数据质量：成功率 84.6%、能量耗尽 0.4%、越界 10.8%、平均速度 3.7 m/s
- `generate_all(use_ppo=False)`：默认内置控制器，只有显式 `use_ppo=True` 才用 PPO
- PPO 路径支持 `./checkpoints/ppo_policy/` 和 `./ppo_policy/` 两处

### 4.8 CUDA 训练稳定性修复
- `data/dataset.py`：DataLoader 改为 `num_workers=0`。数据已全部载入内存，无需多进程；同时避免 CUDA 初始化后反复 fork worker。
- conda 环境重建为 Python 3.10.14 + torch 2.5.1+cu121，`nvidia-nvjitlink-cu12` 固定为 12.1.105。
- torch 2.5.1 下曾完成 35 epoch 隔离实训（loss=0.012296），但正式训练随后仍在 Epoch 16/47 随机触发 `CUDAGuardImpl d.is_cuda()`，因此不能把升级本身视为最终修复。
- Epoch 47 的完整 C++ 堆栈曾将故障定位到 `torch::autograd::Engine` 的设备工作线程，但关闭 autograd 多线程后仍在 Epoch 35 复现，因此该绕行已撤销，不能视为根因修复。
- `train.py` 现每 5 轮原子覆盖保存 `checkpoints/checkpoint_latest.pt`，启动时自动从该文件续训；即使故障复现也最多损失 4 轮。每 100 轮的永久 checkpoint 保留不变。
- `train_watchdog.sh` 最多自动重启异常训练 50 次，每次由 `train.py` 从滚动断点续训；完整训练正常结束后自动停止。

### 4.9 随机崩溃的系统级根因线索（2026-08-27）
- CPU 为 Intel Core i9-13900KF，属于 Intel 官方确认的 13/14 代桌面处理器 Vmin Shift 不稳定范围；典型症状包含应用随机崩溃和整机冻结。
- 主板为 ASUS PRIME Z790-P D4，当前 BIOS 1402（2023-09-11），启动后的 CPU microcode 仅为 `0x11d`。
- Intel 当前要求使用包含 microcode `0x12F` 或更新版本的最新 BIOS，并加载 Intel Default Settings。ASUS 此主板从 BIOS 1820 起包含 `0x12F`，检查时最新版为 1836。
- CUDA 动态库实际加载路径已核验：torch 2.5.1 加载环境内自带的 CUDA 12.1/cuDNN 9 库和系统 NVIDIA 驱动库，没有混用系统 CUDA 10.1/12.0。
- 在更新 BIOS 并恢复 Intel Default Settings 前不要继续长时间训练。更新后再次检查 microcode；若仍随机崩溃，优先考虑 CPU 已发生不可逆退化并走 Intel/整机厂商质保，其次再更新 NVIDIA 535 驱动至系统推荐的 580 分支。

## 5. 数据状态

- 当前 `data/trajectories/trajectories.npz`：**10000 条 × 256 步 × 6 维 + 条件(10000,8) + 风场(10000,32,12,21)**，约 576MB
- 旧 64 步数据备份：`/tmp/trajectories_v1_64step_backup.npz`
- PPO 策略文件：`./ppo_policy/PPO_policy.pth`

## 6. 训练状态（新会话第一件事）

- 第一次训练跑到 **Epoch 138**（loss≈0.0071）时服务器维修断电中断，仅存 `checkpoints/checkpoint_100.pt`（171MB，含 dit / wind_encoder / ema_dit / ema_wind_enc / optimizer）
- 后续在多个随机 Epoch 触发 PyTorch/CUDA 内部断言；升级 torch 2.5.1、改为单进程 DataLoader、关闭 autograd 多线程后仍复现。宿主机检查时 GPU 状态正常且本次启动无 NVIDIA Xid，根因仍在 PyTorch/CUDA 运行时或底层硬件稳定性范围内。
- 已加入滚动断点和自动续训。若要明确从零开始，先将 `checkpoints/checkpoint_latest.pt` 移走；历史 `checkpoint_*.pt` 不会被自动加载。
- 启动指令：
  ```bash
  conda activate dit && cd ~/Diffusion-AS1
  nohup ./train_watchdog.sh > train.log 2>&1 &
  echo $! > train.pid
  ```
- 实测约 10s/epoch，1000 epoch 约 2.8h；数据已缓存，不会重新生成
- **交接后需检查**：`tail train.log`、`ls checkpoints/`、`ps aux | grep train.py`，确认上次重跑是否完成或再次中断

## 7. 待办 / 已知问题

- 训练完成后验证：`python batch_test.py --sampler ddim --steps 50 --cfg 2.0 --tests 30`（图存 `checkpoints/batch_results/`）
- `sweep_gs.py` 已知 bug：硬编码 `WindEncoder()`（4 帧），与 32 帧 `TemporalWindEncoder` 训练不匹配，加载 checkpoint 会报 state_dict 错误——需修（跟随 config）或弃用
- 对比/消融实验：脚本已备好，用户暂不做
- `HARD_SAMPLE_RATIO=0.3`（困难样本占比）已定义未实现（低优先级）
- 项目无 git 提交（全部文件 untracked）；`train.py` 已有滚动断点续训
- README.md 已同步更新到当前配置；`structure.png`、`试验记录.docx` 未改动
