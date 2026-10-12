# Diffusion-AS1：滚动时域浮空器DiT轨迹生成

本项目使用单一条件DiT在预生成动态风场中完成长航时浮空器轨迹生成，不采用分层规划或强化学习执行器。模型读取未来2小时预生成风场，每次生成未来1小时轨迹，只执行前7.5分钟，然后根据新的实际状态和向后移动的风场窗口重新生成。

规则专家仅用于离线生成训练示范和独立基线对照。在线推理执行的控制序列全部由DiT生成，不允许规则专家接管，也不使用规则安全盾替换DiT动作。

## 固定规划设置

| 参数 | 设置 |
|---|---:|
| 物理步长 | 7.5分钟 |
| 单次预测时域 | 1小时（8步） |
| 风场观察时域 | 2小时（16帧） |
| 重规划周期 | 7.5分钟（执行1步） |
| 任务时限 | 按航程、初始能源和初始逆风动态计算（16～48小时） |
| 到达判定 | 目标15 km以内 |

运行逻辑：

```text
当前状态 + 最终目标 + 当前时刻起2小时预生成风场
                         ↓
                        DiT
                         ↓
                 生成8步空速设定值
                         ↓
             物理递推位置、速度和能源
                         ↓
                 只执行第1步
                         ↓
         7.5分钟后更新状态并重新生成
```

这里使用的是从完整预生成风场中截取的未来2小时子序列，不是在线气象预测；DiT控制输出仍为未来1小时。

## 主要改进

- 32小时一次性生成改为1小时预测、7.5分钟滚动更新。
- 任务截止时间不再对所有场景固定为32小时，而是依据起终点距离、初始能源和初始航路逆风程度动态确定，并限制在16～48小时范围内。
- 浮空器最大空速设为20 m/s；推进功耗继续按空速三次方增长，并保持15 m/s工况下的原功耗标定值。
- 合成风场按矢量模长限制在20 m/s以内，避免分别裁剪水平分量造成不符合设定的合成超速。
- DiT生成归一化绝对空速设定值，低层限幅控制器将其转换为速度增量，避免小误差逐步累积到速度饱和。
- 时间由轨迹索引确定，能源由太阳能与推进功耗模型计算。
- 轨迹和目标使用相对坐标条件，提高不同起点间的泛化能力。
- 前8帧风场与8个控制token对齐，额外8帧提供第二小时趋势上下文。
- 风场编码保留空间patch和时间token，不再平均为单个向量。
- 训练、验证和测试使用互不重叠的完整风场序列，并在每个划分内独立保持困难逆风任务比例，避免风场、滑动窗口和难度分布泄漏。
- 正常示范来自成功到达的完整专家任务；失败任务只保留经专家重新规划后安全推进的状态窗口，不把失败控制当作标签。
- 1小时内提前到达目标时，使用风场补偿驻留标签补齐窗口，使模型学习减速和保持目标位置。
- 多候选示范控制器每7.5分钟比较直飞、±25～120 km绕行、驻留和不同巡航速度，使用2小时物理递推评分并截取前1小时作为标签。
- 午后提前进入跨夜能源预算；能源与距离允许时，强逆风候选可使用最大物理空速完成收尾。
- 单一DiT通过直行、左绕、右绕、节能和突围五种条件生成不同候选，而不是依赖在线专家接管。
- 每个任务最多提取8个窗口，最高风险窗口追加不同路线标签；训练数据由5000个任务增至8000个任务，并对困难窗口加权采样。
- 扰动状态必须由专家重新规划后才作为恢复样本，避免条件与控制标签不一致。
- 上一轮规划输入来自专家实际保存并平移后的旧计划，不读取当前窗口的未来标签。
- 训练加入首个实际执行步、专家推进量、末端闭合速度、停滞、状态、能源、边界和平滑约束；错误旧计划只作弱参考并加入专门纠偏样本。
- 推理候选混合不同CFG、DDIM随机度以及继承、清空和扰动的上一轮计划；检测到1小时停滞时清空旧计划并由DiT重新采样，不切换到专家控制器。
- 多候选轨迹通过物理递推后的综合代价选择，并对末段远离目标、闭合速度不足、无效高速耗能和不可行计划进行处罚。
- 以8个中距离逆风任务和4个对照任务组成的固定纯DiT闭环测试选择最佳checkpoint，并根据闭环指标早停，避免窗口损失掩盖长航时失败。
- 支持预测噪声和直接预测干净控制序列两种训练目标。
- 支持DDPM、DDIM、CFG及可选的归一化物理引导。
- 测试输出论文图4/图6单个子图样式的静态图和GIF滚动动画。

## 项目结构

```text
config.py                    # 滚动规划、模型、损失及路径配置
train.py                     # 训练、验证、EMA和版本化checkpoint
test.py                      # 单场景完整滚动测试及动静态可视化
batch_test.py                # 多场景完整滚动评估
rolling_planner.py           # 1小时预测/7.5分钟执行主循环
visualization.py             # 论文子图样式PNG/GIF
data/dataset.py              # 任务级划分与1小时窗口数据集
dynamics/rolling.py          # 可微/NumPy物理递推、条件和候选评分
expert_gate.py               # 正式训练前的30场景专家质量门禁
models/dit.py                # 带时间对齐风场交叉注意力的DiT
models/wind_encoder.py       # 时空风场patch编码器
models/diffusion.py          # 训练损失、DDPM/DDIM、CFG、多候选选择
experiments/run_ablation.py  # 固定场景消融实验
experiments/baseline_ppo.py  # 旧PPO基线（仅作比较）
```

## 环境

```bash
conda activate dit
pip install -r requirements.txt
```

结果动画使用GIF格式，不依赖外部视频编码器。

## 生成数据并训练

首次训练会在 `data/rolling_trajectories_v7/` 自动生成8000个成功任务对应的新版本窗口数据：

正式训练前可先运行固定30场景专家门禁：

```bash
python expert_gate.py --tests 30 --legacy-v4-scenarios
```

```bash
python -u train.py
```

后台训练：

```bash
nohup python -u train.py > train_rolling_v7.log 2>&1 &
echo $! > train_rolling_v7.pid
```

监控：

```bash
tail -f train_rolling_v7.log
tensorboard --logdir logs/rolling_v7
```

新模型保存在 `checkpoints_rolling_v7/`。已有的 `checkpoints_rolling_v6/`、`checkpoints_rolling_v5/` 76.7%纯DiT基线、`checkpoints_rolling_v4/`、
`checkpoints_rolling/` v3模型和最初的 `checkpoints/model_final.pt` 都不会被覆盖。

## 单场景滚动测试与动画

```bash
python test.py \
  --sampler ddim \
  --steps 50 \
  --cfg 2.0 \
  --candidates 8
```

输出位于 `checkpoints_rolling_v7/rolling_result/`：

- `rolling_trajectory.png`：论文子图样式静态图；
- `rolling_trajectory.gif`：默认每秒12帧，并标注出发、当前时刻；
- `rolling_trajectory_frames/`：动画逐帧PNG；
- `rolling_trajectory_diagnostics.png`：距离、能源和速度曲线。

图中使用：

- 彩色等值面：当前风速大小；
- 黑色箭头：当前风向；
- 白色实线：已经执行的轨迹；
- 红色虚线：当前DiT生成的未来1小时轨迹；
- 蓝色圆点：当前位置；
- 红色圆形：最终目标。

## 批量测试

```bash
python batch_test.py \
  --tests 100 \
  --sampler ddim \
  --steps 50 \
  --cfg 2.0 \
  --candidates 8 \
  --seed 20260901 \
  --animate-representative \
  --compare-expert
```

主要指标包括到达率、终点误差、完成时间、路径长度、最低/最终能源、越界率、速度、重规划次数、相邻规划跳变、候选轨迹多样性、安全回退次数和整次任务运行时间。`--compare-expert`会在完全相同的任务上另存`expert_metrics.csv`，用于判断DiT是否真正优于规则示范控制器。

## 消融实验

```bash
python experiments/run_ablation.py \
  --samplers ddim:20,ddim:50 \
  --cfgs 1.0,2.0,3.0 \
  --candidates 1,5 \
  --physics-guidance 0.0 \
  --scenarios 10
```

`config.PREDICTION_TYPE`可在 `epsilon` 与 `x0` 间切换。两种模式的checkpoint元数据不同，切换后必须使用独立训练得到的模型。

## 修改前基线

本次滚动DiT改造前的代码已推送并打标签：

```text
commit: b3a51f3
tag:    pre-rolling-dit-20260901
```

查看旧版：

```bash
git switch --detach pre-rolling-dit-20260901
```

从旧版创建恢复分支：

```bash
git switch -c restore-pre-rolling pre-rolling-dit-20260901
```

旧版最终评估结果保存在 `FINAL_EVALUATION.md`，用于和新滚动模型对照。

## 许可

仅供研究使用。
