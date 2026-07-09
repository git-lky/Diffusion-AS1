"""Diffusion-AS1: Diffusion-based airship trajectory planning."""
import math

# ===== Environment =====
WIDTH = 210; HEIGHT = 120
WIDTH_KM = 525; HEIGHT_KM = 300
MAX_SPEED = 15.0 
RANGE_SCALE = 10                                #风场压缩比例。原始风场 210×120，压缩到 21×12 送入模型
DT = 7.5 * 60                                   #每一步的物理时间（秒）。7.5分钟 × 60 = 450秒
ACTION_RANGE = 3.0                              #每步允许的速度变化范围（m/s² 等效）
TARGET_RANGE = 15                               #到达目标点的判定半径（公里），进入就视为到达
START_TIME = 8; START_ENERGY = 0.8
SOLAR_TIME = [6, 18]; SOLAR_POWER = 18
MAX_POWER = 16                                  #电机最大消耗功率（千瓦）
DEFAULT_POWER = 1                               #基础待机功耗（千瓦）
MAX_ENERGY = 80                                 #电池总容量（千瓦时）

# ===== Trajectory =====
TRAJ_LEN = 64                                   #每条轨迹包含 64 个时间步
STATE_DIM = 6                                   #每步状态：[x, y, vx, vy, time, energy]
WIND_VEC_DIM = 504                              #风场展平后的维度：21 × 12 × 2（u/v分量）

WIND_NUM_FRAMES = 8                             #每轨迹保存的风场帧数（用于时序编码器）
HARD_SAMPLE_RATIO = 0.3                         #长距离/逆风困难样本占比

# ===== Wind Field =====
WIND_BUFFER_SIZE = 200; WIND_HOURS = 120        #预生成 200 组不同的风场序列，每组风场模拟 120 小时（5天）
WIND_NUM_INP = 3                                #时间插值次数（3次二分插值 = 8倍分辨率）
WIND_MIN_AVE_V = 6; WIND_MAX_AVE_V = 10         #基础风速度范围[6,10]m/s
WIND_VARY_SPEED = 8                             #阵风最大强度（m/s）
WIND_VARY_NUM = 8                               #阵风元素个数

# ===== Diffusion =====
DIFFUSION_STEPS = 256                           #去噪步数。推理时从纯噪声走 256 步"浮现"轨迹
DIM = 128                                       #U-Net 的基础通道数
DIM_MULTS = [1, 2, 4]                           #U-Net 各层的通道倍数。实际通道：[128, 256, 512]
WIND_ENC_OUT = 64                               #风场经 CNN 编码后的向量维度
CONDITION_DIM = 8 + WIND_ENC_OUT                #条件总维度 = 起点+目标(8) + 风场编码(64)

# ===== Training =====
BATCH_SIZE = 64; LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
GRAD_CLIP = 1.0                                 #梯度裁剪上限，防爆炸
EMA_DECAY = 0.995                               #指数移动平均衰减率。最终推理用 EMA 模型，更稳定
SAVE_EVERY = 20                                 #每 20 轮保存一次 checkpoint
NUM_TRAJECTORIES = 5000                         #训练数据量，生成训练轨迹

# ===== Paths =====
MODEL_SAVE_DIR = "./checkpoints"
DATA_DIR = "./data/trajectories"
LOG_DIR = "./logs"

#DDPM噪声调度器
def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    x = [math.cos((t/T + s)/(1+s) * math.pi/2)**2 for t in range(steps)]
    alphas_cumprod = [v/x[0] for v in x]
    betas = [1 - alphas_cumprod[t+1]/alphas_cumprod[t] for t in range(T)]
    return [max(min(b, 0.999), 1e-5) for b in betas]
