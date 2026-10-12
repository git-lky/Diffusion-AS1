"""Diffusion-AS1：长航时浮空器滚动时域 DiT 轨迹规划配置。"""
import math

# ===== 仿真环境 =====
WIDTH = 210                     # 仿真区域横向网格宽度（像素）
HEIGHT = 120                    # 仿真区域纵向网格高度（像素）
WIDTH_KM = 525                  # 仿真区域对应的实际横向距离（千米）
HEIGHT_KM = 300                 # 仿真区域对应的实际纵向距离（千米）
RANGE_SCALE = 10               # 风场空间压缩倍数：210×120 压缩为 21×12
DT = 7.5 * 60                  # 单个物理时间步长度（秒），即 7.5 分钟
MAX_SPEED = 20.0               # 浮空器短时允许的最大空速（米/秒）
ACTION_RANGE = 3.0             # 每个时间步速度增量向量允许的最大模长（米/秒）
TARGET_RANGE = 15              # 到达判定半径（千米），进入该范围即认为到达目标
START_TIME = 8.0               # 默认任务起始时刻（当地时间，小时）
START_ENERGY = 0.8             # 默认初始能源比例，取值范围为 0～1
SOLAR_TIME = [6, 18]           # 太阳能有效时段（当地时间，小时）
SOLAR_POWER = 18               # 太阳能模型的标称输入功率（千瓦）
# 保持原15m/s、16kW设计点对应的立方阻力系数；提高最大空速需要同步增加峰值功率，
# 不能通过只改速度上限而凭空降低相同空速下的推进功耗。
PROPULSION_POWER_COEFF = 16.0 / (15.0 ** 3)  # 推进功耗系数（千瓦/(米/秒)^3）
MAX_POWER = PROPULSION_POWER_COEFF * MAX_SPEED ** 3  # 20m/s时约37.9千瓦
DEFAULT_POWER = 1              # 除推进外的基础功耗（千瓦）
MAX_ENERGY = 80                # 储能系统总容量（千瓦时）
ENERGY_SAFE_THRESHOLD = 0.10   # 能源安全阈值，低于 10% 时触发保护或终止

# ===== 滚动时域规划 =====
# DiT 每次生成未来 1 小时（8×7.5 分钟）的空速设定值，
# 实际只执行前 7.5 分钟（1 步），随后根据新状态和新风场窗口重新规划。
PLAN_HOURS = 1                                      # 单次预测覆盖的规划时域（小时）
TRAJ_LEN = int(PLAN_HOURS * 3600 / DT)              # 单次生成序列长度，当前为 8 步
WIND_CONTEXT_HOURS = 2                              # 每次读取的预生成风场上下文（小时）
WIND_CONTEXT_STEPS = int(WIND_CONTEXT_HOURS * 3600 / DT)  # 当前为 16 帧
REPLAN_MINUTES = 7.5                                # 相邻两次滚动规划的时间间隔（分钟）
EXECUTE_STEPS = int(REPLAN_MINUTES * 60 / DT)       # 每轮实际执行步数，当前为 1 步
# 每个任务依据航程、初始能源和初始逆风单独确定时限。以下MAX值只负责保证
# 风场序列和缓存长度足够，并不表示所有场景都统一运行48小时。
MIN_MISSION_HOURS = 16                              # 动态任务时限下限（小时）
MAX_MISSION_HOURS = 48                              # 动态任务时限上限（小时）
MAX_MISSION_STEPS = int(MAX_MISSION_HOURS * 3600 / DT)  # 风场与缓存所需最大步数
MISSION_TIME_FIXED_HOURS = 8.0                      # 跨昼夜、等待及机动固定余量
MISSION_TIME_DISTANCE_FACTOR = 2.0                  # 理想直航时间的放大系数
MISSION_TIME_HEADWIND_BUFFER_HOURS = 10.0           # 初始逆风对应的最大附加时长
MISSION_TIME_LOW_ENERGY_BUFFER_HOURS = 8.0          # 低初始能源对应的最大附加时长

# DiT 输出归一化的二维绝对空速设定值。低层限幅控制器将设定值转换为
# 有界速度变化后再进行物理递推，可避免连续累加微小生成误差导致速度饱和。
ACTION_DIM = 2                              # 每步控制维度：[x 方向空速设定值, y 方向空速设定值]
STATE_DIM = 6                               # 每步状态维度：[x, y, vx, vy, time, energy]
ROUTE_MODE_NAMES = ("direct", "left", "right", "economy", "breakthrough")
ROUTE_MODE_COUNT = len(ROUTE_MODE_NAMES)    # 单一DiT使用的候选路线模式数量
COND_BASE_DIM = 10                          # 不含路线模式的物理任务条件维度
COND_DIM = COND_BASE_DIM + ROUTE_MODE_COUNT # 物理条件加路线模式独热编码
TRAJECTORY_REPRESENTATION = "airspeed_setpoint"  # checkpoint 中记录的序列表征类型

# 条件向量 cond = [x, y, vx, vy, sin(time), cos(time), energy,
#                    relative_goal_x, relative_goal_y, remaining_fraction]
# 以下常量统一规定各条件量在向量中的索引，避免在其他模块中使用魔法数字。
COND_X = 0          # 当前归一化 x  位置
COND_Y = 1          # 当前归一化 y 位置
COND_VX = 2         # 当前归一化 x 方向空速
COND_VY = 3         # 当前归一化 y 方向空速
COND_TIME_SIN = 4   # 当地时刻的周期正弦编码
COND_TIME_COS = 5   # 当地时刻的周期余弦编码
COND_ENERGY = 6     # 当前归一化剩余能源
COND_GOAL_DX = 7    # 目标相对当前位置的归一化 x 位移
COND_GOAL_DY = 8    # 目标相对当前位置的归一化 y 位移
COND_REMAINING = 9  # 完整任务最大时域的剩余比例
COND_ROUTE_MODE_START = COND_BASE_DIM  # 路线模式独热编码起始位置

# 前8帧对应8个控制时间步，额外8帧提供第二小时的趋势上下文；DiT仍只输出8步控制。
WIND_NUM_FRAMES = WIND_CONTEXT_STEPS                         # 单个规划窗口输入的风场帧数
WIND_FRAME_OFFSETS = list(range(WIND_NUM_FRAMES))           # 窗口内连续风场帧的相对索引
WIND_VEC_DIM = (WIDTH // RANGE_SCALE) * (HEIGHT // RANGE_SCALE) * 2  # 单帧 U/V 风场展平维度

# ===== 动态风场 =====
WIND_BUFFER_SIZE = 200      # 数据生成阶段预先缓存的独立风场序列数量
WIND_HOURS = 120            # 每条原始风场序列覆盖时长（小时），即 5 天
WIND_NUM_INP = 3            # 相邻小时风场的二分插值次数，3 次对应 8 倍时间分辨率
WIND_MIN_AVE_V = 6          # 基础风平均速度下限（米/秒）
WIND_MAX_AVE_V = 10         # 基础风平均速度上限（米/秒）
WIND_VARY_SPEED = 8         # 局部阵风扰动的最大强度（米/秒）
WIND_VARY_NUM = 8           # 每个风场中设置的局部阵风单元数量
WIND_MAX_SPEED = 20.0       # 基础风、阵风及小尺度扰动叠加后的合成风速上限（米/秒）

# ===== 训练数据集 =====
NUM_TRAJECTORIES = 8000         # v7正式数据集接受并保存的成功完整任务数量
WINDOWS_PER_MISSION = 8         # 每个完整任务最多提取的滚动训练窗口数量
MIN_WINDOW_STEPS = EXECUTE_STEPS  # 可接受训练窗口的最短有效长度
MIN_WINDOW_PROGRESS_KM = 1.0    # 判断窗口是否有效推进或明显回退的距离容差（千米）
EXPERT_GROUND_SPEED = 8.0       # 示范控制器期望的巡航地速上限（米/秒）
EXPERT_APPROACH_HOURS = 2.0     # 示范控制器用于计算目标接近速度的时间尺度（小时）
EXPERT_AIRSPEED_RESERVE = 0.90  # 非紧急状态下示范空速占最大空速的上限比例
EXPERT_ACTION_NOISE = 0.03      # 示范控制量加入的随机扰动幅度，用于增加样本多样性
TRAIN_FRACTION = 0.8            # 按完整任务划分的训练集比例
VAL_FRACTION = 0.1              # 按完整任务划分的验证集比例，其余 10% 为测试集
HARD_SAMPLE_RATIO = 0.60        # 每个数据划分内中距离逆风困难任务的目标比例
HARD_ROUTE_MIN_KM = 150.0       # 困难任务距离下限，覆盖当前主要失败区间
HARD_ROUTE_MAX_KM = 300.0       # 困难任务距离上限，避免把配额浪费在已稳定的长距离组
HARD_HEADWIND_COSINE = -0.30    # 两小时沿名义航线平均风与目标方向夹角阈值
HARD_SAMPLE_WEIGHT = 2.0        # 训练采样器对困难窗口的相对采样权重
TRAIN_START_ENERGY_RANGE = (0.60, 1.00)  # 普通任务随机初始能源范围
HARD_START_ENERGY_RANGE = (0.55, 0.85)   # 困难任务随机初始能源范围
STATE_PERTURB_POSITION = 0.005  # 训练条件中的归一化位置扰动标准差
STATE_PERTURB_SPEED = 0.02      # 训练条件中的归一化速度扰动标准差
STATE_PERTURB_ENERGY = 0.01     # 训练条件中的归一化能源扰动标准差
RECOVERY_WINDOW_RATIO = 0.25    # 训练窗口改造成偏航/低能源恢复样本的概率
RECOVERY_POSITION_STD = 0.015   # 恢复样本位置扰动标准差
RECOVERY_ENERGY_DROP_MAX = 0.12 # 恢复样本额外降低的最大能源比例
FAILURE_RECOVERY_WINDOWS = 3    # 每条失败任务最多离线重标注的困难状态窗口数
BAD_PREVIOUS_PLAN_RATIO = 0.30  # 训练时构造错误旧计划并要求DiT纠正的样本比例
BAD_PREVIOUS_PLAN_NOISE = 0.20  # 错误旧计划的附加扰动强度

# 多候选示范控制器使用2小时物理评估，但只输出前1小时控制标签。
EXPERT_EVAL_HOURS = 2
EXPERT_EVAL_STEPS = int(EXPERT_EVAL_HOURS * 3600 / DT)
EXPERT_ROUTE_OFFSETS_KM = (-120.0, -80.0, -50.0, -25.0, 0.0, 25.0, 50.0, 80.0, 120.0)
EXPERT_SPEED_RATIOS = (0.0, 0.55, 0.72, 0.88, 1.00)  # 0表示等待并补偿风场驻留
EXPERT_LOOKAHEAD_KM = 110.0     # 中间航路点前向搜索距离（千米）
EXPERT_WAYPOINT_MARGIN_KM = 75.0  # 专家中间航路点与区域边界的最小距离（千米）
EXPERT_TOP_K = 4                # 从代价接近的前若干可行方案中保留路线多样性
EXPERT_RANDOM_TOLERANCE = 0.20  # 随机示范相对最优代价的允许比例
EXPERT_BREAKTHROUGH_ENERGY = 0.40  # 强逆风高速突围候选允许使用全空速的能源门槛
EXPERT_STRONG_HEADWIND_MPS = 6.0  # 触发突围候选的逆风分量阈值
EXPERT_FINISH_DISTANCE_KM = 120.0 # 接近目标后允许利用可用能源完成收尾
EXPERT_NIGHT_RESERVE_MARGIN = 0.08  # 专家在通用动态储备之外保留的推进余量
EXPERT_ENERGY_CONSERVATION_HOUR = 15.0  # 午后太阳能衰减时提前进入跨夜预算
EXPERT_SCORE_SURVIVAL = 5.0       # 长夜前能源余量不足惩罚
EXPERT_SCORE_TREND = 2.0        # 末段继续远离目标的惩罚权重
EXPERT_SCORE_HEADWIND = 0.35    # 航路累计逆风暴露惩罚权重
EXPERT_SCORE_CLOSING = 0.75     # 末端地速未指向目标的惩罚权重
EXPERT_SCORE_PATH = 0.08        # 无效绕行路径长度惩罚权重
MULTIMODAL_HARD_WINDOWS = 2     # 困难任务最高风险窗口追加两条不同路线模式标签
MULTIMODAL_MAX_REGRESSION_KM = 5.0  # 绕行标签首小时允许的暂时远离目标量
DATA_VERSION = "rolling_v7"

# ===== 网络模型 =====
BACKBONE = "dit"           # 去噪主干类型，当前滚动控制序列仅使用 DiT
D_MODEL = 256              # DiT 内部 token 的特征维度
DIT_LAYERS = 6             # 自适应层归一化 DiT 模块数量
DIT_HEADS = 8              # DiT 多头注意力的注意力头数量
WIND_ENC_OUT = 64          # 每个风场空间块输出的特征维度
WIND_PATCH_CHANNELS = 64   # 风场卷积编码器最后的通道数量
WIND_PATCHES_Y = 3         # 风场在纵向压缩后保留的空间块数量
WIND_PATCHES_X = 6         # 风场在横向压缩后保留的空间块数量

# ===== 扩散过程 =====
DIFFUSION_STEPS = 256          # 完整 DDPM 的前向加噪及反向去噪步数
PREDICTION_TYPE = "x0"        # 当前直接预测干净空速设定值；也支持 epsilon 噪声预测
CFG_DROPOUT = 0.1             # 训练时去除目标相关条件的概率，用于分类器自由引导
PREVIOUS_PLAN_DROPOUT = 0.35  # 兼顾冷启动，并保留足够错误旧计划样本供模型学习纠偏
PREVIOUS_PLAN_NOISE = 0.05    # 训练时加入上一轮规划的高斯扰动标准差
PHYSICS_GUIDANCE_SCALE = 0.0  # 推理阶段物理梯度引导强度，0 表示默认关闭
PHYSICS_GUIDANCE_LAST_STEPS = 32  # 仅允许在反向扩散最后若干步应用物理梯度引导
DDIM_ETA = 0.20               # DDIM采样随机性，用于提高多候选差异
INFERENCE_CANDIDATES = 8      # 默认分层随机配置的候选控制序列数量
PREVIOUS_PLAN_CANDIDATE_NOISE = 0.10  # 部分候选对上一轮计划的额外扰动

# 下列辅助损失均基于预测的干净控制序列及其可微物理递推结果计算。
LOSS_STATE = 0.25       # 状态轨迹一致性损失权重
LOSS_PROGRESS = 0.10    # 向最终目标推进及抑制回退的损失权重
LOSS_BOUNDARY = 0.20    # 接近或越过安全边界的损失权重
LOSS_ENERGY = 0.10      # 能源低于安全阈值的损失权重
LOSS_EFFICIENCY = 0.05  # 能源消耗未换来有效目标进展的损失权重
LOSS_SMOOTH = 0.05      # 相邻空速设定值变化的平滑损失权重
LOSS_CONTINUITY = 0.005 # 旧计划只作软参考，避免停滞或错误路线把新规划锁死
LOSS_FIRST_STEP = 0.50  # 放大实际执行的第一个7.5分钟状态误差
LOSS_EXPERT_PROGRESS = 0.40  # 预测推进量低于专家示范时的损失权重
LOSS_CLOSING = 0.20     # 规划末端地速未指向目标的损失权重
LOSS_STAGNATION = 0.20  # 一小时有效推进不足的损失权重
EXPERT_PROGRESS_RATIO = 0.80  # 预测窗口至少达到专家示范推进量的比例
PROGRESS_MARGIN_KM = 2.0  # 终点至少应相对窗口起点取得的目标进展余量（千米）
BOUNDARY_MARGIN_KM = 30.0  # 计算边界风险和可行性时采用的安全边界距离（千米）
BOUNDARY_SHIELD_MARGIN_KM = 10.0  # 仅在逼近真实边界时触发紧急动作替换（千米）

# 多候选控制序列经物理递推后计算综合代价，所有分量归一化后加权求和。
SCORE_PROGRESS = 1.0     # 窗口终点距目标距离的代价权重
SCORE_BOUNDARY = 8.0     # 边界风险代价权重，安全性优先级最高
SCORE_ENERGY = 4.0       # 能源不足风险及能源余量代价权重
SCORE_SMOOTH = 0.2       # 控制序列不平滑程度的代价权重
SCORE_CONTINUITY = 0.05  # 新规划与上一轮规划不连续程度的代价权重
SCORE_REGRESSION = 0.5   # 轨迹中远离目标的回退行为代价权重
SCORE_EARLY_PROGRESS = 2.0  # 实际执行的前7.5分钟明显回退的代价权重
SCORE_EFFICIENCY = 0.25     # 单位能源有效进展不足的代价权重
SCORE_TREND = 2.0           # 规划末段仍在远离目标的惩罚权重
SCORE_CLOSING = 0.75        # 规划末端地速未朝向目标的惩罚权重
SCORE_STAGNATION = 1.0      # 单个规划窗口推进不足的惩罚权重
ENERGY_RESERVE_BASE = 0.15  # 候选计划要求保留的基础能源比例
ENERGY_RESERVE_MAX = 0.60   # 动态能源储备上限，避免过度保守
STAGNATION_WINDOW_STEPS = 8 # 用最近1小时的真实执行状态判断长期停滞
STAGNATION_MIN_PROGRESS_KM = 3.0  # 最近1小时至少应减少的目标距离
STAGNATION_MAX_REGRESSION_KM = 5.0  # 超过历史最近最小距离时触发路线重置

# ===== 训练过程 =====
BATCH_SIZE = 64                       # 每个训练批次包含的滚动窗口数量
LEARNING_RATE = 1e-4                  # AdamW 优化器初始学习率
NUM_EPOCHS = 800                      # 最大训练轮数；由纯DiT困难闭环验证决定早停
RANDOM_SEED = 20260901                # Python、NumPy、PyTorch 共用的随机种子
GRAD_CLIP = 1.0                       # 梯度范数裁剪上限，用于抑制梯度爆炸
EMA_DECAY = 0.995                     # 模型参数指数移动平均的衰减系数
SAVE_EVERY = 100                      # 每100轮保存编号checkpoint，限制长期训练的磁盘占用
RESUME_SAVE_EVERY = 5                 # 每隔多少轮覆盖保存可恢复训练的滚动 checkpoint
VALIDATE_EVERY = 10                   # 每隔多少轮在验证集上计算一次损失
GENERATION_VALIDATE_EVERY = 50        # 每隔多少轮执行一次自由生成质量验证
GENERATION_VALIDATE_SAMPLES = 16      # 每次自由生成验证使用的样本数量
GENERATION_MIN_PROGRESS_KM = 5.0      # 自由生成验证要求的最小平均目标进展（千米）
GENERATION_MIN_POSITIVE_FRACTION = 0.50  # 自由生成样本中取得正向进展的最低比例
GENERATION_MAX_BOUNDARY_FRACTION = 0.25  # 自由生成样本允许发生越界的最高比例
GENERATION_MIN_FEASIBLE_FRACTION = 0.50  # 自由生成样本中至少存在可行候选的最低比例
FULL_MISSION_VALIDATE_EVERY = 100         # 纯DiT完整闭环验证间隔
FULL_MISSION_VALIDATE_SAMPLES = 12        # 固定验证任务数，多数为中距离逆风
FULL_MISSION_VALIDATE_CANDIDATES = 5      # 覆盖五种路线模式，全部由DiT生成且不允许专家接管
FULL_MISSION_VALIDATE_DDIM_STEPS = 25     # 闭环验证DDIM步数
EARLY_STOP_MIN_EPOCHS = 300               # 至少训练到该轮次后才允许闭环早停
EARLY_STOP_PATIENCE = 3                   # 连续若干次纯DiT闭环验证无提升后停止
LR_WARMUP_EPOCHS = 5                      # 学习率线性预热轮数，之后余弦衰减

# ===== 文件路径和版本 =====
# 滚动版本使用独立目录，避免误加载旧版 32 小时状态轨迹模型和训练数据。
MODEL_SAVE_DIR = "./checkpoints_rolling_v7"       # v7独立保存，保留全部v6模型与结果
LEGACY_MODEL_SAVE_DIR = "./checkpoints"           # 旧版 32 小时模型目录，仅用于保留和对照
DATA_DIR = "./data/rolling_trajectories_v7"       # v7物理参数与动态时限训练数据
LOG_DIR = "./logs/rolling_v7"                     # v7 TensorBoard日志目录

MODEL_VERSION = "rolling-dit-v7"  # 写入 checkpoint 的模型版本标识


def cosine_beta_schedule(T, s=0.008):
    """生成余弦 DDPM 噪声调度，并限制每一步 beta 的数值范围。"""
    steps = T + 1
    x = [math.cos((t / T + s) / (1 + s) * math.pi / 2) ** 2 for t in range(steps)]
    alphas_cumprod = [v / x[0] for v in x]
    betas = [1 - alphas_cumprod[t + 1] / alphas_cumprod[t] for t in range(T)]
    return [max(min(b, 0.999), 1e-5) for b in betas]


def model_metadata():
    """返回写入 checkpoint 的关键结构元数据，用于加载时检查配置兼容性。"""
    return {
        "model_version": MODEL_VERSION,
        "data_version": DATA_VERSION,
        "plan_hours": PLAN_HOURS,
        "wind_context_hours": WIND_CONTEXT_HOURS,
        "trajectory_steps": TRAJ_LEN,
        "execute_steps": EXECUTE_STEPS,
        "action_dim": ACTION_DIM,
        "state_dim": STATE_DIM,
        "condition_dim": COND_DIM,
        "route_modes": list(ROUTE_MODE_NAMES),
        "wind_frames": WIND_NUM_FRAMES,
        "trajectory_representation": TRAJECTORY_REPRESENTATION,
        "prediction_type": PREDICTION_TYPE,
        "boundary_margin_km": BOUNDARY_MARGIN_KM,
        "boundary_shield_margin_km": BOUNDARY_SHIELD_MARGIN_KM,
        "expert_ground_speed": EXPERT_GROUND_SPEED,
        "windows_per_mission": WINDOWS_PER_MISSION,
        "hard_sample_ratio": HARD_SAMPLE_RATIO,
        "hard_route_range_km": [HARD_ROUTE_MIN_KM, HARD_ROUTE_MAX_KM],
        "expert_route_offsets_km": list(EXPERT_ROUTE_OFFSETS_KM),
        "expert_waypoint_margin_km": EXPERT_WAYPOINT_MARGIN_KM,
        "expert_speed_ratios": list(EXPERT_SPEED_RATIOS),
        "expert_eval_steps": EXPERT_EVAL_STEPS,
        "recovery_window_ratio": RECOVERY_WINDOW_RATIO,
        "d_model": D_MODEL,
        "dit_layers": DIT_LAYERS,
        "dit_heads": DIT_HEADS,
        "wind_patch_grid": [WIND_PATCHES_Y, WIND_PATCHES_X],
        "dt_seconds": DT,
        "max_mission_steps": MAX_MISSION_STEPS,
        "min_mission_hours": MIN_MISSION_HOURS,
        "max_mission_hours": MAX_MISSION_HOURS,
        "max_speed_mps": MAX_SPEED,
        "max_power_kw": MAX_POWER,
        "propulsion_power_coefficient": PROPULSION_POWER_COEFF,
        "wind_max_speed_mps": WIND_MAX_SPEED,
    }
