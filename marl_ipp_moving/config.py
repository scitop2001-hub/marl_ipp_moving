"""
全局配置文件 — 移动目标版
在原论文基础上创新: 多机器人移动目标信息路径规划
新增: 移动目标运动模型、时空GP、运动预测网络、预测奖励
"""
from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """环境配置"""
    env_size: float = 1.0

    # 建筑
    train_num_buildings: int = 12
    train_windows_range: tuple = (150, 200)  # 移动目标数量略少(计算量大)

    # 传感器
    sensor_fov: float = 90.0
    sensor_range: float = 0.24
    sensor_view_dirs: tuple = (0.0, 1.5708, 3.1416, 4.7124)

    # 通信
    comm_range_train: float = 0.3
    comm_range_test: float = 0.1
    comm_range_global: float = 1e6

    # 碰撞
    collision_dist: float = 0.05
    local_region_size: float = 0.15

    # 占用地图
    voxel_size: float = 0.02
    target_normalize_const: float = 10.0

    # === 新增: 移动目标配置 ===
    num_moving_targets: int = 30       # 移动目标数量
    target_speed_range: tuple = (0.005, 0.02)  # 目标移动速度 (每步位移)
    target_motion_type: str = "random_walk"    # random_walk / constant_velocity / hybrid
    target_boundary: str = "bounce"            # bounce / wrap / respawn
    target_id_enabled: bool = True             # 启用目标ID关联


@dataclass
class GPConfig:
    """高斯过程配置"""
    kernel_type: str = "matern12"
    length_scale: float = 0.1
    sigma_f: float = 1.0
    sigma_n: float = 0.01
    max_train_points: int = 500

    # === 新增: 时空GP配置 ===
    time_length_scale: float = 5.0      # 时间核长度尺度 (步数)
    spatiotemporal_kernel: bool = True  # 使用时空核


@dataclass
class PredictionConfig:
    """=== 新增: 运动预测网络配置 ==="""
    hidden_dim: int = 128
    num_layers: int = 2
    prediction_horizon: int = 5         # 预测未来多少步
    input_dim: int = 6                  # [x, y, z, vx, vy, vz] 或 [x, y, z, t, target_id]
    output_dim: int = 3                 # 预测位置 [x, y, z]
    learning_rate: float = 1e-3
    history_length: int = 10            # 用最近多少步的历史观测做预测


@dataclass
class GraphConfig:
    """协调图配置"""
    num_candidates: int = 80
    action_dim: int = 4

    # === 修改: 特征维度增加 (加了预测特征) ===
    # 原版10维 -> 移动目标版13维
    # [x,y,z,d, u_util, P_util, u_comm, P_comm, budget_mask, dist,
    #  pred_density, pred_uncertainty, target_velocity_est]
    feature_dim: int = 13


@dataclass
class NetworkConfig:
    """策略网络配置"""
    encoder_hidden: int = 128
    encoder_num_heads: int = 4
    encoder_num_layers: int = 2
    encoder_dropout: float = 0.0

    decoder_hidden: int = 128
    decoder_num_heads: int = 4
    decoder_num_layers: int = 2

    value_hidden: int = 256

    feature_dim: int = 13   # 与 GraphConfig 同步
    action_dim: int = 4
    state_dim: int = 64


@dataclass
class PPOConfig:
    """PPO训练配置"""
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 8
    batch_size: int = 1024
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    learning_rate: float = 1e-4
    lr_decay_factor: float = 0.96
    lr_decay_step: int = 512

    num_parallel_envs: int = 36
    max_episodes: int = 10000
    max_timesteps: int = 256
    total_interactions: int = 150000  # 移动目标更难收敛, 加大交互数

    # 奖励权重
    alpha: float = 20.0          # 探索奖励
    beta: float = 1.0            # 效用奖励
    gamma_comm: float = 1.0      # 通信奖励
    delta: float = 0.02
    # === 新增: 预测奖励权重 ===
    eta: float = 5.0             # 预测奖励权重 (鼓励去目标预测位置)

    budget_range: tuple = (7.0, 9.0)


@dataclass
class EvalConfig:
    """评估配置"""
    num_test_envs: int = 25
    num_trials_per_env: int = 10
    num_robots: int = 3
    eval_budget: float = 10.0

    scalability_robots: list = field(default_factory=lambda: [16, 32, 48, 64])
    scalability_env_scales: list = field(default_factory=lambda: [3, 8, 16])


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    device: str = "cuda"
    seed: int = 42

    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    result_dir: str = "results"


def get_config():
    return Config()
