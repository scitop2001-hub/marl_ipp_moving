"""
快速测试脚本 — 移动目标版
运行: python quick_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

print("=" * 60)
print("快速测试: 移动目标多机器人IPP (创新版)")
print("=" * 60)
print()

# 1. 导入
print("[1/7] 测试模块导入...")
from config import get_config
from env.multi_robot_env import MultiRobotMovingEnv
from env.moving_target import MovingTarget
from gp.communication_gp import UtilityGP, CommunicationGP
from gp.spatiotemporal_gp import SpatiotemporalGP, spatiotemporal_kernel
from graph.coordination_graph import CoordinationGraph
from network.policy_network import PolicyNetwork
from network.motion_predictor import MotionPredictor, PredictionManager
from rl.ppo import PPOTrainer
from rl.reward import RewardCalculator
from rl.buffer import RolloutBuffer
print("  OK")
print()

# 2. 环境
print("[2/7] 测试移动目标环境...")
config = get_config()
config.device = 'cpu'
env = MultiRobotMovingEnv(config, num_robots=3, is_training=True, seed=42)
env.reset(budget=7.5)
print(f"  建筑: {len(env.target_env.building_occupancy.grid)} voxels")
print(f"  移动目标: {env.target_env.get_total_targets()}")
print(f"  机器人: {env.num_robots}")

# 测试目标移动
initial_pos = env.target_env.targets[0].pos.copy()
env.step_targets()
new_pos = env.target_env.targets[0].pos.copy()
print(f"  目标0移动: {initial_pos[:2]} -> {new_pos[:2]}, 位移={np.linalg.norm(new_pos-initial_pos):.4f}")
print()

# 3. 时空GP
print("[3/7] 测试时空GP...")
X = np.random.rand(10, 4)  # [x,y,z,t]
y = np.random.rand(10)
gp = SpatiotemporalGP(space_ls=0.1, time_ls=5.0)
gp.set_data(X, y)
mean, var = gp.predict(np.random.rand(5, 3), current_time=5)
print(f"  时空GP预测: mean={mean.shape}, var={var.shape}")
print()

# 4. 运动预测网络
print("[4/7] 测试运动预测网络...")
predictor = MotionPredictor(config)
history = torch.randn(4, 10, 3)  # 4个目标, 10步历史, 3D位置
pred_pos, pred_vel = predictor(history)
print(f"  预测位置: {pred_pos.shape}")
print(f"  估计速度: {pred_vel.shape}")

# 多步预测
future = predictor.predict_future(history, horizon=5)
print(f"  多步预测: {future.shape}")
print()

# 5. 策略网络
print("[5/7] 测试策略网络(13维特征)...")
policy = PolicyNetwork(config)
params = sum(p.numel() for p in policy.parameters())
print(f"  参数量: {params:,}")
L = config.graph.num_candidates
features = torch.randn(1, L, 13)
path = torch.randn(1, 5, 4)
budget = torch.tensor([[5.0]])
mask = torch.ones(1, L)
logits, probs, value = policy(features, path, budget, mask)
print(f"  动作概率: {probs.shape}, sum={probs.sum().item():.4f}")
print()

# 6. 完整时间步
print("[6/7] 测试完整时间步(含移动目标)...")
utility_gp = UtilityGP(config)
comm_gps = [CommunicationGP(config) for _ in range(3)]
coord_graph = CoordinationGraph(config)
reward_calc = RewardCalculator(config)
pred_manager = PredictionManager(config, device='cpu')

env2 = MultiRobotMovingEnv(config, num_robots=3, is_training=True, seed=100)
env2.reset(budget=7.5)
utility_gp.reset()
for gp in comm_gps:
    gp.reset()

# 模拟几步让目标有历史轨迹
for _ in range(15):
    env2.step_targets()

target_histories = env2.target_env.get_target_histories(config.prediction.history_length)
candidates = env2.get_candidate_actions(0, L)
features, adj, bmask, costs = coord_graph.build_batch(
    0, candidates, utility_gp, comm_gps[0],
    pred_manager, target_histories,
    env2.robot_poses[0], env2.robot_budgets[0], current_time=env2.t
)
print(f"  特征矩阵: {features.shape} (13维)")
print(f"  预算掩码: {bmask.sum().item():.0f}/{L}")

path_hist = np.array(env2.robot_paths[0])
path_t = torch.FloatTensor(path_hist).unsqueeze(0)
budget_t = torch.FloatTensor([[env2.robot_budgets[0]]])

action_idx, log_prob, val = policy.act(features, path_t, budget_t, bmask)
selected = candidates[action_idx.item()]
num_new, num_tracked, cost, done = env2.step(0, selected)
print(f"  选中动作: {action_idx.item()}")
print(f"  新发现: {num_new}, 跟踪: {num_tracked}")
print()

# 7. 奖励(含预测奖励)
print("[7/7] 测试4项奖励...")
r_e = reward_calc.compute_exploration_reward(utility_gp, candidates, action_idx.item(), env2.t)
r_c = reward_calc.compute_comm_reward(comm_gps[0], candidates, action_idx.item())
r_p = reward_calc.compute_prediction_reward(candidates, action_idx.item(), target_histories, pred_manager)
total, comps = reward_calc.compute(num_new, r_e, r_c, r_p)
print(f"  r_e={r_e:.4f}, r_u={num_new*config.ppo.delta:.4f}, r_c={r_c:.4f}, r_p={r_p:.4f}")
print(f"  总奖励: {total:.4f}")
print()

print("=" * 60)
print("所有测试通过! 移动目标版系统就绪。")
print("=" * 60)
print()
print("下一步:")
print("  1. python train.py --device cuda    # GPU训练")
print("  2. python train.py --device cpu     # CPU训练")
print("  3. python train.py --num_targets 50 # 50个移动目标")
