"""
训练脚本 — 移动目标版
联合训练: 策略网络(PPO) + 运动预测网络(监督学习)

训练流程:
1. 创建移动目标环境
2. 每个episode:
   a. 每步先移动目标, 再让机器人决策
   b. 机器人构建协调图(含预测特征) -> 策略网络选动作 -> 执行 -> 获得奖励
   c. 同时收集目标轨迹数据, 训练运动预测网络
3. PPO更新策略网络 + 监督更新运动预测网络
"""
import os
import sys
import time
import numpy as np
import torch
from tqdm import tqdm

from config import get_config
from env.multi_robot_env import MultiRobotMovingEnv
from gp.communication_gp import UtilityGP, CommunicationGP
from graph.coordination_graph import CoordinationGraph
from network.policy_network import PolicyNetwork
from network.motion_predictor import PredictionManager
from rl.ppo import PPOTrainer
from rl.buffer import RolloutBuffer, Transition
from rl.reward import RewardCalculator


def collect_episode(env, policy, utility_gp, comm_gps, coord_graph,
                    prediction_manager, reward_calculator, device,
                    deterministic=False):
    """收集一个episode的经验 (移动目标版)"""
    buffer = RolloutBuffer()
    num_robots = env.num_robots

    budget = np.random.uniform(*env.config.ppo.budget_range)
    env.reset(budget=budget)

    utility_gp.reset()
    for gp in comm_gps:
        gp.reset()
    prediction_manager.reset_data()

    episode_rewards = []
    total_new_targets = 0
    total_tracked = 0
    step_count = 0

    while not env.all_done():
        # === 关键: 先移动所有目标 ===
        env.step_targets()

        # 通信
        neighbors = env._get_comm_neighbors()
        env._exchange_paths(neighbors)

        # 获取所有目标历史轨迹 (用于运动预测)
        target_histories = env.target_env.get_target_histories(
            env.config.prediction.history_length
        )

        for robot_idx in range(num_robots):
            if env.robot_budgets[robot_idx] <= 0.001:
                continue

            # 候选动作
            candidates = env.get_candidate_actions(
                robot_idx, env.config.graph.num_candidates
            )

            # 协调图 (含预测特征)
            features, adj, budget_mask, edge_costs = coord_graph.build_batch(
                robot_idx, candidates, utility_gp, comm_gps[robot_idx],
                prediction_manager, target_histories,
                env.robot_poses[robot_idx], env.robot_budgets[robot_idx],
                current_time=env.t
            )

            # 路径历史
            path_history = np.array(env.robot_paths[robot_idx])
            path_tensor = torch.FloatTensor(path_history).unsqueeze(0).to(device)

            # 预算
            rem_budget = torch.FloatTensor([[env.robot_budgets[robot_idx]]]).to(device)

            # 策略网络选动作
            with torch.no_grad():
                action_idx, log_prob, value = policy.act(
                    features, path_tensor, rem_budget, budget_mask,
                    deterministic=deterministic
                )

            action_idx_np = action_idx.item()
            selected_action = candidates[action_idx_np]

            # 奖励计算 (含预测奖励)
            r_e = reward_calculator.compute_exploration_reward(
                utility_gp, candidates, action_idx_np, current_time=env.t
            )
            r_c = reward_calculator.compute_comm_reward(
                comm_gps[robot_idx], candidates, action_idx_np
            )
            r_p = reward_calculator.compute_prediction_reward(
                candidates, action_idx_np, target_histories, prediction_manager
            )

            # 执行动作
            num_new, num_tracked, move_cost, done = env.step(robot_idx, selected_action)

            r_u = num_new
            total_reward, reward_components = reward_calculator.compute(
                r_u, r_e, r_c, r_p
            )

            # 更新GP (时空)
            utility_gp.update(
                [[selected_action[0], selected_action[1], selected_action[2], env.t]],
                [num_new]
            )
            comm_gps[robot_idx].update(env.get_comm_gp_data(robot_idx))

            # 存储transition
            trans = Transition()
            trans.features = features.squeeze(0).cpu().numpy()
            trans.path_history = path_history
            trans.remaining_budget = env.robot_budgets[robot_idx] + move_cost
            trans.budget_mask = budget_mask.squeeze(0).cpu().numpy()
            trans.action_idx = action_idx_np
            trans.log_prob = log_prob.item()
            trans.value = value.item()
            trans.reward = total_reward
            trans.done = done or env.all_done()
            trans.candidate_actions = candidates

            buffer.add(trans)
            episode_rewards.append(total_reward)
            total_new_targets += num_new
            total_tracked += num_tracked
            step_count += 1

            if env.all_done():
                break

    # 训练运动预测网络
    pred_losses = []
    for _ in range(10):  # 每episode训练10步
        loss = prediction_manager.train_step(batch_size=64)
        if loss > 0:
            pred_losses.append(loss)

    episode_info = {
        'total_reward': sum(episode_rewards),
        'mean_reward': np.mean(episode_rewards) if episode_rewards else 0,
        'num_steps': step_count,
        'num_new_targets': total_new_targets,
        'num_tracked': total_tracked,
        'discovered_ratio': env.target_env.get_discovered_ratio(env.global_discovered_ids),
        'tracking_accuracy': env.target_env.get_tracking_accuracy(),
        'total_targets': env.target_env.get_total_targets(),
        'budget': budget,
        'pred_loss': np.mean(pred_losses) if pred_losses else 0,
    }

    return buffer, episode_info


def train(config=None, resume=False, checkpoint_path=None,
          prediction_checkpoint=None):
    """主训练函数"""
    if config is None:
        config = get_config()

    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Training device: {device}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 策略网络
    policy = PolicyNetwork(config)
    print(f"Policy network parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # 运动预测网络
    prediction_manager = PredictionManager(config, device=device)
    print(f"Motion predictor parameters: {sum(p.numel() for p in prediction_manager.predictor.parameters()):,}")

    # PPO训练器
    trainer = PPOTrainer(config, policy, device=device)

    # 加载检查点
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        trainer.load(checkpoint_path)
    if resume and prediction_checkpoint and os.path.exists(prediction_checkpoint):
        prediction_manager.load(prediction_checkpoint)

    # 模块
    utility_gp = UtilityGP(config)
    comm_gps = [CommunicationGP(config) for _ in range(3)]
    coord_graph = CoordinationGraph(config)
    reward_calculator = RewardCalculator(config)

    # 训练循环
    total_interactions = 0
    target_interactions = config.ppo.total_interactions

    log_file = os.path.join(config.log_dir, 'train_log.txt')
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    pbar = tqdm(total=target_interactions, desc="Training (Moving Targets)")
    episode_num = 0

    while total_interactions < target_interactions:
        episode_num += 1

        env = MultiRobotMovingEnv(
            config, num_robots=3, is_training=True,
            seed=config.seed + episode_num
        )

        buffer, info = collect_episode(
            env, policy, utility_gp, comm_gps, coord_graph,
            prediction_manager, reward_calculator, device,
            deterministic=False
        )

        total_interactions += len(buffer)
        pbar.update(len(buffer))

        # PPO更新
        if len(buffer) > 0:
            stats = trainer.update(buffer)

            if episode_num % 10 == 0:
                log_msg = (
                    f"Ep {episode_num} | "
                    f"Steps {total_interactions}/{target_interactions} | "
                    f"Disc {info['discovered_ratio']:.2%} | "
                    f"Track {info['tracking_accuracy']:.2%} | "
                    f"New {info['num_new_targets']} | "
                    f"Reward {info['mean_reward']:.4f} | "
                    f"PredLoss {info['pred_loss']:.4f} | "
                    f"PolicyLoss {stats['policy_loss']:.4f} | "
                    f"ValueLoss {stats['value_loss']:.4f}"
                )
                print(log_msg)
                with open(log_file, 'a') as f:
                    f.write(log_msg + '\n')

        # 保存检查点
        if episode_num % 100 == 0:
            ckpt = os.path.join(config.checkpoint_dir, f'policy_ep{episode_num}.pt')
            trainer.save(ckpt)
            pred_ckpt = os.path.join(config.checkpoint_dir, f'predictor_ep{episode_num}.pt')
            prediction_manager.save(pred_ckpt)

        pbar.set_postfix({
            'disc': f"{info['discovered_ratio']:.1%}",
            'track': f"{info['tracking_accuracy']:.1%}",
        })

    pbar.close()

    final_path = os.path.join(config.checkpoint_dir, 'policy_final.pt')
    trainer.save(final_path)
    pred_final = os.path.join(config.checkpoint_dir, 'predictor_final.pt')
    prediction_manager.save(pred_final)

    print(f"\nTraining complete! Total interactions: {total_interactions}")
    print(f"Policy saved: {final_path}")
    print(f"Predictor saved: {pred_final}")

    return trainer


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='移动目标多机器人IPP训练')
    parser.add_argument('--resume', action='store_true', help='恢复训练')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--pred_checkpoint', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_targets', type=int, default=None,
                        help='覆盖移动目标数量')
    parser.add_argument('--target_speed', type=float, default=None,
                        help='覆盖目标移动速度')
    args = parser.parse_args()

    config = get_config()
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device
    config.seed = args.seed

    if args.num_targets is not None:
        config.env.num_moving_targets = args.num_targets
    if args.target_speed is not None:
        config.env.target_speed_range = (args.target_speed * 0.5, args.target_speed * 1.5)

    print(f"Config: {config.env.num_moving_targets} moving targets, "
          f"speed {config.env.target_speed_range}, device={config.device}")

    train(
        config=config,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        prediction_checkpoint=args.pred_checkpoint
    )
