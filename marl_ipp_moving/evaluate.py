"""
评估脚本 — 移动目标版
评估指标:
  1. 发现目标百分比 (discovered ratio)
  2. 跟踪精度 (tracking accuracy)
  3. 每步规划时间
"""
import os
import time
import json
import numpy as np
import torch
from tqdm import tqdm

from config import get_config
from env.multi_robot_env import MultiRobotMovingEnv
from gp.communication_gp import UtilityGP, CommunicationGP
from graph.coordination_graph import CoordinationGraph
from network.policy_network import PolicyNetwork
from network.motion_predictor import PredictionManager
from rl.reward import RewardCalculator
from baselines.random_agent import RandomAgent


def evaluate_policy(config, policy_path, predictor_path=None,
                    num_test_envs=25, num_trials=10, num_robots=3,
                    budget=10.0, comm_range=0.1, device='cpu'):
    """评估训练好的策略"""
    print(f"\n=== Evaluating Our Approach (Moving Targets) ===")

    policy = PolicyNetwork(config).to(device)
    ckpt = torch.load(policy_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt['policy_state_dict'])
    policy.eval()

    pred_manager = PredictionManager(config, device=device)
    if predictor_path and os.path.exists(predictor_path):
        pred_manager.load(predictor_path)

    utility_gp = UtilityGP(config)
    comm_gps = [CommunicationGP(config) for _ in range(num_robots)]
    coord_graph = CoordinationGraph(config)

    results = []

    for env_idx in range(num_test_envs):
        for trial_idx in range(num_trials):
            env = MultiRobotMovingEnv(
                config, num_robots=num_robots, is_training=False,
                seed=2000 + env_idx
            )
            env.reset(budget=budget, comm_range=comm_range)

            utility_gp.reset()
            for gp in comm_gps:
                gp.reset()

            step_times = []

            while not env.all_done():
                env.step_targets()
                neighbors = env._get_comm_neighbors()
                env._exchange_paths(neighbors)

                target_histories = env.target_env.get_target_histories(
                    config.prediction.history_length
                )

                for robot_idx in range(num_robots):
                    if env.robot_budgets[robot_idx] <= 0.001:
                        continue

                    start_time = time.time()
                    candidates = env.get_candidate_actions(
                        robot_idx, config.graph.num_candidates
                    )
                    features, adj, budget_mask, _ = coord_graph.build_batch(
                        robot_idx, candidates, utility_gp, comm_gps[robot_idx],
                        pred_manager, target_histories,
                        env.robot_poses[robot_idx], env.robot_budgets[robot_idx],
                        current_time=env.t
                    )
                    path_history = np.array(env.robot_paths[robot_idx])
                    path_tensor = torch.FloatTensor(path_history).unsqueeze(0).to(device)
                    rem_budget = torch.FloatTensor([[env.robot_budgets[robot_idx]]]).to(device)

                    with torch.no_grad():
                        action_idx, _, _ = policy.act(
                            features, path_tensor, rem_budget, budget_mask,
                            deterministic=True
                        )

                    step_times.append(time.time() - start_time)

                    selected = candidates[action_idx.item()]
                    num_new, num_tracked, _, _ = env.step(robot_idx, selected)

                    utility_gp.update(
                        [[selected[0], selected[1], selected[2], env.t]], [num_new]
                    )
                    comm_gps[robot_idx].update(env.get_comm_gp_data(robot_idx))

                    if env.all_done():
                        break

            results.append({
                'env_idx': env_idx,
                'trial_idx': trial_idx,
                'discovered_ratio': env.target_env.get_discovered_ratio(env.global_discovered_ids),
                'tracking_accuracy': env.target_env.get_tracking_accuracy(),
                'num_discovered': len(env.global_discovered_ids),
                'total_targets': env.target_env.get_total_targets(),
                'avg_step_time': np.mean(step_times) if step_times else 0,
                'num_steps': env.t,
            })

    ratios = [r['discovered_ratio'] for r in results]
    tracking = [r['tracking_accuracy'] for r in results]
    times = [r['avg_step_time'] for r in results]

    print(f"\nResults (Our Approach - Moving Targets):")
    print(f"  Discovered: {np.mean(ratios):.2%} +/- {np.std(ratios):.2%}")
    print(f"  Tracking:   {np.mean(tracking):.2%} +/- {np.std(tracking):.2%}")
    print(f"  Time/step:  {np.mean(times):.4f}s")

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, required=True)
    parser.add_argument('--predictor', type=str, default=None)
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    config = get_config()
    if args.device == 'auto':
        config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        config.device = args.device

    evaluate_policy(config, args.policy, args.predictor, device=config.device)
