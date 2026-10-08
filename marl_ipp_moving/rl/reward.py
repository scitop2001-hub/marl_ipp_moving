"""
奖励函数 — 移动目标版
r_t = α·r_e + β·r_u + γ·r_c + η·r_p

新增 r_p (预测奖励): 奖励机器人前往目标预测位置附近的动作
"""
import numpy as np


class RewardCalculator:
    """移动目标版奖励计算器"""

    def __init__(self, config):
        self.config = config
        self.alpha = config.ppo.alpha
        self.beta = config.ppo.beta
        self.gamma_comm = config.ppo.gamma_comm
        self.eta = config.ppo.eta  # 预测奖励权重
        self.delta = config.ppo.delta

    def compute(self, num_new_targets, exploration_reward, comm_reward,
                prediction_reward=0.0):
        """
        计算总奖励 (增加了预测奖励项)

        返回:
            total_reward, components_dict
        """
        r_e = exploration_reward
        r_u = num_new_targets * self.delta
        r_c = comm_reward
        r_p = prediction_reward

        total = self.alpha * r_e + self.beta * r_u + self.gamma_comm * r_c + self.eta * r_p

        return total, {
            'r_explore': r_e,
            'r_utility': r_u,
            'r_comm': r_c,
            'r_predict': r_p,
            'r_total': total,
        }

    def compute_exploration_reward(self, utility_gp, candidate_actions, action_idx, current_time=0):
        """探索奖励 r_e"""
        if len(candidate_actions) == 0:
            return 0.0
        return utility_gp.get_trace_reduction(candidate_actions, action_idx, current_time)

    def compute_comm_reward(self, comm_gp, candidate_actions, action_idx):
        """通信奖励 r_c"""
        if len(candidate_actions) == 0:
            return 0.0
        return comm_gp.get_trace_reduction(candidate_actions, action_idx)

    def compute_prediction_reward(self, candidate_actions, action_idx,
                                   target_histories, prediction_manager):
        """
        预测奖励 r_p (新增)
        如果选中的动作位置接近目标预测位置, 给予奖励
        """
        if len(candidate_actions) == 0 or len(target_histories) == 0:
            return 0.0

        # 获取目标预测密度
        density, _ = prediction_manager.predict_target_density(
            target_histories, candidate_actions[:, :3],
            horizon=self.config.prediction.prediction_horizon
        )

        # 预测奖励 = 选中动作位置的预测密度 (归一化)
        max_density = max(np.max(density), 1e-6)
        return density[action_idx] / max_density
