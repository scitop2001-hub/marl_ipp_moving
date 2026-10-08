"""
策略网络 — 论文 Section IV-E
完整的注意力策略网络: 编码器 + 解码器
π(G_t, ψ_{0:t-1}, B̃) -> 动作概率分布 + 值函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import Encoder
from .decoder import Decoder, PlanningStateEncoder


class PolicyNetwork(nn.Module):
    """
    策略网络 (Actor-Critic)

    输入:
        - features: (batch, L, feature_dim) 协调图特征矩阵
        - path_history: (batch, seq_len, action_dim) 已执行路径
        - remaining_budget: (batch, 1) 剩余预算
        - budget_mask: (batch, L) 预算掩码

    输出:
        - action_probs: (batch, L) 动作概率分布
        - value: (batch, 1) 状态值函数
        - action_logits: (batch, L) 原始logits (用于PPO)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 编码器
        self.encoder = Encoder(config)

        # 规划状态编码器
        self.planning_state_encoder = PlanningStateEncoder(config)

        # 解码器
        self.decoder = Decoder(config)

    def forward(self, features, path_history, remaining_budget, budget_mask):
        """
        前向传播

        参数:
            features: (batch, L, feature_dim) 特征矩阵
            path_history: (batch, seq_len, action_dim) 路径历史
            remaining_budget: (batch, 1) 剩余预算
            budget_mask: (batch, L) 预算掩码

        返回:
            action_logits: (batch, L) 动作logits
            action_probs: (batch, L) 动作概率
            value: (batch, 1) 值函数
        """
        # 编码协调图
        context = self.encoder(features)  # (batch, L, hidden)

        # 编码规划状态
        planning_state = self.planning_state_encoder(path_history, remaining_budget)

        # 解码: 动作分布 + 值函数
        action_logits, value = self.decoder(context, planning_state, budget_mask)

        # 计算动作概率 (处理全被mask的情况)
        # 如果所有动作都被mask, 放宽到均匀分布
        all_masked = (budget_mask.sum(dim=1, keepdim=True) == 0)  # (batch, 1)
        if all_masked.any():
            # 对全被mask的样本, 重置logits为0 (均匀分布)
            mask_2d = all_masked.expand_as(action_logits)
            action_logits = torch.where(mask_2d, torch.zeros_like(action_logits), action_logits)

        action_probs = F.softmax(action_logits, dim=-1)

        return action_logits, action_probs, value

    def act(self, features, path_history, remaining_budget, budget_mask, deterministic=False):
        """
        选择动作 (用于部署)

        返回:
            action_idx: (batch,) 选中的动作索引
            log_prob: (batch,) 对数概率
            value: (batch,) 值函数
        """
        action_logits, action_probs, value = self.forward(
            features, path_history, remaining_budget, budget_mask
        )

        if deterministic:
            action_idx = action_probs.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_idx = dist.sample()

        log_prob = torch.log(action_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) + 1e-8)

        return action_idx, log_prob, value.squeeze(-1)

    def evaluate(self, features, path_history, remaining_budget, budget_mask, action_idx):
        """
        评估给定动作 (用于PPO更新)

        返回:
            log_prob: (batch,) 对数概率
            entropy: (batch,) 熵
            value: (batch,) 值函数
        """
        action_logits, action_probs, value = self.forward(
            features, path_history, remaining_budget, budget_mask
        )

        log_prob = torch.log(action_probs.gather(1, action_idx.unsqueeze(1)).squeeze(1) + 1e-8)
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1)

        return log_prob, entropy, value.squeeze(-1)
