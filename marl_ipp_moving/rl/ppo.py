"""
PPO 训练器 — 论文 Section IV-E, V-A
Proximal Policy Optimization (PPO) with CTDE paradigm

关键超参数:
- clip epsilon = 0.2
- Adam optimizer, lr = 1e-4, decay 0.96 every 512 steps
- 8 epochs per update, batch size = 1024
- 36 parallel environments
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from typing import Dict, List
from .buffer import RolloutBuffer, Transition


class PPOTrainer:
    """PPO训练器"""

    def __init__(self, config, policy_network, device='cpu'):
        self.config = config
        self.device = device
        self.policy = policy_network.to(device)

        # 优化器
        self.optimizer = Adam(
            self.policy.parameters(),
            lr=config.ppo.learning_rate
        )

        # 学习率调度器
        self.scheduler = StepLR(
            self.optimizer,
            step_size=config.ppo.lr_decay_step,
            gamma=config.ppo.lr_decay_factor
        )

        # PPO超参数
        self.clip_epsilon = config.ppo.clip_epsilon
        self.ppo_epochs = config.ppo.ppo_epochs
        self.batch_size = config.ppo.batch_size
        self.entropy_coef = config.ppo.entropy_coef
        self.value_coef = config.ppo.value_coef
        self.max_grad_norm = config.ppo.max_grad_norm

        # 日志
        self.total_steps = 0
        self.training_stats = []

    def update(self, buffer: RolloutBuffer):
        """PPO更新"""
        # 计算优势函数
        buffer.compute_advantages(
            gamma=self.config.ppo.gamma,
            gae_lambda=self.config.ppo.gae_lambda
        )

        stats = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'total_loss': [], 'clip_frac': []}

        for epoch in range(self.ppo_epochs):
            for batch in buffer.get_batch(self.batch_size):
                # 移到设备
                features = batch['features'].to(self.device)
                path_history = batch['path_history'].to(self.device)
                remaining_budget = batch['remaining_budget'].to(self.device)
                budget_mask = batch['budget_mask'].to(self.device)
                action_idx = batch['action_idx'].to(self.device)
                old_log_prob = batch['old_log_prob'].to(self.device)
                old_value = batch['old_value'].to(self.device)
                advantages = batch['advantage'].to(self.device)
                returns = batch['return'].to(self.device)

                # 确保维度正确
                if features.dim() == 3:
                    features = features  # (batch, L, feature_dim)
                if path_history.dim() == 2:
                    path_history = path_history.unsqueeze(0)  # 添加seq维
                if budget_mask.dim() == 2:
                    budget_mask = budget_mask

                # 策略网络前向
                log_prob, entropy, value = self.policy.evaluate(
                    features, path_history, remaining_budget, budget_mask, action_idx
                )

                # PPO policy loss (clipped surrogate)
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = F.mse_loss(value, returns)

                # Entropy loss
                entropy_loss = -entropy.mean()

                # Total loss
                total_loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                self.total_steps += 1

                # 统计
                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(entropy.mean().item())
                stats['total_loss'].append(total_loss.item())
                stats['clip_frac'].append(clip_frac)

        # 汇总统计
        avg_stats = {k: np.mean(v) if v else 0.0 for k, v in stats.items()}
        self.training_stats.append(avg_stats)

        return avg_stats

    def save(self, path):
        """保存模型"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_steps': self.total_steps,
            'training_stats': self.training_stats,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.total_steps = checkpoint['total_steps']
        self.training_stats = checkpoint.get('training_stats', [])
        print(f"Model loaded from {path}")
