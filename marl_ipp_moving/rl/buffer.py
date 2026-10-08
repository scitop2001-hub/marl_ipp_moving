"""
经验缓冲区 — 存储训练数据用于PPO on-policy更新
每个transition包含: 特征矩阵, 路径历史, 预算, 掩码, 动作, 奖励, 等
"""
import numpy as np
import torch
from typing import List, Dict


class Transition:
    """单个时间步的转换数据"""

    def __init__(self):
        self.features = None        # (L, feature_dim) 协调图特征
        self.path_history = None    # (seq_len, action_dim) 路径历史
        self.remaining_budget = None  # float 剩余预算
        self.budget_mask = None     # (L,) 预算掩码
        self.action_idx = None      # int 选中的动作索引
        self.log_prob = None        # float 对数概率
        self.value = None           # float 值函数估计
        self.reward = None          # float 奖励
        self.done = None            # bool 是否结束
        self.candidate_actions = None  # (L, 4) 候选动作 (用于在环境中执行)

    def to_dict(self):
        return {
            'features': self.features,
            'path_history': self.path_history,
            'remaining_budget': self.remaining_budget,
            'budget_mask': self.budget_mask,
            'action_idx': self.action_idx,
            'log_prob': self.log_prob,
            'value': self.value,
            'reward': self.reward,
            'done': self.done,
        }


class RolloutBuffer:
    """经验缓冲区"""

    def __init__(self):
        self.transitions: List[Transition] = []
        self.advantages = []
        self.returns = []

    def add(self, transition: Transition):
        self.transitions.append(transition)

    def compute_advantages(self, gamma=0.99, gae_lambda=0.95):
        """计算GAE优势函数和回报"""
        n = len(self.transitions)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        last_value = 0.0

        for t in reversed(range(n)):
            trans = self.transitions[t]
            if trans.done:
                last_gae = 0.0
                last_value = 0.0

            delta = trans.reward + gamma * last_value - trans.value
            last_gae = delta + gamma * gae_lambda * last_gae
            self.advantages[t] = last_gae
            self.returns[t] = last_gae + trans.value
            last_value = trans.value

        # 标准化优势函数
        if len(self.advantages) > 1:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_batch(self, batch_size):
        """获取一个mini-batch"""
        n = len(self.transitions)
        indices = np.random.permutation(n)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            batch = {
                'features': [],
                'path_history': [],
                'remaining_budget': [],
                'budget_mask': [],
                'action_idx': [],
                'old_log_prob': [],
                'old_value': [],
                'advantage': [],
                'return': [],
            }

            for idx in batch_indices:
                trans = self.transitions[idx]
                batch['features'].append(trans.features)
                batch['path_history'].append(trans.path_history)
                batch['remaining_budget'].append([trans.remaining_budget])
                batch['budget_mask'].append(trans.budget_mask)
                batch['action_idx'].append(trans.action_idx)
                batch['old_log_prob'].append(trans.log_prob)
                batch['old_value'].append(trans.value)
                batch['advantage'].append(self.advantages[idx])
                batch['return'].append(self.returns[idx])

            # 转为张量
            for key in batch:
                if key == 'path_history':
                    # 变长路径: pad到batch内最大长度
                    max_len = max(len(p) for p in batch[key])
                    action_dim = batch[key][0].shape[-1] if len(batch[key][0]) > 0 else 4
                    padded = np.zeros((len(batch[key]), max(max_len, 1), action_dim), dtype=np.float32)
                    for i, p in enumerate(batch[key]):
                        if len(p) > 0:
                            padded[i, :len(p)] = p
                    batch[key] = torch.FloatTensor(padded)
                elif key in ['features', 'budget_mask']:
                    batch[key] = torch.FloatTensor(np.array(batch[key]))
                elif key in ['remaining_budget', 'old_log_prob', 'old_value', 'advantage', 'return']:
                    batch[key] = torch.FloatTensor(np.array(batch[key]))
                elif key == 'action_idx':
                    batch[key] = torch.LongTensor(np.array(batch[key]))

            yield batch

    def __len__(self):
        return len(self.transitions)

    def clear(self):
        self.transitions.clear()
        self.advantages = []
        self.returns = []
