"""
随机策略基线 — 论文 Table I, Fig 3
随机选择候选动作 (性能下界)
"""
import numpy as np


class RandomAgent:
    """随机策略"""

    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.name = "Random"

    def select_action(self, candidates, budget_mask=None, **kwargs):
        """随机选择一个候选动作"""
        if budget_mask is not None:
            valid_indices = np.where(budget_mask > 0)[0]
            if len(valid_indices) == 0:
                valid_indices = np.arange(len(candidates))
        else:
            valid_indices = np.arange(len(candidates))

        return self.rng.choice(valid_indices)
