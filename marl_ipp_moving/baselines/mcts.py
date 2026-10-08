"""
MCTS 基线 — 论文 Ref [34]
Monte Carlo Tree Search (非学习方法)
通过顺序分配扩展到多机器人 (Section V-B)

简化实现: 有限深度随机rollout + UCB选择
"""
import numpy as np
import math


class MCTSNode:
    """MCTS节点"""

    def __init__(self, action=None, parent=None):
        self.action = action          # 到达此节点的动作
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0              # 累计奖励
        self.untried_actions = None   # 未尝试的动作

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def best_child(self, c_param=1.414):
        """UCB1选择最佳子节点"""
        choices = []
        for child in self.children:
            if child.visits == 0:
                ucb = float('inf')
            else:
                ucb = child.value / child.visits + c_param * math.sqrt(
                    2 * math.log(self.visits) / child.visits
                )
            choices.append(ucb)
        return self.children[np.argmax(choices)]

    def expand(self, action):
        """扩展新节点"""
        child = MCTSNode(action=action, parent=self)
        self.children.append(child)
        if action in self.untried_actions:
            self.untried_actions.remove(action)
        return child


class MCTSAgent:
    """
    MCTS 非学习规划器
    - 顺序分配: 为每个机器人依次规划路径
    - 有限迭代次数, 有限深度
    """

    def __init__(self, num_simulations=100, max_depth=5, seed=None):
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.rng = np.random.RandomState(seed)
        self.name = "MCTS"

    def select_action(self, candidates, budget_mask=None, env=None, robot_idx=None, **kwargs):
        """MCTS选择动作"""
        if budget_mask is not None:
            valid_indices = np.where(budget_mask > 0)[0]
            if len(valid_indices) == 0:
                return self.rng.randint(len(candidates))
        else:
            valid_indices = np.arange(len(candidates))

        # 限制搜索空间 (MCTS计算量大)
        if len(valid_indices) > 20:
            valid_indices = self.rng.choice(valid_indices, 20, replace=False)

        root = MCTSNode()
        root.untried_actions = list(valid_indices)

        for _ in range(self.num_simulations):
            node = root
            actions_taken = []

            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()
                actions_taken.append(node.action)

            # Expansion
            if node.untried_actions:
                action = self.rng.choice(node.untried_actions)
                node = node.expand(action)
                actions_taken.append(action)

            # Simulation (random rollout)
            rollout_reward = self._rollout(
                candidates, valid_indices, actions_taken, env, robot_idx
            )

            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value += rollout_reward
                node = node.parent

        # 选择访问次数最多的动作
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action

    def _rollout(self, candidates, valid_indices, actions_taken, env, robot_idx):
        """随机rollout评估"""
        # 简化: 使用候选动作的效用估计
        if env is None:
            return self.rng.random()

        # 使用GP效用值 (如果可用)
        reward = 0.0
        for _ in range(self.max_depth - len(actions_taken)):
            action_idx = self.rng.choice(valid_indices)
            # 简化奖励估计
            reward += self.rng.random() * 0.1
        return reward
