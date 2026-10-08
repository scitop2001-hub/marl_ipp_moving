"""
协调图 — 移动目标版
特征维度从10扩展到13, 新增3维预测特征:
  pred_density:    运动预测网络预测的该位置附近目标密度
  pred_uncertainty: 预测不确定性
  target_velocity:  估计的目标平均速度
"""
import numpy as np
import torch


class CoordinationGraph:
    """移动目标版协调图"""

    def __init__(self, config):
        self.config = config
        self.num_candidates = config.graph.num_candidates
        self.action_dim = config.graph.action_dim
        self.feature_dim = config.graph.feature_dim  # 13

    def build(self, robot_idx, candidate_actions, utility_gp, comm_gp,
              prediction_manager, target_histories,
              current_pose, remaining_budget, current_time=0):
        """
        构建协调图 (增加了预测特征)

        返回:
            feature_matrix: (L, 13)
            adjacency: (L, L)
            budget_mask: (L,)
            edge_costs: (L,)
        """
        L = len(candidate_actions)

        # 1. 时空效用GP
        u_utility, p_utility = utility_gp.query(candidate_actions, current_time=current_time)

        # 2. 通信GP
        u_comm, p_comm = comm_gp.query(candidate_actions)

        # 3. 运动预测特征 (新增)
        pred_density, pred_uncertainty = prediction_manager.predict_target_density(
            target_histories, candidate_actions[:, :3],
            horizon=self.config.prediction.prediction_horizon
        )

        # 4. 边代价
        edge_costs = np.linalg.norm(
            candidate_actions[:, :3] - current_pose[:3], axis=1
        )

        # 5. 预算掩码
        budget_mask = (edge_costs <= remaining_budget).astype(np.float32)

        # 6. 特征矩阵 (13维)
        feature_matrix = np.zeros((L, self.feature_dim), dtype=np.float32)
        feature_matrix[:, 0] = candidate_actions[:, 0]     # x
        feature_matrix[:, 1] = candidate_actions[:, 1]     # y
        feature_matrix[:, 2] = candidate_actions[:, 2]     # z
        feature_matrix[:, 3] = candidate_actions[:, 3]     # d (view dir)
        feature_matrix[:, 4] = u_utility                   # 效用均值
        feature_matrix[:, 5] = p_utility                   # 效用方差
        feature_matrix[:, 6] = u_comm                      # 通信概率
        feature_matrix[:, 7] = p_comm                      # 通信方差
        feature_matrix[:, 8] = budget_mask                 # 预算掩码
        feature_matrix[:, 9] = edge_costs / max(remaining_budget, 1e-6)  # 归一化距离
        # === 新增3维 ===
        feature_matrix[:, 10] = pred_density               # 预测目标密度
        feature_matrix[:, 11] = pred_uncertainty           # 预测不确定性
        feature_matrix[:, 12] = np.mean(pred_density)      # 全局密度估计

        # 7. 邻接矩阵
        adjacency = np.ones((L, L), dtype=np.float32) - np.eye(L, dtype=np.float32)

        return feature_matrix, adjacency, budget_mask, edge_costs

    def build_batch(self, robot_idx, candidate_actions, utility_gp, comm_gp,
                    prediction_manager, target_histories,
                    current_pose, remaining_budget, current_time=0):
        """构建并转为张量"""
        feature_matrix, adjacency, budget_mask, edge_costs = self.build(
            robot_idx, candidate_actions, utility_gp, comm_gp,
            prediction_manager, target_histories,
            current_pose, remaining_budget, current_time
        )

        device = self.config.device
        features = torch.FloatTensor(feature_matrix).unsqueeze(0).to(device)
        adj = torch.FloatTensor(adjacency).unsqueeze(0).to(device)
        mask = torch.FloatTensor(budget_mask).unsqueeze(0).to(device)

        return features, adj, mask, edge_costs
