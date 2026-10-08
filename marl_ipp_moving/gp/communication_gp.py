"""
效用GP + 通信GP — 移动目标版
效用GP改用时空核 (SpatiotemporalGP)
通信GP保持空间核不变 (其他机器人路径是空间概念)
"""
import numpy as np
from .gaussian_process import GaussianProcess
from .spatiotemporal_gp import SpatiotemporalGP


class UtilityGP:
    """时空效用GP (移动目标版)"""

    def __init__(self, config):
        self.use_spatiotemporal = config.gp.spatiotemporal_kernel
        if self.use_spatiotemporal:
            self.gp = SpatiotemporalGP(
                space_ls=config.gp.length_scale,
                time_ls=config.gp.time_length_scale,
                sigma_f=config.gp.sigma_f,
                sigma_n=config.gp.sigma_n,
                max_train_points=config.gp.max_train_points
            )
        else:
            self.gp = GaussianProcess(
                kernel_type=config.gp.kernel_type,
                length_scale=config.gp.length_scale,
                sigma_f=config.gp.sigma_f,
                sigma_n=config.gp.sigma_n,
                max_train_points=config.gp.max_train_points
            )
        self.normalize_const = config.env.target_normalize_const
        self.use_spatiotemporal = config.gp.spatiotemporal_kernel

    def update(self, actions_with_time, utilities):
        """更新GP (actions含时间维)"""
        if len(actions_with_time) == 0:
            return
        actions_with_time = np.array(actions_with_time)
        utilities = np.array(utilities) / self.normalize_const
        self.gp.set_data(actions_with_time, utilities)

    def add_observation(self, action, num_new_targets, timestep=0):
        """添加单条时空观测"""
        if self.use_spatiotemporal:
            x = np.array([[action[0], action[1], action[2], timestep]])
        else:
            x = np.array([action[:3]])
        self.gp.add_data(x if hasattr(self.gp, 'add_data') else x,
                        num_new_targets / self.normalize_const)
        # SpatiotemporalGP 没有 add_data, 用 set_data
        if isinstance(self.gp, SpatiotemporalGP):
            # 收集所有数据后一次性 set
            pass  # 在外部批量更新

    def query(self, candidate_actions, current_time=0):
        """
        查询候选动作的效用和不确定性
        candidate_actions: (L, 4) [x,y,z,d]
        """
        positions = candidate_actions[:, :3]  # 只取位置
        if self.use_spatiotemporal:
            mean, var = self.gp.predict(positions, current_time=current_time)
        else:
            mean, var = self.gp.predict(positions)
        return mean, var

    def get_trace_reduction(self, candidate_actions, action_idx, current_time=0):
        """探索奖励: GP方差迹减少"""
        positions = candidate_actions[:, :3]
        if self.use_spatiotemporal:
            prior_var, _ = self.gp.predict(positions, current_time)
        else:
            prior_var, _ = self.gp.predict(positions)
        trace_prior = np.sum(prior_var)
        if trace_prior < 1e-10:
            return 0.0
        action = positions[action_idx].reshape(1, -1)
        if self.use_spatiotemporal:
            _, action_var = self.gp.predict(action, current_time)
        else:
            _, action_var = self.gp.predict(action)
        trace_posterior = max(trace_prior - action_var[0], 0)
        return (trace_prior - trace_posterior) / trace_prior

    def reset(self):
        self.gp.reset()


class CommunicationGP:
    """通信GP (与静态版一致, 空间核)"""

    def __init__(self, config):
        self.gp = GaussianProcess(
            kernel_type=config.gp.kernel_type,
            length_scale=config.gp.length_scale,
            sigma_f=config.gp.sigma_f,
            sigma_n=config.gp.sigma_n,
            max_train_points=config.gp.max_train_points
        )

    def update(self, waypoints):
        if len(waypoints) == 0:
            self.gp.reset()
            return
        waypoints = np.array(waypoints)[:, :3]
        y = np.ones(len(waypoints))
        self.gp.set_data(waypoints, y)

    def query(self, candidate_actions):
        positions = candidate_actions[:, :3]
        mean, var = self.gp.predict(positions)
        return mean, var

    def get_trace_reduction(self, candidate_actions, action_idx):
        prior_var, _ = self.gp.predict(candidate_actions[:, :3])
        trace_prior = np.sum(prior_var)
        if trace_prior < 1e-10:
            return 0.0
        action = candidate_actions[action_idx, :3].reshape(1, -1)
        _, action_var = self.gp.predict(action)
        trace_posterior = max(trace_prior - action_var[0], 0)
        return (trace_prior - trace_posterior) / trace_prior

    def reset(self):
        self.gp.reset()
