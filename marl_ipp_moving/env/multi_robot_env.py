"""
多机器人移动目标环境
整合: 移动目标环境 + 多机器人协调 + 通信 + 时空GP数据收集
关键差异(与静态版):
  1. 每步先移动所有目标, 再让机器人决策
  2. 观测返回目标ID, 用于去重
  3. 收集目标轨迹数据, 用于训练运动预测网络
"""
import numpy as np
from typing import List, Dict
from .moving_target_env import MovingTargetEnv
from .occupancy_map import OccupancyMap


class MultiRobotMovingEnv:
    """多机器人移动目标环境"""

    def __init__(self, config, num_robots=3, is_training=True, env_scale=1.0, seed=None):
        self.config = config
        self.num_robots = num_robots
        self.is_training = is_training
        self.env_scale = env_scale

        self.target_env = MovingTargetEnv(config, is_training, env_scale, seed)

        if is_training:
            self.comm_range = config.env.comm_range_train
        else:
            self.comm_range = config.env.comm_range_global

        # 机器人状态
        self.robot_poses = []
        self.robot_paths = []
        self.robot_budgets = []
        self.robot_occupancy_maps = []

        # 全局已发现ID集合
        self.global_discovered_ids = set()

        # 效用GP数据 (时空: [x, y, z, t])
        self.gp_train_actions = []     # [x, y, z, t]
        self.gp_train_utilities = []   # 新发现数
        self.gp_train_times = []       # 时间戳

        # 通信GP数据
        self.comm_gp_data = [[] for _ in range(num_robots)]

        # 运动预测训练数据
        self.prediction_train_data = []  # list of (history, future_pos)

        self.t = 0
        self.max_timesteps = config.ppo.max_timesteps
        self.start_pose = np.array([0.0, 0.0, 0.0, np.pi / 2])

    def reset(self, budget=None, comm_range=None, seed=None):
        """重置"""
        if seed is not None:
            self.target_env.reset(seed)
        else:
            self.target_env.reset()

        if comm_range is not None:
            self.comm_range = comm_range

        if budget is None:
            budget = self.config.ppo.budget_range[0] + np.random.rand() * (
                self.config.ppo.budget_range[1] - self.config.ppo.budget_range[0]
            )

        per_robot_budget = budget / self.num_robots
        self.robot_poses = []
        self.robot_paths = []
        self.robot_budgets = []
        self.robot_occupancy_maps = []

        for i in range(self.num_robots):
            pose = self.start_pose.copy()
            pose[0] += i * 0.02
            self.robot_poses.append(pose)
            self.robot_paths.append([pose.copy()])
            self.robot_budgets.append(per_robot_budget)
            self.robot_occupancy_maps.append(
                OccupancyMap(
                    env_size=self.target_env.env_size,
                    voxel_size=self.config.env.voxel_size
                )
            )

        self.global_discovered_ids = set()
        self.gp_train_actions = []
        self.gp_train_utilities = []
        self.gp_train_times = []
        self.comm_gp_data = [[] for _ in range(self.num_robots)]
        self.prediction_train_data = []
        self.t = 0

        return self._get_state()

    def _get_comm_neighbors(self):
        neighbors = [[] for _ in range(self.num_robots)]
        for i in range(self.num_robots):
            for j in range(self.num_robots):
                if i == j:
                    continue
                dist = np.linalg.norm(self.robot_poses[i][:3] - self.robot_poses[j][:3])
                if dist <= self.comm_range:
                    neighbors[i].append(j)
        return neighbors

    def _exchange_paths(self, neighbors):
        for i in range(self.num_robots):
            for j in neighbors[i]:
                for waypoint in self.robot_paths[j]:
                    self.comm_gp_data[i].append(tuple(waypoint[:3]))

    def step_targets(self):
        """移动所有目标 (在机器人决策前调用)"""
        self.target_env.step_targets()

    def get_candidate_actions(self, robot_idx, num_candidates=80):
        """采样候选动作 (与静态版基本一致)"""
        cfg = self.config.env
        current_pose = self.robot_poses[robot_idx]
        occ_map = self.robot_occupancy_maps[robot_idx]
        neighbors = self._get_comm_neighbors()[robot_idx]

        candidates = []
        attempts = 0
        max_attempts = num_candidates * 10

        while len(candidates) < num_candidates and attempts < max_attempts:
            attempts += 1
            new_pos = current_pose[:3] + np.random.uniform(
                -cfg.local_region_size, cfg.local_region_size, size=3
            )
            new_pos = np.clip(new_pos, 0.01, self.target_env.env_size - 0.01)
            new_pos[2] = np.clip(new_pos[2], 0.01, self.target_env.env_size * 0.8)

            collision = False
            for j in neighbors:
                dist = np.linalg.norm(new_pos - self.robot_poses[j][:3])
                if dist < cfg.collision_dist:
                    collision = True
                    break
            if collision:
                continue

            if not occ_map.check_path_collision(current_pose, new_pos):
                continue

            view_dir = cfg.sensor_view_dirs[np.random.randint(len(cfg.sensor_view_dirs))]
            move_cost = np.linalg.norm(new_pos - current_pose[:3])
            if move_cost > self.robot_budgets[robot_idx]:
                continue

            candidates.append(np.array([new_pos[0], new_pos[1], new_pos[2], view_dir]))

        while len(candidates) < num_candidates:
            view_dir = cfg.sensor_view_dirs[np.random.randint(len(cfg.sensor_view_dirs))]
            candidates.append(np.array([
                current_pose[0], current_pose[1], current_pose[2], view_dir
            ]))

        return np.array(candidates)

    def step(self, robot_idx, action):
        """
        机器人执行动作
        返回: num_new_discovered, num_tracked, move_cost, done
        """
        current_pose = self.robot_poses[robot_idx]
        target_pos = action[:3]
        view_dir = action[3]

        move_cost = np.linalg.norm(target_pos - current_pose[:3])
        self.robot_budgets[robot_idx] -= move_cost

        self.robot_poses[robot_idx] = action.copy()
        self.robot_paths[robot_idx].append(action.copy())

        # 更新占用地图
        occ_map = self.robot_occupancy_maps[robot_idx]
        for t in np.linspace(0, 1, 10):
            pos = current_pose[:3] + t * (target_pos - current_pose[:3])
            occ_map.set_free(pos)
            if self.target_env.check_collision(pos):
                occ_map.set_occupied(pos)

        # 观测移动目标 (返回新发现 + 已跟踪)
        new_discovered, tracked = self.target_env.observe(
            target_pos, view_dir, self.global_discovered_ids
        )

        num_new = len(new_discovered)
        num_tracked = len(tracked)

        # 更新时空GP数据
        self.gp_train_actions.append([action[0], action[1], action[2], self.t])
        self.gp_train_utilities.append(num_new / self.config.env.target_normalize_const)
        self.gp_train_times.append(self.t)

        # 更新通信GP
        self.comm_gp_data[robot_idx].append(tuple(target_pos))

        # 收集运动预测训练数据
        if len(self.target_env.targets) > 0 and self.t > self.config.prediction.history_length:
            histories = self.target_env.get_target_histories(
                self.config.prediction.history_length
            )
            for target in self.target_env.targets:
                hist = target.get_history(self.config.prediction.history_length)
                future = target.pos.copy()
                self.prediction_train_data.append((hist.copy(), future.copy()))

        done = self.robot_budgets[robot_idx] <= 0.001
        self.t += 1

        return num_new, num_tracked, move_cost, done

    def _get_state(self):
        return {
            't': self.t,
            'robot_poses': [p.copy() for p in self.robot_poses],
            'robot_budgets': [b for b in self.robot_budgets],
            'num_discovered': len(self.global_discovered_ids),
            'total_targets': self.target_env.get_total_targets(),
            'discovered_ratio': self.target_env.get_discovered_ratio(self.global_discovered_ids),
            'tracking_accuracy': self.target_env.get_tracking_accuracy(),
        }

    def all_done(self):
        return all(b <= 0.001 for b in self.robot_budgets) or self.t >= self.max_timesteps

    def get_spatiotemporal_gp_data(self):
        """获取时空GP训练数据 [x,y,z,t] -> utility"""
        if len(self.gp_train_actions) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        X = np.array(self.gp_train_actions)
        y = np.array(self.gp_train_utilities)
        return X, y

    def get_comm_gp_data(self, robot_idx):
        data = self.comm_gp_data[robot_idx]
        if len(data) == 0:
            return np.zeros((0, 3))
        return np.array(list(set(data)))

    def get_prediction_train_data(self):
        """获取运动预测训练数据"""
        return self.prediction_train_data
