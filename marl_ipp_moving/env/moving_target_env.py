"""
移动目标城市环境 — 在原3D城市环境基础上增加移动目标
- 保留建筑(静态障碍物)
- 窗户(静态目标)替换为移动目标(在地面上移动的ArUco标记/车辆等)
- 每个移动目标有唯一ID, 按运动模型每步更新位置
- 观测返回目标ID (用于去重和跟踪)
"""
import numpy as np
from typing import List, Tuple, Dict
from .sensor import RGBDSensor
from .occupancy_map import OccupancyMap
from .moving_target import MovingTarget


class MovingTargetEnv:
    """移动目标3D城市环境"""

    def __init__(self, config, is_training=True, env_scale=1.0, seed=None):
        self.config = config
        self.env_size = config.env.env_size * env_scale
        self.is_training = is_training
        self.env_scale = env_scale
        self.rng = np.random.RandomState(seed)

        # 传感器
        self.sensor = RGBDSensor(
            fov_deg=config.env.sensor_fov,
            sensor_range=config.env.sensor_range * env_scale,
            env_size=self.env_size
        )

        # 障碍物 (建筑, 静态)
        self.building_occupancy = OccupancyMap(
            env_size=self.env_size,
            voxel_size=config.env.voxel_size
        )
        self._generate_buildings()

        # 移动目标
        self.targets: List[MovingTarget] = []
        self._generate_moving_targets()

        # 时间步
        self.t = 0

        # 已发现目标ID集合 (全局去重)
        self.discovered_ids = set()

        # 目标跟踪状态: {target_id: last_seen_pos, last_seen_step}
        self.tracking_state: Dict[int, Tuple[np.ndarray, int]] = {}

    def _generate_buildings(self):
        """生成建筑(静态障碍物)"""
        cfg = self.config.env
        num_buildings = cfg.train_num_buildings if self.is_training else cfg.test_num_buildings

        if self.is_training:
            grid_n = int(np.ceil(np.sqrt(num_buildings)))
            spacing = self.env_size / (grid_n + 1)
            building_size = np.array([spacing * 0.4, spacing * 0.4, self.env_size * 0.4])
            idx = 0
            for i in range(grid_n):
                for j in range(grid_n):
                    if idx >= num_buildings:
                        break
                    center = np.array([
                        spacing * (i + 1),
                        spacing * (j + 1),
                        building_size[2] / 2
                    ])
                    self._add_building(center, building_size)
                    idx += 1
        else:
            for _ in range(num_buildings):
                w = self.rng.uniform(0.05, 0.12) * self.env_scale
                d = self.rng.uniform(0.05, 0.12) * self.env_scale
                h = self.rng.uniform(0.15, 0.4) * self.env_scale
                center = np.array([
                    self.rng.uniform(w/2 + 0.02, self.env_size - w/2 - 0.02),
                    self.rng.uniform(d/2 + 0.02, self.env_size - d/2 - 0.02),
                    h / 2
                ])
                self._add_building(center, np.array([w, d, h]))

    def _add_building(self, center, size):
        """添加建筑到占用地图"""
        occ = self.building_occupancy
        b = [
            center[0] - size[0]/2, center[0] + size[0]/2,
            center[1] - size[1]/2, center[1] + size[1]/2,
            center[2] - size[2]/2, center[2] + size[2]/2,
        ]
        step = occ.voxel_size
        for x in np.arange(b[0], b[1], step):
            for y in np.arange(b[2], b[3], step):
                for z in np.arange(b[4], b[5], step):
                    occ.set_occupied([x, y, z])

    def _generate_moving_targets(self):
        """生成移动目标"""
        cfg = self.config.env
        num_targets = cfg.num_moving_targets
        speed_min, speed_max = cfg.target_speed_range

        for i in range(num_targets):
            # 随机位置 (避开建筑)
            for _ in range(50):
                pos = self.rng.uniform(0.05, self.env_size - 0.05, 3)
                pos[2] = self.rng.uniform(0.01, 0.05)  # 低高度
                if not self.building_occupancy.is_occupied(pos):
                    break

            speed = self.rng.uniform(speed_min, speed_max)
            target = MovingTarget(
                target_id=i,
                pos=pos,
                speed=speed,
                motion_type=cfg.target_motion_type,
                env_size=self.env_size,
                rng=self.rng
            )
            self.targets.append(target)

    def step_targets(self):
        """所有移动目标前进一步"""
        boundary = self.config.env.target_boundary
        for target in self.targets:
            target.step(boundary=boundary)
        self.t += 1

    def observe(self, robot_pos, view_dir, discovered_ids=None):
        """
        机器人观测移动目标

        返回:
            new_discovered: list of (target_id, pos) 新发现的目标
            tracked: list of (target_id, pos) 重新观测到的已知目标
        """
        target_positions = [t.pos for t in self.targets]
        visible_indices = self.sensor.get_visible_targets(
            robot_pos, view_dir, target_positions
        )

        new_discovered = []
        tracked = []

        for idx in visible_indices:
            target = self.targets[idx]
            target.ever_discovered = True
            target.last_seen_step = self.t

            if discovered_ids is not None and target.id not in discovered_ids:
                discovered_ids.add(target.id)
                new_discovered.append((target.id, target.pos.copy()))
            else:
                tracked.append((target.id, target.pos.copy()))

            # 更新跟踪状态
            self.tracking_state[target.id] = (target.pos.copy(), self.t)

        return new_discovered, tracked

    def check_collision(self, pos):
        """检查是否与建筑碰撞"""
        return self.building_occupancy.is_occupied(pos)

    def check_path_collision(self, start, end):
        """检查路径碰撞"""
        return self.building_occupancy.check_path_collision(start, end)

    def get_total_targets(self):
        return len(self.targets)

    def get_discovered_ratio(self, discovered_ids):
        if len(self.targets) == 0:
            return 0.0
        return len(discovered_ids) / len(self.targets)

    def get_target_positions(self):
        """获取所有目标当前位置"""
        return np.array([t.pos for t in self.targets])

    def get_target_states(self):
        """获取所有目标状态 (位置+速度)"""
        return np.array([t.get_state() for t in self.targets])

    def get_target_histories(self, length=10):
        """获取所有目标历史轨迹 (用于运动预测训练)"""
        return np.array([t.get_history(length) for t in self.targets])

    def get_tracking_accuracy(self):
        """
        获取跟踪精度: 最近5步内被观测到的目标比例
        """
        if len(self.targets) == 0:
            return 0.0
        recent = sum(1 for t in self.targets
                     if t.last_seen_step >= 0 and self.t - t.last_seen_step <= 5)
        return recent / len(self.targets)

    def reset(self, seed=None):
        """重置环境"""
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        self.building_occupancy = OccupancyMap(
            env_size=self.env_size,
            voxel_size=self.config.env.voxel_size
        )
        self._generate_buildings()
        self.targets = []
        self._generate_moving_targets()
        self.t = 0
        self.discovered_ids = set()
        self.tracking_state = {}
