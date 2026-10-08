"""
RGB-D 传感器模型 — 论文 Section V-A
- 90° 视场角
- 感知范围限制为环境大小的 24%
- 单向传感器 (unidirectional)
- 噪声无关分类器 (noiseless classifier)
"""
import numpy as np


class RGBDSensor:
    def __init__(self, fov_deg=90.0, sensor_range=0.24, env_size=1.0):
        self.fov = np.radians(fov_deg)
        self.sensor_range = sensor_range
        self.env_size = env_size
        self.half_fov = self.fov / 2.0

    def get_visible_targets(self, robot_pos, view_dir, targets):
        """
        获取传感器视野内的目标

        参数:
            robot_pos: (x, y, z) 机器人位置
            view_dir: 观察方向 (弧度, 绕z轴)
            targets: list of (x, y, z) 目标位置

        返回:
            visible_indices: 可见目标的索引列表
        """
        robot_pos = np.array(robot_pos[:3])
        visible_indices = []

        # 观察方向向量 (xy平面)
        view_vec = np.array([np.cos(view_dir), np.sin(view_dir), 0.0])

        for idx, target in enumerate(targets):
            target_pos = np.array(target[:3])
            delta = target_pos - robot_pos
            dist = np.linalg.norm(delta)

            # 距离检查
            if dist > self.sensor_range or dist < 1e-6:
                continue

            # 视场角检查 (在xy平面)
            if abs(delta[2]) > self.sensor_range * 0.5:
                # z方向差太大也不可见 (近似)
                continue

            delta_xy = delta[:2]
            if np.linalg.norm(delta_xy) < 1e-6:
                continue

            # 计算角度差
            target_dir = delta_xy / np.linalg.norm(delta_xy)
            cos_angle = np.dot(view_vec[:2], target_dir)
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            if angle <= self.half_fov:
                visible_indices.append(idx)

        return visible_indices

    def get_visible_volume(self, robot_pos, view_dir, num_rays=64):
        """
        获取传感器扫描到的空间范围 (用于更新占用地图)
        简化版: 返回视锥内的采样点
        """
        robot_pos = np.array(robot_pos[:3])
        view_vec = np.array([np.cos(view_dir), np.sin(view_dir), 0.0])

        samples = []
        for i in range(num_rays):
            angle_offset = np.linspace(-self.half_fov, self.half_fov, num_rays)[i]
            ray_dir = np.array([
                np.cos(view_dir + angle_offset),
                np.sin(view_dir + angle_offset),
                0.0
            ])
            # 沿射线采样
            for t in np.linspace(0.01, self.sensor_range, 10):
                sample = robot_pos + t * ray_dir
                if np.all(sample >= 0) and np.all(sample <= self.env_size):
                    samples.append(sample)

        return np.array(samples) if samples else np.zeros((0, 3))
