"""
占用地图 — 论文 Section IV-A
每个机器人维护一个占用地图用于避障。
体素值: 0=空闲, 1=未知, 2=占用(目标或障碍物)
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class OccupancyMap:
    """3D体素占用地图"""
    env_size: float = 1.0
    voxel_size: float = 0.02

    def __post_init__(self):
        self.grid_size = int(self.env_size / self.voxel_size)
        # 初始化为未知空间 (1)
        self.grid = np.ones(
            (self.grid_size, self.grid_size, self.grid_size),
            dtype=np.int8
        )
        # 已发现目标集合 (去重)
        self.discovered_targets = set()

    def world_to_voxel(self, pos):
        """世界坐标 -> 体素索引"""
        idx = np.clip(
            (np.array(pos) / self.voxel_size).astype(int),
            0, self.grid_size - 1
        )
        return tuple(idx)

    def voxel_to_world(self, idx):
        """体素索引 -> 世界坐标 (体素中心)"""
        return (np.array(idx) + 0.5) * self.voxel_size

    def set_occupied(self, pos):
        """标记体素为占用"""
        idx = self.world_to_voxel(pos)
        self.grid[idx] = 2

    def set_free(self, pos):
        """标记体素为空闲"""
        idx = self.world_to_voxel(pos)
        self.grid[idx] = 0

    def is_occupied(self, pos):
        """检查体素是否为占用或未知(不可通行)"""
        idx = self.world_to_voxel(pos)
        return self.grid[idx] == 2

    def is_known_free(self, pos):
        """检查体素是否为已知空闲(可通行)"""
        idx = self.world_to_voxel(pos)
        return self.grid[idx] == 0

    def is_unknown(self, pos):
        """检查体素是否为未知"""
        idx = self.world_to_voxel(pos)
        return self.grid[idx] == 1

    def check_path_collision(self, start, end, num_samples=20):
        """
        沿直线检查碰撞 (可达性检查)
        返回 True 如果路径无碰撞
        """
        start = np.array(start[:3])
        end = np.array(end[:3])
        for t in np.linspace(0, 1, num_samples):
            pos = start + t * (end - start)
            # 越界检查
            if np.any(pos < 0) or np.any(pos > self.env_size):
                return False
            # 占用检查 (已知障碍物)
            if self.is_occupied(pos):
                return False
        return True

    def get_known_free_ratio(self):
        """返回已知空闲区域占比"""
        return np.mean(self.grid == 0)

    def reset(self):
        """重置占用地图"""
        self.grid.fill(1)
        self.discovered_targets.clear()
