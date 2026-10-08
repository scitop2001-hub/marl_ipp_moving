"""
移动目标 — 论文创新核心
每个目标有唯一ID、位置、速度，按运动模型每步更新位置
支持: 随机游走 / 匀速运动 / 混合运动
边界处理: 反弹 / 环绕 / 重生
"""
import numpy as np


class MovingTarget:
    """单个移动目标"""

    def __init__(self, target_id, pos, speed, motion_type="random_walk", env_size=1.0, rng=None):
        self.id = target_id
        self.pos = np.array(pos, dtype=np.float64)
        self.env_size = env_size
        self.motion_type = motion_type
        self.speed = speed
        self.rng = rng if rng is not None else np.random.RandomState()

        # 速度方向 (随机初始方向)
        angle = self.rng.uniform(0, 2 * np.pi)
        self.velocity = np.array([
            speed * np.cos(angle),
            speed * np.sin(angle),
            0.0  # z方向不动 (简化: 目标在地面/建筑表面移动)
        ])

        # 历史轨迹 (用于运动预测训练)
        self.trajectory = [self.pos.copy()]

        # 是否被发现过
        self.ever_discovered = False
        self.last_seen_step = -1

    def step(self, boundary="bounce"):
        """移动一步"""
        if self.motion_type == "random_walk":
            # 随机游走: 每步随机改变方向
            angle = self.rng.uniform(0, 2 * np.pi)
            self.velocity[0] = self.speed * np.cos(angle)
            self.velocity[1] = self.speed * np.sin(angle)
        elif self.motion_type == "constant_velocity":
            # 匀速: 方向不变, 碰边界反弹
            pass
        elif self.motion_type == "hybrid":
            # 混合: 大部分时间匀速, 偶尔变向
            if self.rng.random() < 0.1:
                angle = self.rng.uniform(0, 2 * np.pi)
                self.velocity[0] = self.speed * np.cos(angle)
                self.velocity[1] = self.speed * np.sin(angle)

        # 更新位置
        self.pos += self.velocity

        # 边界处理
        if boundary == "bounce":
            for i in range(3):
                if self.pos[i] < 0.02:
                    self.pos[i] = 0.02
                    self.velocity[i] = abs(self.velocity[i])
                elif self.pos[i] > self.env_size - 0.02:
                    self.pos[i] = self.env_size - 0.02
                    self.velocity[i] = -abs(self.velocity[i])
        elif boundary == "wrap":
            self.pos = self.pos % self.env_size
        elif boundary == "respawn":
            for i in range(3):
                if self.pos[i] < 0 or self.pos[i] > self.env_size:
                    self.pos = self.rng.uniform(0.02, self.env_size - 0.02, 3)
                    break

        # 记录轨迹
        self.trajectory.append(self.pos.copy())
        if len(self.trajectory) > 200:
            self.trajectory.pop(0)

    def get_state(self):
        """获取目标状态 (位置+速度)"""
        return np.concatenate([self.pos, self.velocity])

    def get_history(self, length=10):
        """获取最近length步的轨迹"""
        traj = self.trajectory[-length:]
        if len(traj) < length:
            # 补零
            pad = [traj[0]] * (length - len(traj))
            traj = pad + traj
        return np.array(traj)
