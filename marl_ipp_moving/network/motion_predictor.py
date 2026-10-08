"""
运动预测网络 — 创新模块
基于LSTM, 输入目标历史轨迹, 预测目标未来位置
与策略网络联合训练 (也可预训练)
"""
import torch
import torch.nn as nn
import numpy as np


class MotionPredictor(nn.Module):
    """
    目标运动预测网络
    输入: (batch, seq_len, 3) 目标历史位置序列
    输出: (batch, 3) 预测的未来位置
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        hidden = config.prediction.hidden_dim
        num_layers = config.prediction.num_layers
        input_dim = 3  # [x, y, z]
        output_dim = config.prediction.output_dim

        # LSTM 编码历史轨迹
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0
        )

        # 预测头
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, output_dim)
        )

        # 速度估计 (从LSTM隐状态推断)
        self.velocity_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 3)  # [vx, vy, vz]
        )

    def forward(self, history):
        """
        参数:
            history: (batch, seq_len, 3) 历史位置序列

        返回:
            predicted_pos: (batch, 3) 预测位置
            estimated_vel: (batch, 3) 估计速度
        """
        _, (h_n, _) = self.lstm(history)
        # h_n: (num_layers, batch, hidden), 取最后一层
        h = h_n[-1]  # (batch, hidden)

        predicted_pos = self.head(h)
        estimated_vel = self.velocity_head(h)

        return predicted_pos, estimated_vel

    def predict_future(self, history, horizon=5):
        """
        多步预测 (自回归)

        参数:
            history: (batch, seq_len, 3)
            horizon: 预测步数

        返回:
            predictions: (batch, horizon, 3)
        """
        predictions = []
        current_history = history.clone()

        for _ in range(horizon):
            pred_pos, _ = self.forward(current_history)
            predictions.append(pred_pos)

            # 滑动窗口: 去掉最早的, 加入预测
            current_history = torch.cat([
                current_history[:, 1:, :],
                pred_pos.unsqueeze(1)
            ], dim=1)

        return torch.stack(predictions, dim=1)  # (batch, horizon, 3)


class PredictionManager:
    """运动预测管理器 (管理数据收集和训练)"""

    def __init__(self, config, device='cpu'):
        self.config = config
        self.device = device
        self.predictor = MotionPredictor(config).to(device)
        self.optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=config.prediction.learning_rate
        )
        self.train_data = []  # list of (history, future)

    def add_data(self, history, future_pos):
        """添加训练数据"""
        self.train_data.append((history.copy(), future_pos.copy()))

    def train_step(self, batch_size=64):
        """训练一步"""
        if len(self.train_data) < batch_size:
            return 0.0

        # 采样batch
        indices = np.random.choice(len(self.train_data), batch_size, replace=False)
        histories = []
        futures = []
        for idx in indices:
            h, f = self.train_data[idx]
            histories.append(h)
            futures.append(f)

        histories = torch.FloatTensor(np.array(histories)).to(self.device)
        futures = torch.FloatTensor(np.array(futures)).to(self.device)

        # 前向
        pred_pos, pred_vel = self.predictor(histories)

        # 损失: 位置预测MSE + 速度估计正则
        pos_loss = nn.functional.mse_loss(pred_pos, futures)

        # 速度正则: 估计速度应该接近 (future - last_pos)
        last_pos = histories[:, -1, :]
        true_vel = futures - last_pos
        vel_loss = nn.functional.mse_loss(pred_vel, true_vel)

        loss = pos_loss + 0.1 * vel_loss

        # 反向
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def predict_target_density(self, target_histories, candidate_positions, horizon=5):
        """
        预测未来目标密度: 给定候选动作位置, 返回预测的目标密度和不确定性

        参数:
            target_histories: (num_targets, seq_len, 3) 所有目标的历史轨迹
            candidate_positions: (L, 3) 候选动作位置

        返回:
            density: (L,) 预测目标密度 (有多少目标会出现在附近)
            uncertainty: (L,) 不确定性
        """
        if len(target_histories) == 0:
            return np.zeros(len(candidate_positions)), np.ones(len(candidate_positions))

        with torch.no_grad():
            histories = torch.FloatTensor(target_histories).to(self.device)
            predictions = self.predictor.predict_future(histories, horizon=horizon)
            # predictions: (num_targets, horizon, 3)
            # 取最后一步的预测
            final_preds = predictions[:, -1, :].cpu().numpy()  # (num_targets, 3)

        # 计算每个候选位置附近的目标密度
        density = np.zeros(len(candidate_positions))
        uncertainty = np.zeros(len(candidate_positions))

        for i, pos in enumerate(candidate_positions):
            dists = np.linalg.norm(final_preds - pos, axis=1)
            # 密度 = 距离小于阈值的目标数
            nearby = np.sum(dists < self.config.env.sensor_range)
            density[i] = nearby / max(len(final_preds), 1)
            # 不确定性 = 预测距离的方差
            uncertainty[i] = np.std(dists) if len(dists) > 1 else 1.0

        return density, uncertainty

    def save(self, path):
        torch.save({
            'model_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.predictor.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def reset_data(self):
        self.train_data.clear()
