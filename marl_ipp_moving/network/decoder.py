"""
解码器 — 论文 Section IV-E
利用编码器的上下文特征 + 规划状态 + 预算掩码
输出: 候选动作的概率分布 + 值函数估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PlanningStateEncoder(nn.Module):
    """规划状态编码器: 编码已执行路径 + 剩余预算"""

    def __init__(self, config):
        super().__init__()
        self.action_dim = config.network.action_dim
        self.state_dim = config.network.state_dim
        self.hidden_dim = config.network.decoder_hidden

        # 路径编码 (LSTM处理已执行的动作序列)
        self.path_lstm = nn.LSTM(
            input_size=self.action_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            num_layers=1
        )

        # 预算编码
        self.budget_proj = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, self.hidden_dim)
        )

        # 融合
        self.fusion = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.state_dim),
            nn.GELU()
        )

    def forward(self, path_history, remaining_budget):
        """
        参数:
            path_history: (batch, seq_len, action_dim) 已执行路径
            remaining_budget: (batch, 1) 剩余预算

        返回:
            planning_state: (batch, state_dim) 规划状态编码
        """
        if path_history.size(1) == 0:
            # 空路径
            batch_size = path_history.size(0)
            path_feat = torch.zeros(batch_size, self.hidden_dim, device=path_history.device)
        else:
            _, (h_n, _) = self.path_lstm(path_history)
            path_feat = h_n[-1]  # (batch, hidden_dim)

        budget_feat = self.budget_proj(remaining_budget)

        planning_state = self.fusion(torch.cat([path_feat, budget_feat], dim=-1))
        return planning_state


class DecoderLayer(nn.Module):
    """解码器层: 交叉注意力 + 前馈网络"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, context):
        """
        参数:
            query: (batch, 1, d_model) 来自规划状态
            context: (batch, L, d_model) 来自编码器
        """
        attn_out, _ = self.cross_attn(query, context, context)
        x = self.norm1(query + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Decoder(nn.Module):
    """
    解码器: 输出动作概率分布和值函数

    输入:
        - context: (batch, L, hidden_dim) 编码器输出
        - planning_state: (batch, state_dim) 规划状态
        - budget_mask: (batch, L) 预算掩码

    输出:
        - action_logits: (batch, L) 动作logits
        - value: (batch, 1) 状态值函数
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.network.decoder_hidden
        self.state_dim = config.network.state_dim
        self.num_heads = config.network.decoder_num_heads
        self.num_layers = config.network.decoder_num_layers

        # 规划状态投影
        self.state_proj = nn.Linear(self.state_dim, self.hidden_dim)

        # 解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4)
            for _ in range(self.num_layers)
        ])

        # 动作打分 (context -> scalar score)
        self.action_scorer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1)
        )

        # 值函数估计
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, config.network.value_hidden),
            nn.GELU(),
            nn.Linear(config.network.value_hidden, 1)
        )

    def forward(self, context, planning_state, budget_mask):
        """
        返回:
            action_logits: (batch, L) 被budget_mask屏蔽后的logits
            value: (batch, 1) 状态值
        """
        # 规划状态作为query
        query = self.state_proj(planning_state).unsqueeze(1)  # (batch, 1, hidden)

        # 解码
        for layer in self.layers:
            query = layer(query, context)

        # 动作打分: 对每个候选动作计算得分
        # 使用query和context的交互
        # 扩展query: (batch, L, hidden)
        query_expanded = query.expand(-1, context.size(1), -1)

        # 融合query和context
        fused = query_expanded + context  # (batch, L, hidden)

        action_logits = self.action_scorer(fused).squeeze(-1)  # (batch, L)

        # 应用预算掩码 (被屏蔽的动作logit设为-inf)
        action_logits = action_logits.masked_fill(budget_mask == 0, float('-inf'))

        # 值函数 (使用query的输出)
        value = self.value_head(query.squeeze(1))  # (batch, 1)

        return action_logits, value
