"""
编码器 — 论文 Section IV-E
基于注意力的编码器, 学习候选动作间的依赖关系, 形成观测上下文
输入: 特征矩阵 M_t (L, feature_dim)
输出: 上下文特征 (L, hidden_dim)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """位置编码 (为候选动作添加位置信息)"""

    def __init__(self, d_model, max_len=128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class EncoderLayer(nn.Module):
    """编码器层: 多头自注意力 + 前馈网络"""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, L, d_model)
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    """
    编码器: 将特征矩阵编码为上下文特征
    输入: (batch, L, feature_dim)
    输出: (batch, L, hidden_dim)
    """

    def __init__(self, config):
        super().__init__()
        self.feature_dim = config.network.feature_dim
        self.hidden_dim = config.network.encoder_hidden
        self.num_heads = config.network.encoder_num_heads
        self.num_layers = config.network.encoder_num_layers
        self.dropout = config.network.encoder_dropout

        # 输入投影
        self.input_proj = nn.Linear(self.feature_dim, self.hidden_dim)

        # 位置编码
        self.pos_enc = PositionalEncoding(self.hidden_dim, max_len=128)

        # 编码器层
        self.layers = nn.ModuleList([
            EncoderLayer(self.hidden_dim, self.num_heads, self.hidden_dim * 4, self.dropout)
            for _ in range(self.num_layers)
        ])

    def forward(self, features):
        """
        参数:
            features: (batch, L, feature_dim)

        返回:
            context: (batch, L, hidden_dim) 上下文特征
            attention_weights: list of (batch, L, L) 各层注意力权重
        """
        x = self.input_proj(features)
        x = self.pos_enc(x)

        attention_weights = []
        for layer in self.layers:
            x = layer(x)

        return x
