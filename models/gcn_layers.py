"""
图卷积层实现
包括GCN和GAT层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCNLayer(nn.Module):
    """图卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: str = 'relu',
        dropout: float = 0.3
    ):
        """
        初始化GCN层
        
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            activation: 激活函数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        参数:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges]
            
        返回:
            输出特征 [num_nodes, out_channels]
        """
        x = self.conv(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class GATLayer(nn.Module):
    """图注意力层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        activation: str = 'relu',
        dropout: float = 0.3,
        concat: bool = True
    ):
        """
        初始化GAT层
        
        参数:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            heads: 注意力头数
            activation: 激活函数
            dropout: Dropout比例
            concat: 是否拼接多头输出
        """
        super().__init__()
        
        self.conv = GATConv(
            in_channels,
            out_channels,
            heads=heads,
            dropout=dropout,
            concat=concat
        )
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        参数:
            x: 节点特征
            edge_index: 边索引
            edge_weight: 边权重
            
        返回:
            输出特征
        """
        x = self.conv(x, edge_index)
        x = self.activation(x)
        x = self.dropout(x)
        return x

