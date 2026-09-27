"""
时间卷积层实现
包括TCN和LSTM层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNLayer(nn.Module):
    """时间卷积网络层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        activation: str = 'relu',
        dropout: float = 0.3
    ):
        """
        初始化TCN层
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            dilation: 膨胀系数
            activation: 激活函数
            dropout: Dropout比例
        """
        super().__init__()
        
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        
        self.norm = nn.BatchNorm1d(out_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            self.activation = nn.Identity()
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征 [batch_size, in_channels, seq_len]
            
        返回:
            输出特征 [batch_size, out_channels, seq_len]
        """
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class LSTMLayer(nn.Module):
    """LSTM层"""
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        """
        初始化LSTM层
        
        参数:
            input_size: 输入特征维度
            hidden_size: 隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比例
            bidirectional: 是否双向
        """
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入序列 [batch_size, seq_len, input_size]
            
        返回:
            输出序列 [batch_size, seq_len, hidden_size * num_directions]
        """
        output, (hn, cn) = self.lstm(x)
        return output

