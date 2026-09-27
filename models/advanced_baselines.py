"""
高级Baseline模型 - 完整实现
基于论文原文，用PyG实现完整版本（非简化）

包含模型：
1. DCRNN - Diffusion Convolutional Recurrent Neural Network (ICLR 2018)
2. ASTGCN - Attention Based Spatial-Temporal Graph Convolutional Networks (AAAI 2019)
3. MTGNN - Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks (KDD 2020)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import math


# ============================================================================
# 1. DCRNN - Diffusion Convolutional Recurrent Neural Network
# 论文: "Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting" (ICLR 2018)
# ============================================================================

class DiffusionGraphConv(nn.Module):
    """
    扩散图卷积层
    实现K步随机游走的扩散过程
    
    论文公式：
    H = Σ_{k=0}^{K} (θ_k,1 * (D_O^{-1}W)^k + θ_k,2 * (D_I^{-1}W^T)^k) * X
    
    其中：
    - D_O: 出度对角矩阵
    - D_I: 入度对角矩阵  
    - W: 邻接矩阵
    - K: 扩散步数
    """
    def __init__(self, in_channels, out_channels, K=2):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 前向和后向扩散的参数
        self.weight_forward = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.weight_backward = nn.Parameter(torch.Tensor(K + 1, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_forward)
        nn.init.xavier_uniform_(self.weight_backward)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # 计算度归一化矩阵
        row, col = edge_index
        deg_out = degree(row, num_nodes, dtype=x.dtype)  # 出度
        deg_in = degree(col, num_nodes, dtype=x.dtype)   # 入度
        
        deg_out_inv = deg_out.pow(-1)
        deg_out_inv[deg_out_inv == float('inf')] = 0
        deg_in_inv = deg_in.pow(-1)
        deg_in_inv[deg_in_inv == float('inf')] = 0
        
        # 前向扩散: D_O^{-1}W
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        norm_forward = deg_out_inv[row] * edge_weight
        
        # 后向扩散: D_I^{-1}W^T (交换row和col)
        norm_backward = deg_in_inv[col] * edge_weight
        
        # K步扩散
        out = torch.zeros(num_nodes, self.out_channels, device=x.device)
        
        # 前向扩散
        x_forward = x
        for k in range(self.K + 1):
            out += torch.matmul(x_forward, self.weight_forward[k])
            if k < self.K:
                # 扩散一步
                x_forward = self._propagate(x_forward, edge_index, norm_forward, num_nodes)
        
        # 后向扩散（反向边）
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
        x_backward = x
        for k in range(self.K + 1):
            out += torch.matmul(x_backward, self.weight_backward[k])
            if k < self.K:
                x_backward = self._propagate(x_backward, edge_index_t, norm_backward, num_nodes)
        
        return out + self.bias
    
    def _propagate(self, x, edge_index, norm, num_nodes):
        """消息传播"""
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])
        return out


class DCGRUCell(nn.Module):
    """
    扩散卷积GRU单元
    将扩散图卷积集成到GRU中
    
    论文公式：
    r_t = σ(Θ_r * [X_t, H_{t-1}] + b_r)
    u_t = σ(Θ_u * [X_t, H_{t-1}] + b_u)
    C_t = tanh(Θ_c * [X_t, r_t ⊙ H_{t-1}] + b_c)
    H_t = u_t ⊙ H_{t-1} + (1 - u_t) ⊙ C_t
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, K=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        
        # Reset gate
        self.graph_conv_r = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
        
        # Update gate
        self.graph_conv_u = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
        
        # Candidate state
        self.graph_conv_c = DiffusionGraphConv(input_dim + hidden_dim, hidden_dim, K)
    
    def forward(self, x, h, edge_index, edge_weight=None):
        """
        Args:
            x: [num_nodes, input_dim]
            h: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            h_new: [num_nodes, hidden_dim]
        """
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=-1)
        
        # Reset gate
        r = torch.sigmoid(self.graph_conv_r(combined, edge_index, edge_weight))
        
        # Update gate
        u = torch.sigmoid(self.graph_conv_u(combined, edge_index, edge_weight))
        
        # Candidate hidden state
        combined_r = torch.cat([x, r * h], dim=-1)
        c = torch.tanh(self.graph_conv_c(combined_r, edge_index, edge_weight))
        
        # New hidden state
        h_new = u * h + (1 - u) * c
        
        return h_new


class DCRNN(nn.Module):
    """
    完整的DCRNN模型
    
    架构：
    - Encoder: 2层 DCGRU
    - Decoder: 2层 DCGRU (seq2seq)
    
    论文: Li et al., "Diffusion Convolutional Recurrent Neural Network:
           Data-Driven Traffic Forecasting", ICLR 2018
    """
    def __init__(self, input_dim, hidden_dim, num_nodes, K=2, num_layers=2, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Encoder layers
        self.encoder_cells = nn.ModuleList([
            DCGRUCell(
                input_dim if i == 0 else hidden_dim,
                hidden_dim,
                num_nodes,
                K
            ) for i in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch, seq_len, num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [batch, num_nodes] - 预测下一个时间步
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        
        # 初始化隐藏状态
        h = [torch.zeros(batch_size, num_nodes, self.hidden_dim, device=x.device)
             for _ in range(self.num_layers)]
        
        # Encoder
        for t in range(seq_len):
            x_t = x[:, t, :, :]  # [batch, num_nodes, input_dim]
            
            for layer in range(self.num_layers):
                # 对batch中的每个样本处理
                h_new = []
                for b in range(batch_size):
                    h_b = self.encoder_cells[layer](
                        x_t[b] if layer == 0 else h[layer - 1][b],
                        h[layer][b],
                        edge_index,
                        edge_weight
                    )
                    h_new.append(h_b)
                h[layer] = torch.stack(h_new, dim=0)
                
                if layer < self.num_layers - 1:
                    h[layer] = F.dropout(h[layer], p=self.dropout, training=self.training)
        
        # 使用最后一层的隐藏状态预测
        output = self.output_proj(h[-1])  # [batch, num_nodes, 1]
        
        return output.squeeze(-1)  # [batch, num_nodes]


# ============================================================================
# 2. ASTGCN - Attention Based Spatial-Temporal Graph Convolutional Networks
# 论文: "Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting" (AAAI 2019)
# ============================================================================

class SpatialAttention(nn.Module):
    """
    空间注意力机制
    学习不同节点之间的重要性
    
    论文公式：
    S = V_s * σ(((X^T * U_1) * U_2) * (U_3 * X) + b_s)
    S' = softmax(S)
    """
    def __init__(self, num_nodes, num_features, num_time_steps):
        super().__init__()
        self.W1 = nn.Parameter(torch.Tensor(num_time_steps))
        self.W2 = nn.Parameter(torch.Tensor(num_features, num_time_steps))
        self.W3 = nn.Parameter(torch.Tensor(num_features))
        self.bs = nn.Parameter(torch.Tensor(1, num_nodes, num_nodes))
        self.Vs = nn.Parameter(torch.Tensor(num_nodes, num_nodes))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.W1)
        nn.init.xavier_uniform_(self.W2)
        nn.init.uniform_(self.W3)
        nn.init.zeros_(self.bs)
        nn.init.xavier_uniform_(self.Vs)
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, num_features, num_time_steps]
        Returns:
            [batch, num_nodes, num_nodes]
        """
        # 计算空间注意力分数
        lhs = torch.matmul(torch.matmul(x.permute(0, 1, 3, 2), self.W2), self.W1)  # [batch, num_nodes, 1]
        rhs = torch.matmul(self.W3, x).squeeze(-1)  # [batch, num_nodes, num_features]
        
        product = torch.matmul(lhs, rhs.unsqueeze(1))  # [batch, num_nodes, num_nodes]
        S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))
        
        # Softmax归一化
        S = F.softmax(S, dim=-1)
        
        return S


class TemporalAttention(nn.Module):
    """
    时间注意力机制
    学习不同时间步的重要性
    
    论文公式：
    E = V_e * σ(((X * U_1) * U_2) * (U_3 * X^T) + b_e)
    E' = softmax(E)
    """
    def __init__(self, num_nodes, num_features, num_time_steps):
        super().__init__()
        self.U1 = nn.Parameter(torch.Tensor(num_nodes))
        self.U2 = nn.Parameter(torch.Tensor(num_features, num_nodes))
        self.U3 = nn.Parameter(torch.Tensor(num_features))
        self.be = nn.Parameter(torch.Tensor(1, num_time_steps, num_time_steps))
        self.Ve = nn.Parameter(torch.Tensor(num_time_steps, num_time_steps))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.U1)
        nn.init.xavier_uniform_(self.U2)
        nn.init.uniform_(self.U3)
        nn.init.zeros_(self.be)
        nn.init.xavier_uniform_(self.Ve)
    
    def forward(self, x):
        """
        Args:
            x: [batch, num_nodes, num_features, num_time_steps]
        Returns:
            [batch, num_time_steps, num_time_steps]
        """
        # 计算时间注意力分数
        lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U2), self.U1)  # [batch, num_time_steps, 1]
        rhs = torch.matmul(self.U3, x.permute(0, 2, 1, 3))  # [batch, num_features, num_time_steps]
        
        product = torch.matmul(lhs, rhs.permute(0, 2, 1).unsqueeze(2))  # [batch, num_time_steps, num_time_steps]
        E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))
        
        # Softmax归一化
        E = F.softmax(E, dim=-1)
        
        return E


class ChebConv(nn.Module):
    """
    切比雪夫图卷积
    使用切比雪夫多项式近似谱图卷积
    
    论文公式：
    g_θ * x ≈ Σ_{k=0}^{K-1} θ_k * T_k(L̃) * x
    其中 L̃ = 2L/λ_max - I
    """
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [num_nodes, in_channels]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        
        # 归一化拉普拉斯矩阵
        row, col = edge_index
        deg = degree(col, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=x.device)
        
        norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        
        # 切比雪夫递归
        Tx_0 = x
        out = torch.matmul(Tx_0, self.weight[0])
        
        if self.K > 1:
            Tx_1 = self._propagate(x, edge_index, norm, num_nodes)
            out += torch.matmul(Tx_1, self.weight[1])
            
            for k in range(2, self.K):
                Tx_2 = 2 * self._propagate(Tx_1, edge_index, norm, num_nodes) - Tx_0
                out += torch.matmul(Tx_2, self.weight[k])
                Tx_0, Tx_1 = Tx_1, Tx_2
        
        return out + self.bias
    
    def _propagate(self, x, edge_index, norm, num_nodes):
        """切比雪夫传播"""
        row, col = edge_index
        out = torch.zeros_like(x)
        out.index_add_(0, row, norm.view(-1, 1) * x[col])
        return out


class STBlock(nn.Module):
    """
    时空注意力块
    结合空间注意力、时间注意力和切比雪夫图卷积
    """
    def __init__(self, num_nodes, in_channels, out_channels, num_time_steps, K=3):
        super().__init__()
        self.spatial_attention = SpatialAttention(num_nodes, in_channels, num_time_steps)
        self.temporal_attention = TemporalAttention(num_nodes, in_channels, num_time_steps)
        self.cheb_conv = ChebConv(in_channels, out_channels, K)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.ln = nn.LayerNorm(out_channels)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch, num_nodes, in_channels, num_time_steps]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [batch, num_nodes, out_channels, num_time_steps]
        """
        batch_size, num_nodes, in_channels, num_time_steps = x.shape
        
        # 空间注意力
        S = self.spatial_attention(x)  # [batch, num_nodes, num_nodes]
        
        # 时间注意力
        E = self.temporal_attention(x)  # [batch, num_time_steps, num_time_steps]
        
        # 应用注意力
        x_sa = torch.matmul(S, x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # [batch, num_nodes, in_channels, num_time_steps]
        x_sta = torch.matmul(x_sa.permute(0, 1, 3, 2), E).permute(0, 1, 3, 2)  # [batch, num_nodes, in_channels, num_time_steps]
        
        # 图卷积
        x_reshaped = x_sta.permute(0, 3, 1, 2).contiguous().view(-1, num_nodes, in_channels)  # [batch*time_steps, num_nodes, in_channels]
        out = []
        for t in range(num_time_steps):
            out_t = self.cheb_conv(x_reshaped[t::num_time_steps], edge_index, edge_weight)
            out.append(out_t)
        out = torch.stack(out, dim=1)  # [batch, num_time_steps, num_nodes, out_channels]
        out = out.permute(0, 2, 3, 1)  # [batch, num_nodes, out_channels, num_time_steps]
        
        # 残差连接
        residual = self.residual_conv(x.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        out = out + residual
        
        # Layer normalization
        out = self.ln(out.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        return F.relu(out)


class ASTGCN(nn.Module):
    """
    完整的ASTGCN模型
    
    架构：
    - 3个时空注意力块
    - 输出层
    
    论文: Guo et al., "Attention Based Spatial-Temporal Graph Convolutional Networks
           for Traffic Flow Forecasting", AAAI 2019
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, num_time_steps, K=3, num_blocks=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_blocks = num_blocks
        
        # ST blocks
        self.st_blocks = nn.ModuleList([
            STBlock(num_nodes, input_dim if i == 0 else hidden_dim, hidden_dim, num_time_steps, K)
            for i in range(num_blocks)
        ])
        
        # 输出层
        self.output_conv = nn.Conv2d(hidden_dim * num_blocks, 1, kernel_size=(1, num_time_steps))
        self.output_fc = nn.Linear(num_nodes, num_nodes)
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x: [batch, num_nodes, input_dim, num_time_steps] 或 [batch, seq_len, num_nodes, input_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [batch, num_nodes]
        """
        # 如果输入是 [batch, seq_len, num_nodes, input_dim]，转换为 [batch, num_nodes, input_dim, seq_len]
        if x.dim() == 4 and x.size(1) != self.num_nodes:
            x = x.permute(0, 2, 3, 1)
        
        # 通过ST blocks
        outs = []
        for block in self.st_blocks:
            x = block(x, edge_index, edge_weight)
            outs.append(x)
        
        # 拼接所有块的输出
        x = torch.cat(outs, dim=2)  # [batch, num_nodes, hidden_dim*num_blocks, num_time_steps]
        
        # 输出预测
        x = x.permute(0, 2, 1, 3)  # [batch, hidden_dim*num_blocks, num_nodes, num_time_steps]
        x = self.output_conv(x).squeeze(1)  # [batch, num_nodes]
        x = self.output_fc(x)
        
        return x


# ============================================================================
# 3. MTGNN - Graph Neural Networks for Multivariate Time Series
# 论文: "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks" (KDD 2020)
# ============================================================================

class GraphLearning(nn.Module):
    """
    图结构学习模块
    从数据中自动学习图结构
    
    论文公式：
    A = softmax(ReLU(E1 * E2^T))
    """
    def __init__(self, num_nodes, hidden_dim):
        super().__init__()
        self.emb1 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.emb2 = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb1)
        nn.init.xavier_uniform_(self.emb2)
    
    def forward(self):
        """
        Returns:
            adj: [num_nodes, num_nodes] 学习到的邻接矩阵
        """
        # 计算节点相似度
        adj = torch.matmul(F.relu(self.emb1), self.emb2.t())
        
        # Softmax归一化
        adj = F.softmax(F.relu(adj), dim=-1)
        
        return adj


class MixProp(nn.Module):
    """
    混合传播层
    结合局部和全局的图信息传播
    
    论文公式：
    H = Σ_k α_k * A^k * X
    """
    def __init__(self, in_channels, out_channels, num_nodes, dropout=0.3):
        super().__init__()
        self.mlp = nn.Linear((2 + 1) * in_channels, out_channels)  # +1 for self-loop
        self.dropout = dropout
    
    def forward(self, x, adj):
        """
        Args:
            x: [batch, num_nodes, in_channels]
            adj: [num_nodes, num_nodes]
        Returns:
            [batch, num_nodes, out_channels]
        """
        # 自连接
        out = [x]
        
        # 一阶邻居
        out.append(torch.matmul(adj, x))
        
        # 二阶邻居
        out.append(torch.matmul(torch.matmul(adj, adj), x))
        
        # 拼接
        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        return h


class DilatedCausalConv(nn.Module):
    """
    因果膨胀卷积
    捕捉长时序依赖，保持因果性（不看未来）
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
    
    def forward(self, x):
        """
        Args:
            x: [batch, in_channels, seq_len]
        Returns:
            [batch, out_channels, seq_len]
        """
        x = self.conv(x)
        # 因果性：去掉未来的信息
        x = x[:, :, :x.size(2) - self.conv.padding[0]]
        return x


class MTGNN(nn.Module):
    """
    完整的MTGNN模型
    
    架构：
    - 图结构学习
    - 膨胀因果卷积 + 图卷积
    - 跳跃连接
    
    论文: Wu et al., "Connecting the Dots: Multivariate Time Series Forecasting
           with Graph Neural Networks", KDD 2020
    """
    def __init__(self, num_nodes, input_dim, hidden_dim, num_layers=3, dropout=0.3):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        
        # 图学习
        self.graph_learning = GraphLearning(num_nodes, hidden_dim)
        
        # 输入投影
        self.start_conv = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        
        # 膨胀卷积层
        self.dilated_convs = nn.ModuleList([
            DilatedCausalConv(hidden_dim, hidden_dim, kernel_size=2, dilation=2**i)
            for i in range(num_layers)
        ])
        
        # 图卷积层
        self.graph_convs = nn.ModuleList([
            MixProp(hidden_dim, hidden_dim, num_nodes, dropout)
            for _ in range(num_layers)
        ])
        
        # 跳跃连接
        self.skip_convs = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.end_conv_1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self.end_conv_2 = nn.Conv1d(hidden_dim, 1, kernel_size=1)
    
    def forward(self, x, edge_index=None, edge_weight=None):
        """
        Args:
            x: [batch, seq_len, num_nodes, input_dim]
            edge_index: 忽略（自己学习图）
            edge_weight: 忽略
        Returns:
            [batch, num_nodes]
        """
        batch_size, seq_len, num_nodes, input_dim = x.shape
        
        # 学习图结构
        adj = self.graph_learning()  # [num_nodes, num_nodes]
        
        # 转换输入形状
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size * num_nodes, input_dim, seq_len)
        
        # 起始卷积
        x = self.start_conv(x)  # [batch*num_nodes, hidden_dim, seq_len]
        
        # 跳跃连接累加器
        skip = 0
        
        # 通过层
        for i in range(self.num_layers):
            # 膨胀因果卷积
            residual = x
            x = self.dilated_convs[i](x)
            x = F.relu(x)
            
            # 调整形状用于图卷积
            x_graph = x.permute(0, 2, 1).contiguous().view(batch_size, seq_len, num_nodes, -1)
            x_graph = x_graph[:, -1, :, :]  # 取最后时间步 [batch, num_nodes, hidden_dim]
            
            # 图卷积
            x_graph = self.graph_convs[i](x_graph, adj)  # [batch, num_nodes, hidden_dim]
            
            # 转回时序形状
            x_graph = x_graph.unsqueeze(1).expand(-1, seq_len, -1, -1)
            x_graph = x_graph.contiguous().view(batch_size * num_nodes, seq_len, -1).permute(0, 2, 1)
            
            # 残差连接
            x = x + x_graph + residual
            
            # 跳跃连接
            s = self.skip_convs[i](x)
            skip = skip + s
        
        # 输出
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [batch*num_nodes, 1, seq_len]
        
        # 取最后时间步
        x = x[:, :, -1].view(batch_size, num_nodes)
        
        return x


# ============================================================================
# 适配器：统一接口
# ============================================================================

class AdvancedBaselineWrapper(nn.Module):
    """
    包装器，统一不同模型的接口
    """
    def __init__(self, model_type, **kwargs):
        super().__init__()
        
        if model_type == 'dcrnn':
            self.model = DCRNN(**kwargs)
        elif model_type == 'astgcn':
            self.model = ASTGCN(**kwargs)
        elif model_type == 'mtgnn':
            self.model = MTGNN(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
    
    def forward(self, x, edge_index, edge_weight=None):
        """
        统一接口
        
        Args:
            x: [batch, seq_len, num_nodes, feature_dim]
            edge_index: [2, num_edges]
            edge_weight: [num_edges]
        Returns:
            [batch, num_nodes]
        """
        return self.model(x, edge_index, edge_weight)

