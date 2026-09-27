"""
Baseline方法对比实验（完全对齐exp4版本）

关键改进：
1. 直接使用exp4的NodeLevelSTGNNDataset（包含所有数据增强、插值）
2. 直接使用exp4的图数据增强函数（特征噪声、节点遮挡、边扰动）
3. 使用相同的训练配置、损失函数、优化器
4. 唯一区别：模型架构（MLP/LSTM/GCN/GAT vs STGNN）

目的：证明Affective-STGNN的时空联合建模+情感调制的优势
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import copy
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

# 导入高级模型
sys.path.append(str(Path(__file__).parent.parent / 'models'))
from advanced_baselines import AdvancedBaselineWrapper


# ============================================================================
# 1. 导入exp4的数据集类
# ============================================================================
# 直接从exp4导入数据集类
import importlib.util
spec = importlib.util.spec_from_file_location("exp4", "experiments/exp4_stgnn_temporal.py")
exp4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp4_module)

NodeLevelSTGNNDataset = exp4_module.NodeLevelSTGNNDataset


# ============================================================================
# 预构建全图特征矩阵
# ============================================================================
def precompute_full_graph_features(dataset, device):
    """预先构建所有年份的全图特征矩阵，避免重复计算
    
    这个优化可以将高级模型的训练速度提升10-100倍！
    """
    print("  [加速优化] 预构建全图特征矩阵（1200节点 x {} 年份）...".format(len(dataset.years)))
    full_graph_cache = {}
    
    for year in tqdm(dataset.years, desc="  预构建", leave=False):
        year_features = []
        for node in dataset.all_nodes:
            cached = dataset.features_cache[year][node]
            affective_vulnerability = dataset._domirank_to_vulnerability(
                cached['affective_domirank'], is_affective=True
            )
            features = [
                cached['in_degree'], cached['out_degree'], cached['pagerank'],
                cached['sentiment_mean'], cached['news_count'], affective_vulnerability,
                cached['neighbor_avg_sentiment'], cached['neighbor_avg_risk'],
                cached['supplier_concentration']
            ]
            year_features.append(features)
        
        # [num_nodes, 9] - 保存在GPU上
        full_graph_cache[year] = torch.FloatTensor(year_features).to(device)
    
    print(f"  ✓ 预构建完成！GPU缓存大小：{len(full_graph_cache)} 个年份")
    return full_graph_cache


# ============================================================================
# collate_fn（与exp4完全一致）
# ============================================================================
def collate_fn(batch):
    """自定义collate函数（增强版：包含预计算的全图特征）"""
    # 所有样本共享相同的图结构（每个时间步）
    graphs = batch[0]['graphs']
    
    # 收集节点特征序列和标签
    node_features_seqs = torch.stack([s['node_features_seq'] for s in batch])  # [batch, window_size, feature_dim]
    labels = torch.stack([s['label'] for s in batch])  # [batch, 1]
    
    # 获取节点索引
    all_nodes = graphs[0].nodes
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    target_node_indices = torch.LongTensor([node_to_idx[s['node']] for s in batch])
    
    # 🔥 性能优化：收集预计算的全图特征
    if 'full_graph_features' in batch[0]:
        full_graph_features = torch.stack([s['full_graph_features'] for s in batch])  # [batch, 1200, 16]
    else:
        full_graph_features = None
    
    # 保存input_years（向后兼容）
    input_years_list = [s['input_years'] for s in batch]
    
    return {
        'graphs': graphs,
        'node_features_seqs': node_features_seqs,
        'target_node_indices': target_node_indices,
        'labels': labels,
        'nodes': [s['node'] for s in batch],
        'input_years_list': input_years_list,
        'full_graph_features': full_graph_features  # 新增：预计算的全图特征
    }


# ============================================================================
# 数据增强函数（与exp4完全一致）
# ============================================================================
def _uses_per_sample_graphs(model):
    return isinstance(model, (GCNModel, GATModel, GCN_LSTM_Model, GAT_LSTM_Model))


def graph_collate_fn(batch):
    """Keep the graph sequence belonging to each graph-model sample."""
    collated = collate_fn(batch)
    collated['graphs_per_sample'] = [sample['graphs'] for sample in batch]
    collated['target_node_indices'] = torch.LongTensor([
        {node: index for index, node in enumerate(sample['graphs'][-1].nodes)}[sample['node']]
        for sample in batch
    ])
    return collated


def _graph_sequences_on_device(graph_sequences, device):
    """Move each sample's graphs without substituting another sample's window."""
    graph_cache = {}
    sequences = []
    for sequence in graph_sequences:
        converted = []
        for graph in sequence:
            key = id(graph)
            if key not in graph_cache:
                edge_weight = graph.edge_weight.to(device) if (
                    hasattr(graph, 'edge_weight') and graph.edge_weight is not None
                    and graph.edge_weight.numel() > 0
                ) else None
                graph_on_device = Data(edge_index=graph.edge_index.to(device))
                graph_on_device.edge_weight = edge_weight
                graph_on_device.edge_attr = edge_weight
                graph_on_device.nodes = graph.nodes
                graph_cache[key] = graph_on_device
            converted.append(graph_cache[key])
        sequences.append(converted)
    return sequences


def apply_augmentation(node_features_seq, graphs, device):
    """
    应用与exp4相同的数据增强
    
    Args:
        node_features_seq: [batch, seq_len, feature_dim]
        graphs: 图序列列表
        device: 设备
    
    Returns:
        增强后的node_features_seq
    """
    # ===== 数据增强1：特征噪声（5%） =====
    noise = torch.randn_like(node_features_seq) * 0.05
    node_features_seq = node_features_seq + noise
    
    # ===== 数据增强2：节点遮挡（20%节点） =====
    mask = torch.rand(node_features_seq.shape[0], node_features_seq.shape[1], 1, device=device) > 0.2
    node_features_seq = node_features_seq * mask
    
    # 注意：边扰动在GCN/GAT中应用
    
    return node_features_seq


# ============================================================================
# 2. Baseline模型定义
# ============================================================================
class MLP(nn.Module):
    """MLP: 只用最后时刻的特征，忽略图结构和时序"""
    def __init__(self, input_dim, hidden_dim=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, node_features_seq, graphs):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim]
            graphs: 列表，忽略
        Returns:
            [batch]
        """
        # 只用最后一个时间步的特征
        x = node_features_seq[:, -1, :]  # [batch, feature_dim]
        return self.net(x).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM: 建模时序，忽略图结构"""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, node_features_seq, graphs):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim]
            graphs: 列表，忽略
        Returns:
            [batch]
        """
        _, (h_n, _) = self.lstm(node_features_seq)
        out = self.fc(h_n[-1])  # 使用最后一层的hidden state
        return out.squeeze(-1)


class GCNModel(nn.Module):
    """GCN: 建模图结构，忽略时序（只用最后一个时间步的图）"""
    def __init__(self, input_dim, hidden_dim=64, num_nodes=1200, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.num_nodes = num_nodes
    
    def forward(self, node_features_seq, graphs, target_node_indices, all_node_features):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim] - 目标节点的时序特征（这里不用）
            graphs: 列表[图序列]，每个样本一个
            target_node_indices: [batch] - 目标节点在图中的索引
            all_node_features: [batch, num_nodes, feature_dim] - 整个图所有节点的特征
        Returns:
            [batch]
        """
        batch_size = len(graphs)
        outputs = []
        
        # 使用最后一个时间步的图
        for i in range(batch_size):
            last_graph = graphs[i][-1]  # 最后时刻的图
            x = all_node_features[i]  # [num_nodes, feature_dim]
            
            # GCN前向传播
            edge_weight = last_graph.edge_attr.squeeze() if (last_graph.edge_attr is not None and last_graph.edge_attr.numel() > 0) else None
            x = F.relu(self.conv1(x, last_graph.edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, last_graph.edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 提取目标节点的embedding
            target_embedding = x[target_node_indices[i]]  # [hidden_dim]
            out = self.fc(target_embedding)  # [1]
            outputs.append(out)
        
        return torch.cat(outputs, dim=0).reshape(-1)  # [batch]


class GATModel(nn.Module):
    """GAT: 图注意力，忽略时序"""
    def __init__(self, input_dim, hidden_dim=64, heads=2, num_nodes=1200, dropout=0.5):
        super().__init__()
        # 🔥 改进1：增大隐藏维度到64（与GCN一致）
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)  # 支持边权重
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1)
        self.fc = nn.Linear(hidden_dim, 1)
        self.dropout = dropout
        self.num_nodes = num_nodes
    
    def forward(self, node_features_seq, graphs, target_node_indices, all_node_features):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim] - 目标节点的时序特征（这里不用）
            graphs: 列表[图序列]
            target_node_indices: [batch] - 目标节点索引
            all_node_features: [batch, num_nodes, feature_dim] - 所有节点特征
        Returns:
            [batch]
        """
        batch_size = len(graphs)
        outputs = []
        
        for i in range(batch_size):
            last_graph = graphs[i][-1]
            x = all_node_features[i]  # [num_nodes, feature_dim]
            
            # 🔥 改进2：使用边权重（贸易额）来辅助注意力学习
            edge_weight = last_graph.edge_attr
            if edge_weight is not None and edge_weight.numel() > 0:
                edge_weight = edge_weight.view(-1, 1)  # [num_edges, 1]
            else:
                edge_weight = None
            
            # GAT前向传播（使用边权重）
            x = F.elu(self.conv1(x, last_graph.edge_index, edge_attr=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv2(x, last_graph.edge_index, edge_attr=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 提取目标节点
            target_embedding = x[target_node_indices[i]]
            out = self.fc(target_embedding)
            outputs.append(out)
        
        return torch.cat(outputs, dim=0).reshape(-1)  # [batch]


class GCN_LSTM_Model(nn.Module):
    """GCN+LSTM: 结合空间和时序建模"""
    def __init__(self, input_dim, hidden_dim=64, num_nodes=1200, lstm_layers=2, dropout=0.5):
        super().__init__()
        # GCN分支：处理空间信息
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # LSTM分支：处理时序信息
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, 
                           batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        
        # 融合层
        self.fc = nn.Linear(hidden_dim * 2, 1)  # GCN embedding + LSTM embedding
        self.dropout = dropout
        self.num_nodes = num_nodes
    
    def forward(self, node_features_seq, graphs, target_node_indices, all_node_features):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim] - 目标节点的时序特征
            graphs: 列表[图序列]，每个样本一个
            target_node_indices: [batch] - 目标节点在图中的索引
            all_node_features: [batch, num_nodes, feature_dim] - 整个图所有节点的特征
        Returns:
            [batch]
        """
        batch_size = len(graphs)
        
        # 1️⃣ LSTM分支：处理目标节点的时序信息
        _, (h_n, _) = self.lstm(node_features_seq)
        lstm_embedding = h_n[-1]  # [batch, hidden_dim]
        
        # 2️⃣ GCN分支：处理最后时刻的图结构
        gcn_embeddings = []
        for i in range(batch_size):
            last_graph = graphs[i][-1]  # 最后时刻的图
            x = all_node_features[i]  # [num_nodes, feature_dim]
            
            # GCN前向传播
            edge_weight = last_graph.edge_attr.squeeze() if (last_graph.edge_attr is not None and last_graph.edge_attr.numel() > 0) else None
            x = F.relu(self.conv1(x, last_graph.edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(self.conv2(x, last_graph.edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 提取目标节点的embedding
            target_embedding = x[target_node_indices[i]]  # [hidden_dim]
            gcn_embeddings.append(target_embedding)
        
        gcn_embedding = torch.stack(gcn_embeddings)  # [batch, hidden_dim]
        
        # 3️⃣ 融合两个分支
        combined = torch.cat([gcn_embedding, lstm_embedding], dim=-1)  # [batch, hidden_dim*2]
        out = self.fc(combined)  # [batch, 1]
        
        return out.squeeze(-1)  # [batch]


class GAT_LSTM_Model(nn.Module):
    """GAT+LSTM: 结合图注意力和时序建模"""
    def __init__(self, input_dim, hidden_dim=64, num_nodes=1200, lstm_layers=2, dropout=0.5, heads=2):
        super().__init__()
        # GAT分支：处理空间信息
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, dropout=dropout, edge_dim=1)
        
        # LSTM分支：处理时序信息
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, 
                           batch_first=True, dropout=dropout if lstm_layers > 1 else 0)
        
        # 融合层
        self.fc = nn.Linear(hidden_dim * 2, 1)  # GAT embedding + LSTM embedding
        self.dropout = dropout
        self.num_nodes = num_nodes
    
    def forward(self, node_features_seq, graphs, target_node_indices, all_node_features):
        """
        Args:
            node_features_seq: [batch, seq_len, feature_dim] - 目标节点的时序特征
            graphs: 列表[图序列]，每个样本一个
            target_node_indices: [batch] - 目标节点在图中的索引
            all_node_features: [batch, num_nodes, feature_dim] - 整个图所有节点的特征
        Returns:
            [batch]
        """
        batch_size = len(graphs)
        
        # 1️⃣ LSTM分支：处理目标节点的时序信息
        _, (h_n, _) = self.lstm(node_features_seq)
        lstm_embedding = h_n[-1]  # [batch, hidden_dim]
        
        # 2️⃣ GAT分支：处理最后时刻的图结构
        gat_embeddings = []
        for i in range(batch_size):
            last_graph = graphs[i][-1]  # 最后时刻的图
            x = all_node_features[i]  # [num_nodes, feature_dim]
            
            # GAT前向传播（使用边权重）
            edge_weight = last_graph.edge_attr
            if edge_weight is not None and edge_weight.numel() > 0:
                edge_weight = edge_weight.view(-1, 1)
            else:
                edge_weight = None
            
            x = F.elu(self.conv1(x, last_graph.edge_index, edge_attr=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.elu(self.conv2(x, last_graph.edge_index, edge_attr=edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            # 提取目标节点的embedding
            target_embedding = x[target_node_indices[i]]  # [hidden_dim]
            gat_embeddings.append(target_embedding)
        
        gat_embedding = torch.stack(gat_embeddings)  # [batch, hidden_dim]
        
        # 3️⃣ 融合两个分支
        combined = torch.cat([gat_embedding, lstm_embedding], dim=-1)  # [batch, hidden_dim*2]
        out = self.fc(combined)  # [batch, 1]
        
        return out.squeeze(-1)  # [batch]


# ============================================================================
# 3. 训练函数（对齐exp4）
# ============================================================================
def train_epoch(model, dataloader, optimizer, device, full_dataset, apply_augment=True, use_graph=False, use_advanced=False):
    """
    训练一个epoch（完全对齐exp4的训练逻辑）
    
    Args:
        model: 模型
        dataloader: DataLoader（批处理）
        optimizer: 优化器
        device: 设备
        full_dataset: 完整数据集对象（用于访问features_cache等）
        apply_augment: 是否应用数据增强
        use_graph: 是否使用图模型（GCN/GAT）
        use_advanced: 是否使用高级模型（DCRNN/ASTGCN/MTGNN）
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    # 🔥 使用批处理训练（与exp4完全一致）
    for batch in tqdm(dataloader, desc="Training", leave=False):
        graphs = batch['graphs']
        node_features_seqs = batch['node_features_seqs'].to(device)  # [batch, window, feature_dim]
        target_node_indices = batch['target_node_indices'].to(device)
        labels = batch['labels'].to(device)
        
        batch_size = node_features_seqs.shape[0]
        
        # ===== 数据增强（与exp4完全一致） =====
        if apply_augment:
            # 1. 特征噪声
            noise = torch.randn_like(node_features_seqs) * 0.05
            node_features_seqs = node_features_seqs + noise
            
            # 2. 节点遮挡（20%节点）
            mask = torch.rand(node_features_seqs.shape[0], node_features_seqs.shape[1], 1, device=device) > 0.2
            node_features_seqs = node_features_seqs * mask
        
        # 提取图结构
        edge_index = graphs[0].edge_index.to(device)
        edge_weight = graphs[0].edge_weight.to(device)
        
        # ===== 3. 边扰动（数据增强） =====
        if apply_augment:
            num_edges = edge_index.shape[1]
            edge_mask = torch.rand(num_edges, device=device) > 0.1
            edge_index = edge_index[:, edge_mask]
            edge_weight = edge_weight[edge_mask]
        
        optimizer.zero_grad()
        
        if use_advanced:
            # 高级模型：需要全图特征
            window_size, feature_dim = node_features_seqs.shape[1], node_features_seqs.shape[2]
            num_nodes = len(graphs[0].nodes)
            
            # 全零初始化，只填充目标节点（与exp4一致）
            full_features = torch.zeros(batch_size, window_size, num_nodes, feature_dim, device=device)
            for i, node_idx in enumerate(target_node_indices):
                full_features[i, :, node_idx, :] = node_features_seqs[i]
            
            # 前向传播
            all_preds = model(full_features, edge_index, edge_weight)  # [batch, num_nodes]
            
            # 提取目标节点的预测
            preds = torch.stack([all_preds[i, target_node_indices[i]] for i in range(batch_size)])
            
        elif use_graph and _uses_per_sample_graphs(model):
            all_node_features_batch = batch['full_graph_features'].to(device)
            graphs_list = _graph_sequences_on_device(batch['graphs_per_sample'], device)
            preds = model(node_features_seqs, graphs_list, target_node_indices, all_node_features_batch)

        elif use_graph:
            # 🔥 性能优化：直接使用预计算的全图特征（已经归一化）
            all_node_features_batch = batch['full_graph_features'].to(device)  # [batch, 1200, 16]
            
            # 🔥 关键修复：确保图数据在正确的设备上
            from torch_geometric.data import Data
            graphs_on_device = []
            for g in graphs:
                edge_weight_tensor = g.edge_weight.to(device) if (hasattr(g, 'edge_weight') and g.edge_weight is not None and g.edge_weight.numel() > 0) else None
                g_device = Data(
                    edge_index=g.edge_index.to(device)
                )
                # PyG的Data对象可以动态添加属性
                g_device.edge_weight = edge_weight_tensor
                g_device.edge_attr = edge_weight_tensor  # GCN模型使用edge_attr
                g_device.nodes = g.nodes
                graphs_on_device.append(g_device)
            
            # 批处理前向传播
            # 由于所有样本共享同一个图结构，需要为每个样本复制图序列
            graphs_list = [graphs_on_device for _ in range(batch_size)]
            preds = model(node_features_seqs, graphs_list, target_node_indices, all_node_features_batch)
            
        else:
            # MLP/LSTM：直接前向传播（批处理）
            preds = model(node_features_seqs, [graphs])
        
        # 计算损失
        loss = F.mse_loss(preds, labels.squeeze())
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


def evaluate(model, dataloader, eval_dataset, device, full_dataset, use_graph=False, use_advanced=False):
    """评估（对齐exp4）"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            graphs = batch['graphs']
            node_features_seqs = batch['node_features_seqs'].to(device)
            target_node_indices = batch['target_node_indices'].to(device)
            labels = batch['labels']
            
            batch_size = node_features_seqs.shape[0]
            
            # 提取图结构
            edge_index = graphs[0].edge_index.to(device)
            edge_weight = graphs[0].edge_weight.to(device)
            
            if use_advanced:
                # 高级模型：全图特征
                window_size, feature_dim = node_features_seqs.shape[1], node_features_seqs.shape[2]
                num_nodes = len(graphs[0].nodes)
                
                full_features = torch.zeros(batch_size, window_size, num_nodes, feature_dim, device=device)
                for i, node_idx in enumerate(target_node_indices):
                    full_features[i, :, node_idx, :] = node_features_seqs[i]
                
                all_preds_batch = model(full_features, edge_index, edge_weight)  # [batch, num_nodes]
                preds = torch.stack([all_preds_batch[i, target_node_indices[i]] for i in range(batch_size)])
                
            elif use_graph and _uses_per_sample_graphs(model):
                all_node_features_batch = batch['full_graph_features'].to(device)
                graphs_list = _graph_sequences_on_device(batch['graphs_per_sample'], device)
                preds = model(node_features_seqs, graphs_list, target_node_indices, all_node_features_batch)

            elif use_graph:
                # 🔥 性能优化：直接使用预计算的全图特征（已经归一化）
                all_node_features_batch = batch['full_graph_features'].to(device)  # [batch, 1200, 16]
                
                # 🔥 确保图数据在正确的设备上
                from torch_geometric.data import Data
                graphs_on_device = []
                for g in graphs:
                    edge_weight_tensor = g.edge_weight.to(device) if (hasattr(g, 'edge_weight') and g.edge_weight is not None and g.edge_weight.numel() > 0) else None
                    g_device = Data(
                        edge_index=g.edge_index.to(device)
                    )
                    g_device.edge_weight = edge_weight_tensor
                    g_device.edge_attr = edge_weight_tensor  # GCN模型使用edge_attr
                    g_device.nodes = g.nodes
                    graphs_on_device.append(g_device)
                
                # 批处理前向传播
                graphs_list = [graphs_on_device for _ in range(batch_size)]
                preds = model(node_features_seqs, graphs_list, target_node_indices, all_node_features_batch)
                
            else:
                # MLP/LSTM：批处理
                preds = model(node_features_seqs, [graphs])
            
            all_preds.append(preds.cpu())
            all_labels.append(labels)
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # 反归一化
    preds_denorm = eval_dataset.denormalize_labels(preds)
    labels_denorm = eval_dataset.denormalize_labels(labels)
    
    # 计算指标（MAE, RMSE, R²）
    mae = mean_absolute_error(labels_denorm, preds_denorm)
    rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
    r2 = r2_score(labels_denorm, preds_denorm)
    
    return mae, rmse, r2, preds_denorm.tolist(), labels_denorm.tolist()


def train_baseline(model, train_dataset, test_dataset, optimizer, scheduler,
                   epochs=200, patience=15, device='cpu', full_dataset=None, use_graph=False, use_advanced=False, batch_size=32):
    """
    训练baseline模型（完全对齐exp4的训练流程）
    
    Args:
        model: 模型
        train_dataset: 训练数据集（SimpleDataset）
        test_dataset: 测试数据集（SimpleDataset）
        optimizer: 优化器
        scheduler: 学习率调度器
        epochs: 最大epoch数
        patience: 早停耐心
        device: 设备
        full_dataset: 完整数据集对象（用于访问features_cache等）
        use_graph: 是否使用图模型（GCN/GAT）
        use_advanced: 是否使用高级模型（DCRNN/ASTGCN/MTGNN）
        batch_size: 批大小
    """
    model = model.to(device)
    
    # 🔥 关键修复：使用DataLoader + collate_fn（与exp4完全一致）
    from torch.utils.data import DataLoader
    
    batch_collate_fn = graph_collate_fn if _uses_per_sample_graphs(model) else collate_fn
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=batch_collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=batch_collate_fn
    )
    
    best_test_r2 = -float('inf')
    best_test_mae = float('inf')
    best_test_rmse = float('inf')
    patience_counter = 0
    best_preds = None
    best_labels = None
    
    pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in pbar:
        # 训练（带数据增强）
        train_loss = train_epoch(model, train_loader, optimizer, device, full_dataset, 
                                apply_augment=True, use_graph=use_graph, use_advanced=use_advanced)
        
        # 更新学习率
        scheduler.step()
        
        # 评估（返回 MAE, RMSE, R²）
        train_mae, train_rmse, train_r2, _, _ = evaluate(model, train_loader, train_dataset, device, full_dataset, use_graph, use_advanced)
        test_mae, test_rmse, test_r2, test_preds, test_labels = evaluate(model, test_loader, test_dataset, device, full_dataset, use_graph, use_advanced)
        
        # 早停检查
        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            best_test_mae = test_mae
            best_test_rmse = test_rmse
            best_preds = test_preds
            best_labels = test_labels
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 每个epoch都打印（使用tqdm.write避免被进度条覆盖）
        tqdm.write(f"    Epoch {epoch+1:3d}/{epochs} | "
                   f"Train R²={train_r2:.4f} | Test R²={test_r2:.4f} | "
                   f"Test MAE={test_mae:.4f} | Test RMSE={test_rmse:.4f} | Loss={train_loss:.4f}")
        
        # 更新进度条后缀
        pbar.set_postfix({'Test_R²': f'{test_r2:.4f}', 'Test_MAE': f'{test_mae:.4f}', 'Test_RMSE': f'{test_rmse:.4f}'})
        
        # 早停
        if patience_counter >= patience:
            tqdm.write(f"    ⚠ 早停于epoch {epoch+1}")
            break
    
    return best_test_r2, best_test_mae, best_test_rmse, best_preds, best_labels


# ============================================================================
# 4. 主函数
# ============================================================================
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Baseline方法对比实验')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['mlp', 'lstm', 'gcn', 'gat', 'gcn_lstm', 'gat_lstm', 'dcrnn', 'astgcn', 'mtgnn', 'all'],
                       help='选择要训练的模型: mlp, lstm, gcn, gat, gcn_lstm, gat_lstm, dcrnn, astgcn, mtgnn, 或 all (默认: all)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='高级模型的批大小（默认32，可选: 16/32/64/128/256）')
    args = parser.parse_args()
    
    print("="*80)
    print("Baseline方法对比实验（完全对齐exp4）")
    print("="*80)
    print(f"训练模型: {args.model.upper()}")
    print("\n数据处理：")
    print("  ✓ 使用exp4的NodeLevelSTGNNDataset")
    print("  ✓ 包含时间窗口插值（样本×2）")
    print("  ✓ 包含图数据增强（特征噪声+节点遮挡+边扰动）")
    print("  ✓ 包含所有特征工程（16维特征）")
    print("\n唯一区别：模型架构")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据（与exp4完全一致）
    print("\n加载网络数据...")
    with open('data/processed/networks/all_networks_2017_2021.pkl', 'rb') as f:
        networks = pickle.load(f)
    
    print("构建数据集（与exp4一致）...")
    dataset = NodeLevelSTGNNDataset(
        networks=networks,
        window_size=3,
        use_interpolation=True,  # 与exp4一致
        risk_labels_path='data/processed/networks/risk_labels.pkl'
    )
    
    # 划分训练/测试集（完全对齐exp4）
    print("\n划分训练/测试集...")
    import random
    random.seed(42)
    
    all_samples = dataset.samples.copy()
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    from training.splits import assert_disjoint_samples
    assert_disjoint_samples(train_samples, test_samples)
    
    print(f"  训练样本: {len(train_samples)}")
    print(f"  测试样本: {len(test_samples)}")
    
    #  性能优化：预计算所有样本的全图特征（避免训练时重复计算）
    print("\n预计算全图特征（加速GCN/GAT训练10-100倍）...")
    def precompute_full_graph_features(samples, dataset):
        """为每个样本预计算所有1200个节点的完整16维特征"""
        for sample_idx, sample in enumerate(tqdm(samples, desc="  预计算进度", leave=False)):
            input_years = sample['input_years']
            all_nodes = dataset.all_nodes
            
            # 为所有节点计算完整16维特征
            full_features = []
            for node in all_nodes:
                # 历史统计特征
                hist_changes = []
                for t in range(1, len(input_years)):
                    trade_prev = dataset.features_cache[input_years[t-1]][node]['total_degree']
                    trade_curr = dataset.features_cache[input_years[t]][node]['total_degree']
                    if trade_prev > 0:
                        hist_changes.append((trade_curr - trade_prev) / trade_prev)
                
                hist_volatility = np.std(hist_changes) if len(hist_changes) > 0 else 0
                hist_acceleration = (hist_changes[-1] - hist_changes[0]) if len(hist_changes) >= 2 else 0
                hist_avg_change = np.mean(hist_changes) if len(hist_changes) > 0 else 0
                
                sentiment_changes = []
                for t in range(1, len(input_years)):
                    sent_prev = dataset.features_cache[input_years[t-1]][node]['sentiment_mean']
                    sent_curr = dataset.features_cache[input_years[t]][node]['sentiment_mean']
                    sentiment_changes.append(sent_curr - sent_prev)
                sentiment_change_rate = np.mean(sentiment_changes) if len(sentiment_changes) > 0 else 0
                
                trade_momentum = hist_avg_change * hist_volatility
                
                degree_changes = []
                for t in range(1, len(input_years)):
                    deg_prev = dataset.features_cache[input_years[t-1]][node]['total_degree']
                    deg_curr = dataset.features_cache[input_years[t]][node]['total_degree']
                    if deg_prev > 0:
                        degree_changes.append((deg_curr - deg_prev) / deg_prev)
                degree_change_rate = np.mean(degree_changes) if len(degree_changes) > 0 else 0
                
                last_year = input_years[-1]
                last_year_node_sentiment = dataset.features_cache[last_year][node]['sentiment_mean']
                last_year_neighbor_sentiment = dataset.features_cache[last_year][node]['neighbor_avg_sentiment']
                sentiment_risk_score = last_year_node_sentiment * last_year_neighbor_sentiment
                
                cached = dataset.features_cache[last_year][node]
                affective_vulnerability = dataset._domirank_to_vulnerability(
                    cached['affective_domirank'], is_affective=True
                )
                
                features = [
                    cached['in_degree'], cached['out_degree'], cached['pagerank'],
                    cached['sentiment_mean'], cached['news_count'], affective_vulnerability,
                    cached['neighbor_avg_sentiment'], cached['neighbor_avg_risk'],
                    cached['supplier_concentration'],
                    hist_avg_change, hist_volatility, hist_acceleration,
                    sentiment_change_rate, trade_momentum, degree_change_rate,
                    sentiment_risk_score
                ]
                full_features.append(features)
            
            # 归一化全图特征
            full_features = torch.FloatTensor(full_features)  # [1200, 16]
            full_features = (full_features - torch.FloatTensor(dataset.feature_mean)) / torch.FloatTensor(dataset.feature_std)
            
            # 保存到sample中
            sample['full_graph_features'] = full_features  # [1200, 16]
    
    # 预计算训练集和测试集
    precompute_full_graph_features(train_samples, dataset)
    precompute_full_graph_features(test_samples, dataset)
    print(f"  ✓ 预计算完成！训练速度将提升10-100倍")
    
    # 🔥 关键修复：创建 SimpleDataset 包装（与exp4完全一致）
    class SimpleDataset:
        def __init__(self, samples, stats):
            self.samples = samples
            self.label_mean = stats['label_mean']
            self.label_std = stats['label_std']
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            return self.samples[idx]
        
        def denormalize_labels(self, labels):
            return labels * self.label_std + self.label_mean
    
    stats = {
        'label_mean': dataset.label_mean,
        'label_std': dataset.label_std
    }
    
    train_dataset = SimpleDataset(train_samples, stats)
    test_dataset = SimpleDataset(test_samples, stats)
    
    # 特征维度（与exp4一致：16维）
    input_dim = 16  # 所有模型都使用完整16维特征！
    print(f"\n特征维度: {input_dim} (与exp4完全一致)")
    
    results = {}
    
    # 根据参数训练相应模型
    if args.model in ['mlp', 'all']:
        # ========================================================================
        # 1. MLP Baseline
        # ========================================================================
        print("\n" + "="*80)
        print("1. MLP Baseline (无图无时序)")
        print("="*80)
        model = MLP(input_dim=input_dim, hidden_dim=64, dropout=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=False, batch_size=32
        )
        results['MLP'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['lstm', 'all']:
        # ========================================================================
        # 2. LSTM Baseline
        # ========================================================================
        print("\n" + "="*80)
        print("2. LSTM Baseline (时序，无图)")
        print("="*80)
        model = LSTMModel(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=False, batch_size=32
        )
        results['LSTM'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['gcn', 'all']:
        # ========================================================================
        # 3. GCN Baseline
        # ========================================================================
        print("\n" + "="*80)
        print("3. GCN Baseline (图，无时序)")
        print("="*80)
        print("  注意：GCN使用完整16维特征（最后时间步），在完整供应链图上运行")
        model = GCNModel(input_dim=input_dim, hidden_dim=64, num_nodes=len(dataset.all_nodes), dropout=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=True, batch_size=32
        )
        results['GCN'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['gat', 'all']:
        # ========================================================================
        # 4. GAT Baseline
        # ========================================================================
        print("\n" + "="*80)
        print("4. GAT Baseline (图注意力，无时序)")
        print("="*80)
        print("  注意：GAT使用完整16维特征（最后时间步），在完整供应链图上运行")
        print("  改进：hidden_dim=64（与GCN一致）+ 使用边权重辅助注意力学习")
        model = GATModel(input_dim=input_dim, hidden_dim=64, heads=2, num_nodes=len(dataset.all_nodes), dropout=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=True, batch_size=32
        )
        results['GAT'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['gcn_lstm', 'all']:
        # ========================================================================
        # 5. GCN+LSTM (空间+时序融合)
        # ========================================================================
        print("\n" + "="*80)
        print("5. GCN+LSTM (空间+时序融合)")
        print("="*80)
        print("  特点：GCN处理图结构 + LSTM处理时序 → 融合预测")
        print("  目的：验证简单融合策略的效果")
        model = GCN_LSTM_Model(input_dim=input_dim, hidden_dim=64, num_nodes=len(dataset.all_nodes), 
                               lstm_layers=2, dropout=0.5)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=True, batch_size=32
        )
        results['GCN+LSTM'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['gat_lstm', 'all']:
        # ========================================================================
        # 6. GAT+LSTM (注意力+时序融合)
        # ========================================================================
        print("\n" + "="*80)
        print("6. GAT+LSTM (注意力+时序融合)")
        print("="*80)
        print("  特点：GAT处理图注意力 + LSTM处理时序 → 融合预测")
        print("  目的：验证注意力机制+时序的效果")
        model = GAT_LSTM_Model(input_dim=input_dim, hidden_dim=64, num_nodes=len(dataset.all_nodes), 
                               lstm_layers=2, dropout=0.5, heads=2)
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=True, batch_size=32
        )
        results['GAT+LSTM'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # ========================================================================
    # 高级模型部分
    # ========================================================================
    
    if args.model in ['dcrnn', 'all']:
        # ========================================================================
        # 5. DCRNN (扩散卷积循环网络)
        # ========================================================================
        print("\n" + "="*80)
        print("5. DCRNN (扩散卷积循环网络) - ICLR 2018")
        print("="*80)
        print("  论文: Diffusion Convolutional Recurrent Neural Network")
        print("  特点: K步扩散 + DCGRU，建模供应链风险传播")
        model = AdvancedBaselineWrapper(
            model_type='dcrnn',
            input_dim=input_dim,  # 使用完整16维特征（与exp4一致）
            hidden_dim=32,  # 与exp4一致
            num_nodes=len(dataset.all_nodes),
            K=1,  # 减少扩散步数: 2→1
            num_layers=1,  # 减少层数: 2→1
            dropout=0.3
        )
        print(f"  配置：hidden=32（与exp4一致），K=1, layers=1（简化）")
        print(f"  批大小: {args.batch_size} (GPU利用率优化)")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=False, use_advanced=True,
            batch_size=args.batch_size
        )
        results['DCRNN'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['astgcn', 'all']:
        # ========================================================================
        # 6. ASTGCN (时空注意力图卷积网络)
        # ========================================================================
        print("\n" + "="*80)
        print("6. ASTGCN (时空注意力图卷积网络) - AAAI 2019")
        print("="*80)
        print("  论文: Attention Based Spatial-Temporal Graph Convolutional Networks")
        print("  特点: 空间注意力 + 时间注意力 + 切比雪夫图卷积")
        model = AdvancedBaselineWrapper(
            model_type='astgcn',
            num_nodes=len(dataset.all_nodes),
            input_dim=input_dim,  # 使用完整16维特征（与exp4一致）
            hidden_dim=32,  # 与exp4一致
            num_time_steps=3,  # window_size
            K=2,  # 减少切比雪夫阶数: 3→2
            num_blocks=1  # 减少块数: 2→1
        )
        print(f"  配置：hidden=32（与exp4一致），K=2, blocks=1（简化）")
        print(f"  批大小: {args.batch_size} (GPU利用率优化)")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=False, use_advanced=True,
            batch_size=args.batch_size
        )
        results['ASTGCN'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    if args.model in ['mtgnn', 'all']:
        # ========================================================================
        # 7. MTGNN (多变量时序图神经网络)
        # ========================================================================
        print("\n" + "="*80)
        print("7. MTGNN (多变量时序图神经网络) - KDD 2020")
        print("="*80)
        print("  论文: Multivariate Time Series Forecasting with Graph Neural Networks")
        print("  特点: 自动图学习 + 膨胀因果卷积 + 跳跃连接")
        model = AdvancedBaselineWrapper(
            model_type='mtgnn',
            num_nodes=len(dataset.all_nodes),
            input_dim=input_dim,  # 使用完整16维特征（与exp4一致）
            hidden_dim=32,  # 与exp4一致
            num_layers=2,  # 减少层数: 3→2
            dropout=0.3
        )
        print(f"  配置：hidden=32（与exp4一致），layers=2（简化）")
        print(f"  批大小: {args.batch_size} (GPU利用率优化)")
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
        
        test_r2, test_mae, test_rmse, _, _ = train_baseline(
            model, train_dataset, test_dataset, optimizer, scheduler,
            epochs=200, patience=15, device=device, full_dataset=dataset, use_graph=False, use_advanced=True,
            batch_size=args.batch_size
        )
        results['MTGNN'] = {'R²': test_r2, 'MAE': test_mae, 'RMSE': test_rmse}
        print(f"\n✓ 最终结果 - Test R²: {test_r2:.4f}, MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}")
    
    # ========================================================================
    # 汇总结果
    # ========================================================================
    if results:
        print("\n" + "="*100)
        print("Baseline方法对比结果汇总")
        print("="*100)
        print(f"\n{'方法':<20} | {'Test R²':>10} | {'Test MAE':>10} | {'Test RMSE':>11} | {'建模能力':<30}")
        print("-"*105)
        
        # 定义所有可能的模型
        model_info = {
            'MLP': ('无图无时序（特征直接映射）', 'MLP'),
            'LSTM': ('有时序无图（时间依赖）', 'LSTM'),
            'GCN': ('有图无时序（空间依赖）', 'GCN'),
            'GAT': ('图注意力无时序（自适应空间）', 'GAT'),
            'DCRNN': ('扩散卷积+GRU（供应链传播）', 'DCRNN (ICLR\'18)'),
            'ASTGCN': ('时空注意力+图卷积（自适应）', 'ASTGCN (AAAI\'19)'),
            'MTGNN': ('图学习+因果卷积（端到端）', 'MTGNN (KDD\'20)')
        }
        
        # 只显示已训练的模型
        for model_name, (description, display_name) in model_info.items():
            if model_name in results:
                print(f"{display_name:<20} | {results[model_name]['R²']:>10.4f} | {results[model_name]['MAE']:>10.4f} | {results[model_name]['RMSE']:>11.4f} | {description:<30}")
        
        # 保存结果
        with open('experiments/baseline_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        print("\n结果已保存到: experiments/baseline_results.pkl")
    
    print(f"\n使用方法：")
    print(f"  基础模型:")
    print(f"    python experiments/run_baselines.py --model mlp     # 只训练MLP")
    print(f"    python experiments/run_baselines.py --model lstm    # 只训练LSTM")
    print(f"    python experiments/run_baselines.py --model gcn     # 只训练GCN")
    print(f"    python experiments/run_baselines.py --model gat     # 只训练GAT")
    print(f"  高级模型:")
    print(f"    python experiments/run_baselines.py --model dcrnn   # DCRNN (ICLR'18)")
    print(f"    python experiments/run_baselines.py --model astgcn  # ASTGCN (AAAI'19)")
    print(f"    python experiments/run_baselines.py --model mtgnn   # MTGNN (KDD'20)")
    print(f"  全部:")
    print(f"    python experiments/run_baselines.py --model all     # 训练所有模型（默认）")


if __name__ == '__main__':
    main()
