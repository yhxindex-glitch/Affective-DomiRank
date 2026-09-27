"""
供应链数据集类
将数据转换为PyTorch Dataset格式
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Tuple


class SupplyChainDataset(Dataset):
    """供应链时序数据集"""
    
    def __init__(
        self,
        networks_file: str,
        dataset_file: str,
        seq_length: int = 3,
        prediction_horizon: int = 1,
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """
        初始化数据集
        
        参数:
            networks_file: 网络pickle文件路径
            dataset_file: 表格数据CSV文件路径
            seq_length: 输入序列长度
            prediction_horizon: 预测时间步长
            split: 'train', 'val', 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
        """
        self.seq_length = seq_length
        self.prediction_horizon = prediction_horizon
        
        # 加载网络数据
        with open(networks_file, 'rb') as f:
            self.networks = pickle.load(f)
        
        # 加载表格数据
        self.df = pd.read_csv(dataset_file)
        
        # 获取所有时间周期
        self.periods = sorted(self.networks.keys())
        
        # 构建时序样本
        self.samples = self._build_samples()
        
        # 划分数据集
        self.samples = self._split_data(split, train_ratio, val_ratio)
        
        print(f"  {split} 数据集: {len(self.samples)} 个样本")
    
    def _build_samples(self) -> List[Dict]:
        """构建时序样本"""
        samples = []
        
        # 对于每个可能的时间窗口
        for i in range(len(self.periods) - self.seq_length - self.prediction_horizon + 1):
            # 输入序列：periods[i:i+seq_length]
            # 预测目标：periods[i+seq_length:i+seq_length+prediction_horizon]
            
            input_periods = self.periods[i:i+self.seq_length]
            target_period = self.periods[i+self.seq_length]
            
            sample = {
                'input_periods': input_periods,
                'target_period': target_period,
            }
            
            samples.append(sample)
        
        return samples
    
    def _split_data(
        self,
        split: str,
        train_ratio: float,
        val_ratio: float
    ) -> List[Dict]:
        """划分数据集"""
        n_samples = len(self.samples)
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        
        if split == 'train':
            return self.samples[:n_train]
        elif split == 'val':
            return self.samples[n_train:n_train+n_val]
        elif split == 'test':
            return self.samples[n_train+n_val:]
        else:
            return self.samples
    
    def __len__(self) -> int:
        """数据集大小"""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        参数:
            idx: 样本索引
            
        返回:
            (node_features, edge_index, edge_weight, target)
        """
        sample = self.samples[idx]
        input_periods = sample['input_periods']
        target_period = sample['target_period']
        
        # 构建节点特征序列
        node_features_seq = []
        
        for period in input_periods:
            G = self.networks[period]
            
            # 提取节点特征
            node_features = []
            nodes = sorted(G.nodes())
            
            for node in nodes:
                attrs = G.nodes[node]
                
                # 特征：[网络特征4维 + 情感特征7维 + DomiRank等4维]
                features = [
                    attrs.get('in_degree', 0.0),
                    attrs.get('out_degree', 0.0),
                    attrs.get('total_degree', 0.0),
                    attrs.get('pagerank', 0.0),
                    attrs.get('sentiment_polarity', 0.0),
                    attrs.get('sentiment_intensity', 0.0),
                    attrs.get('fear', 0.0),
                    attrs.get('trust', 0.0),
                    attrs.get('anger', 0.0),
                    attrs.get('joy', 0.0),
                    attrs.get('sadness', 0.0),
                ]
                
                node_features.append(features)
            
            node_features_seq.append(node_features)
        
        # 转换为tensor
        node_features = torch.FloatTensor(node_features_seq)  # [seq_len, num_nodes, feature_dim]
        node_features = node_features.permute(1, 0, 2)  # [num_nodes, seq_len, feature_dim]
        
        # 构建边索引（使用最后一个时间步的网络结构）
        G_last = self.networks[input_periods[-1]]
        edge_list = list(G_last.edges())
        nodes = sorted(G_last.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        edge_index = []
        edge_weight = []
        
        for u, v in edge_list:
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            weight = G_last[u][v].get('normalized_weight', 1.0)
            edge_weight.append(weight)
        
        if edge_index:
            edge_index = torch.LongTensor(edge_index).t()  # [2, num_edges]
            edge_weight = torch.FloatTensor(edge_weight)
        else:
            edge_index = torch.LongTensor([[],[]])
            edge_weight = torch.FloatTensor([])
        
        # 构建目标（简化：预测网络是否会中断）
        # 这里用网络规模变化作为代理指标
        G_target = self.networks[target_period]
        G_prev = self.networks[input_periods[-1]]
        
        # 目标：网络规模是否下降（二分类）
        size_change = (G_target.number_of_nodes() - G_prev.number_of_nodes()) / G_prev.number_of_nodes()
        
        # 如果节点数下降超过5%，标记为中断
        target = 1.0 if size_change < -0.05 else 0.0
        target = torch.FloatTensor([target])
        
        return node_features, edge_index, edge_weight, target


if __name__ == "__main__":
    # 测试数据集
    print("=" * 80)
    print("测试数据集加载")
    print("=" * 80)
    
    networks_file = "data/processed/final/supply_chain_network_with_emotions.pkl"
    dataset_file = "data/processed/final/final_dataset.csv"
    
    # 创建数据集
    dataset = SupplyChainDataset(
        networks_file=networks_file,
        dataset_file=dataset_file,
        seq_length=2,  # 用2个月预测
        prediction_horizon=1,
        split='train'
    )
    
    print(f"\n数据集样本数: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试加载
        node_features, edge_index, edge_weight, target = dataset[0]
        
        print(f"\n样本形状:")
        print(f"  节点特征: {node_features.shape}")
        print(f"  边索引: {edge_index.shape}")
        print(f"  边权重: {edge_weight.shape}")
        print(f"  目标: {target.shape}, 值={target.item()}")
        
        print("\n✓ 数据集测试完成!")
    else:
        print("\n警告: 数据集为空（时间步不够）")

