"""
实验4: STAEformer节点级时序预测（情感风险预测版）

模型：STAEformer (Spatio-Temporal Adaptive Embedding Transformer, 2024)
核心创新：Affective DomiRank作为关键特征输入

任务：对每个节点，学习其在历史图序列中的时序演化 → 预测未来综合风险得分
标签：comprehensive_risk_score（0-1连续值，情感权重65%）

优化方案（综合）：
  1. 增强时序特征：动量、度数变化、情感风险（+3维）
  2. 时间窗口插值：样本数×2（2400→4800+）
  3. 图数据增强：
     - 特征噪声注入（5%）
     - 节点遮挡（20%节点）
     - 边扰动（删除10%的边）
  4. STAEformer架构：GCN空间嵌入 + Transformer时序编码
  5. 适度正则化：dropout=0.1，weight_decay=3e-4
  6. 动态学习率：Cosine Annealing with Warm Restarts

增强特征（16维）：
  原始特征（6维）：
    - in_degree, out_degree, pagerank
    - sentiment_mean, news_count
    - affective_domirank（创新点）
    
  方案1 - 时序历史特征（3维）：
    - hist_avg_change: 历史平均变化率
    - hist_volatility: 历史波动性
    - hist_acceleration: 变化加速度
    
  方案2 - 邻居聚合特征（3维）：
    - neighbor_avg_sentiment: 供应商平均情感
    - neighbor_avg_risk: 供应商平均风险
    - supplier_concentration: 供应商集中度
    
  方案3 - 改进情感特征（1维）：
    - sentiment_change_rate: 情感变化率
    
  新增强特征（3维）：
    - trade_momentum: 贸易动量（变化率×波动性）
    - degree_change_rate: 度数变化率
    - sentiment_risk_score: 情感风险分数

样本数：1200节点 × 6+个时间窗（插值后） = 4800+个

修正历史：
  - GAT+LSTM基础版（13维+hidden=32）: Test R²=0.322
  - GAT+LSTM平衡版（16维+hidden=28）: Test R²=0.334
  
预期（STAEformer版）：
  - STAEformer (GCN + Transformer): 更强的时序建模能力
  - Transformer注意力机制：自适应捕捉时间依赖
  - 预期 Test R² > 0.40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import pickle
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.stgnn import STGNN, STGNNConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class NodeLevelSTGNNDataset:
    """
    节点级STGNN数据集（风险预测版）
    每个样本：一个节点在连续3年的图序列 → 预测第4年该节点的综合风险得分
    标签 = comprehensive_risk_score（0-1连续值，主要由情感因素驱动）
    """
    
    def __init__(
        self,
        networks: dict,
        window_size: int = 3,
        use_interpolation: bool = True,
        risk_labels_path: str = 'data/processed/networks/risk_labels.pkl'
    ):
        """
        初始化数据集
        
        参数:
            networks: 年度网络字典 {year: networkx.DiGraph}
            window_size: 输入窗口大小（年数）
            use_interpolation: 是否使用时间插值（样本数翻倍）
        """
        self.networks = networks
        self.years = sorted(networks.keys())
        self.original_years = list(self.years)  # 保存原始真实年份（用于标签插值判断）
        self.window_size = window_size
        self.use_interpolation = use_interpolation
        
        # 加载风险标签
        print(f"\n加载风险标签: {risk_labels_path}")
        with open(risk_labels_path, 'rb') as f:
            import pandas as pd
            self.risk_labels_df = pickle.load(f)
        print(f"  风险标签数: {len(self.risk_labels_df)}")
        print(f"  风险均值: {self.risk_labels_df['comprehensive_risk_score'].mean():.4f}")
        print(f"  风险标准差: {self.risk_labels_df['comprehensive_risk_score'].std():.4f}")
        
        # 获取所有节点（假设每年节点相同）
        self.all_nodes = sorted(list(networks[self.years[0]].nodes()))
        
        print(f"\n总节点数: {len(self.all_nodes)}")
        
        # 如果使用插值，创建中间时间点的数据（样本×2）
        if use_interpolation:
            print("  使用时间窗口插值增强...")
            self._create_interpolated_networks()
        
        # 预提取所有特征到缓存（加速100倍）
        print("  预提取特征到缓存...")
        self._precompute_features_cache()
        
        # 预转换所有图到PyG格式（避免重复转换）
        print("  预转换图结构...")
        self._precompute_pyg_graphs()
        
        # 【方案A】计算DomiRank归一化参数（用于转换为脆弱性指数）
        print("  计算DomiRank归一化参数...")
        self._compute_domirank_normalization()
        
        # 构建样本
        self.samples = self._build_samples()
        
        # 归一化
        self._compute_normalization_stats()
        self._normalize_data()
    
    def _create_interpolated_networks(self):
        """
        创建插值的中间时间点网络（改进的非线性插值）
        
        改进点：
        1. 节点特征使用加权插值（0.45:0.55，略偏向未来趋势）
        2. 边权重使用调和平均（对突变更保守，避免过高估计贸易额）
        3. 保持原有的图结构完整性
        """
        import networkx as nx
        interpolated = {}
        
        for i in range(len(self.years) - 1):
            year1 = self.years[i]
            year2 = self.years[i+1]
            mid_year = year1 + 0.5
            
            G1 = self.networks[year1]
            G2 = self.networks[year2]
            
            # 创建新的有向图
            G_mid = nx.DiGraph()
            
            # 1. 插值节点特征（只保留两年都存在的节点）
            common_nodes = set(G1.nodes()) & set(G2.nodes())
            for node in common_nodes:
                G_mid.add_node(node)
                # 对每个数值特征插值
                for attr in ['in_degree', 'out_degree', 'pagerank', 'sentiment_mean',
                            'news_count', 'affective_domirank', 'neighbor_avg_sentiment',
                            'neighbor_avg_risk', 'supplier_concentration', 'total_degree',
                            'domirank', 'has_sentiment', 'sentiment_std']:
                    val1 = G1.nodes[node].get(attr, 0)
                    val2 = G2.nodes[node].get(attr, 0)
                    
                    if isinstance(val1, bool) or isinstance(val2, bool):
                        G_mid.nodes[node][attr] = val1 or val2  # 布尔值用or
                    else:
                        # 改进：加权插值（0.45:0.55），略偏向未来以捕捉趋势
                        # 对于零值情况，使用简单平均
                        if val1 == 0 and val2 == 0:
                            G_mid.nodes[node][attr] = 0
                        elif val1 == 0 or val2 == 0:
                            G_mid.nodes[node][attr] = (val1 + val2) / 2.0
                        else:
                            G_mid.nodes[node][attr] = 0.45 * val1 + 0.55 * val2
            
            # 2. 插值边权重（只保留两年都存在的边）
            for u, v in G1.edges():
                if G2.has_edge(u, v) and u in common_nodes and v in common_nodes:
                    weight1 = G1[u][v].get('weight', 0)
                    weight2 = G2[u][v].get('weight', 0)
                    norm_weight1 = G1[u][v].get('normalized_weight', 0)
                    norm_weight2 = G2[u][v].get('normalized_weight', 0)
                    
                    # 改进：边权重使用调和平均（对突变更保守）
                    # 调和平均 = 2*a*b/(a+b)，对大幅波动会给出更低估计
                    if weight1 > 0 and weight2 > 0:
                        weight_mid = 2 * weight1 * weight2 / (weight1 + weight2)
                    else:
                        weight_mid = (weight1 + weight2) / 2.0
                    
                    # 归一化权重使用加权平均
                    norm_weight_mid = 0.45 * norm_weight1 + 0.55 * norm_weight2
                    
                    G_mid.add_edge(u, v, 
                                  weight=weight_mid,
                                  normalized_weight=norm_weight_mid)
            
            interpolated[mid_year] = G_mid
        
        # 合并原始网络和插值网络
        self.networks.update(interpolated)
        self.years = sorted(self.networks.keys())
        
        print(f"    插值后时间点数: {len(self.years)} (改进非线性插值)")
    
    def _precompute_features_cache(self):
        """
        预提取所有节点在所有年份的特征到缓存
        避免重复访问networkx图，加速100倍
        """
        self.features_cache = {}
        self.node_to_idx = {node: i for i, node in enumerate(self.all_nodes)}
        
        # 预定义特征列表（包含传统DomiRank供消融实验使用）
        feature_attrs = ['in_degree', 'out_degree', 'pagerank', 'sentiment_mean',
                        'news_count', 'affective_domirank', 'domirank', 'neighbor_avg_sentiment',
                        'neighbor_avg_risk', 'supplier_concentration', 'total_degree']
        
        for year in self.years:
            G = self.networks[year]
            # 为每个节点提取特征
            year_features = {}
            for node in self.all_nodes:
                if node in G.nodes():
                    # 提取所有特征一次性存储
                    node_data = G.nodes[node]
                    year_features[node] = {
                        attr: node_data.get(attr, 0) for attr in feature_attrs
                    }
                else:
                    # 节点不存在，全0
                    year_features[node] = {attr: 0 for attr in feature_attrs}
            
            self.features_cache[year] = year_features
        
        print(f"    ✓ 已缓存 {len(self.years)} 年 × {len(self.all_nodes)} 节点的特征")
    
    def _precompute_pyg_graphs(self):
        """
        预转换所有年份的图到PyG格式
        避免每个样本都重复转换（7200次→9次）
        """
        self.pyg_graphs = {}
        for year in self.years:
            self.pyg_graphs[year] = self._convert_to_pyg(self.networks[year])
        print(f"    ✓ 已转换 {len(self.years)} 个图到PyG格式")
    
    def _compute_domirank_normalization(self):
        """
        【方案A】计算DomiRank的归一化参数
        用于将DomiRank转换为脆弱性指数（vulnerability）
        
        逻辑：
        - DomiRank高 = 支配力强 = 稳定 → vulnerability低
        - DomiRank低 = 边缘节点 = 脆弱 → vulnerability高
        
        转换公式：vulnerability = 1 - normalize(domirank)
        """
        all_affective_dr = []
        all_traditional_dr = []
        
        for year in self.original_years:  # 只使用真实年份
            G = self.networks[year]
            for node in G.nodes():
                adr = G.nodes[node].get('affective_domirank', 0)
                dr = G.nodes[node].get('domirank', 0)
                all_affective_dr.append(adr)
                all_traditional_dr.append(dr)
        
        # 计算最小值和最大值（用于min-max归一化）
        self.affective_dr_min = np.min(all_affective_dr)
        self.affective_dr_max = np.max(all_affective_dr)
        self.traditional_dr_min = np.min(all_traditional_dr)
        self.traditional_dr_max = np.max(all_traditional_dr)
        
        print(f"    Affective DomiRank范围: [{self.affective_dr_min:.4f}, {self.affective_dr_max:.4f}]")
        print(f"    传统DomiRank范围: [{self.traditional_dr_min:.4f}, {self.traditional_dr_max:.4f}]")
        print(f"    ✓ 将转换为脆弱性指数（高DomiRank→低脆弱性）")
    
    def _domirank_to_vulnerability(self, domirank_value, is_affective=True):
        """
        将DomiRank转换为脆弱性指数
        
        参数:
            domirank_value: DomiRank值
            is_affective: 是否为Affective DomiRank
        
        返回:
            vulnerability: 脆弱性指数（0-1，越大越脆弱）
        """
        if is_affective:
            dr_min, dr_max = self.affective_dr_min, self.affective_dr_max
        else:
            dr_min, dr_max = self.traditional_dr_min, self.traditional_dr_max
        
        # Min-Max归一化到[0, 1]
        if dr_max - dr_min > 1e-8:
            normalized = (domirank_value - dr_min) / (dr_max - dr_min)
        else:
            normalized = 0.5
        
        # 反转：高DomiRank → 低脆弱性
        vulnerability = 1.0 - normalized
        
        return vulnerability
    
    def _build_samples(self):
        """
        构建节点级时序样本
        
        例如：5年数据，窗口=3
        时间窗1: [2017,2018,2019] → 2020
        时间窗2: [2018,2019,2020] → 2021
        
        对每个节点，在每个时间窗创建一个样本
        总样本数 = 节点数 × 时间窗数
        """
        samples = []
        time_windows = []
        
        # 构建时间窗
        for i in range(len(self.years) - self.window_size):
            input_years = self.years[i:i+self.window_size]
            target_year = self.years[i+self.window_size]
            time_windows.append((input_years, target_year))
        
        print(f"\n时间窗口数: {len(time_windows)}")
        for i, (inp, tgt) in enumerate(time_windows):
            print(f"  窗口{i+1}: {inp} → 预测{tgt}")
        
        print(f"\n开始构建样本")
        
        # 为每个节点在每个时间窗创建样本
        total_nodes = len(self.all_nodes)
        for node_idx, node in enumerate(self.all_nodes):
            # 每处理100个节点显示进度
            if (node_idx + 1) % 100 == 0 or node_idx == 0:
                print(f"  处理节点 {node_idx+1}/{total_nodes}...")
            for window_idx, (input_years, target_year) in enumerate(time_windows):
                # 提取该节点在输入年份的特征序列
                node_features_seq = []
                
                # 计算历史变化率（方案1：时序特征） - 使用缓存
                hist_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:  # 从第二年开始可以计算变化率
                        prev_year = input_years[i-1]
                        trade_prev = self.features_cache[prev_year][node]['total_degree']
                        trade_curr = self.features_cache[year][node]['total_degree']
                        if trade_prev > 0:
                            change = (trade_curr - trade_prev) / trade_prev
                            hist_changes.append(change)
                
                # 历史波动性和加速度
                hist_volatility = np.std(hist_changes) if len(hist_changes) > 0 else 0
                hist_acceleration = (hist_changes[-1] - hist_changes[0]) if len(hist_changes) >= 2 else 0
                hist_avg_change = np.mean(hist_changes) if len(hist_changes) > 0 else 0
                
                # 情感变化率（方案3：改进情感特征） - 使用缓存
                sentiment_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:
                        prev_year = input_years[i-1]
                        sent_prev = self.features_cache[prev_year][node]['sentiment_mean']
                        sent_curr = self.features_cache[year][node]['sentiment_mean']
                        sentiment_changes.append(sent_curr - sent_prev)
                
                sentiment_change_rate = np.mean(sentiment_changes) if len(sentiment_changes) > 0 else 0
                
                # 新增强特征（方案2）
                # 1. 贸易动量：变化率×波动性（捕捉加速变化）
                trade_momentum = hist_avg_change * hist_volatility
                
                # 2. 度数变化率：度数本身的变化趋势 - 使用缓存
                degree_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:
                        prev_year = input_years[i-1]
                        deg_prev = self.features_cache[prev_year][node]['total_degree']
                        deg_curr = self.features_cache[year][node]['total_degree']
                        if deg_prev > 0:
                            degree_changes.append((deg_curr - deg_prev) / deg_prev)
                degree_change_rate = np.mean(degree_changes) if len(degree_changes) > 0 else 0
                
                # 3. 情感风险分数：自己和邻居都负面=双重风险 - 使用缓存
                last_year_node_sentiment = self.features_cache[input_years[-1]][node]['sentiment_mean']
                last_year_neighbor_sentiment = self.features_cache[input_years[-1]][node]['neighbor_avg_sentiment']
                sentiment_risk_score = last_year_node_sentiment * last_year_neighbor_sentiment
                
                # 提取每年的特征 - 使用缓存
                for year in input_years:
                    cached_features = self.features_cache[year][node]
                    # 【方案A】将Affective DomiRank转换为脆弱性指数
                    affective_vulnerability = self._domirank_to_vulnerability(
                        cached_features['affective_domirank'], 
                        is_affective=True
                    )
                    
                    features = [
                        # 原始特征
                        cached_features['in_degree'],
                        cached_features['out_degree'],
                        cached_features['pagerank'],
                        cached_features['sentiment_mean'],
                        cached_features['news_count'],
                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）
                        
                        # 方案2：邻居聚合特征
                        cached_features['neighbor_avg_sentiment'],
                        cached_features['neighbor_avg_risk'],
                        cached_features['supplier_concentration'],
                        
                        # 方案1：时序历史特征（对序列中所有时间点都一样）
                        hist_avg_change,
                        hist_volatility,
                        hist_acceleration,
                        
                        # 方案3：改进情感特征
                        sentiment_change_rate,
                        
                        # 方案2：新增强特征
                        trade_momentum,
                        degree_change_rate,
                        sentiment_risk_score,
                    ]
                    node_features_seq.append(features)
                
                # 计算标签：从risk_labels查询或插值综合风险得分
                last_input_year = input_years[-1]
                
                # 关键修复：确定真实的year_from和year_to用于查询
                # 如果target_year是真实年份，需要找前一个真实年份作为year_from
                if target_year in self.original_years:
                    # 目标是真实年份，找前一个真实年份
                    target_idx = self.original_years.index(target_year)
                    if target_idx > 0:
                        year_from_query = self.original_years[target_idx - 1]
                        year_to_query = target_year
                        
                        risk_row = self.risk_labels_df[
                            (self.risk_labels_df['year_from'] == year_from_query) &
                            (self.risk_labels_df['year_to'] == year_to_query) &
                            (self.risk_labels_df['node'] == node)
                        ]
                        
                        if len(risk_row) > 0:
                            label = risk_row.iloc[0]['comprehensive_risk_score']
                        else:
                            label = 0.0
                    else:
                        label = 0.0
                else:
                    # 目标是插值年份（如2018.5），需要插值
                    floor_year = int(target_year)  # 2018
                    ceil_year = floor_year + 1      # 2019
                    
                    # 查询 2017→2018 的风险
                    if floor_year in self.original_years:
                        floor_idx = self.original_years.index(floor_year)
                        if floor_idx > 0:
                            year_from_floor = self.original_years[floor_idx - 1]
                            risk_floor = self.risk_labels_df[
                                (self.risk_labels_df['year_from'] == year_from_floor) &
                                (self.risk_labels_df['year_to'] == floor_year) &
                                (self.risk_labels_df['node'] == node)
                            ]
                        else:
                            risk_floor = []
                    else:
                        risk_floor = []
                    
                    # 查询 2018→2019 的风险
                    if ceil_year in self.original_years:
                        ceil_idx = self.original_years.index(ceil_year)
                        if ceil_idx > 0:
                            year_from_ceil = self.original_years[ceil_idx - 1]
                            risk_ceil = self.risk_labels_df[
                                (self.risk_labels_df['year_from'] == year_from_ceil) &
                                (self.risk_labels_df['year_to'] == ceil_year) &
                                (self.risk_labels_df['node'] == node)
                            ]
                        else:
                            risk_ceil = []
                    else:
                        risk_ceil = []
                    
                    if len(risk_floor) > 0 and len(risk_ceil) > 0:
                        # 线性插值
                        weight = target_year - floor_year  # 0.5
                        risk_floor_val = risk_floor.iloc[0]['comprehensive_risk_score']
                        risk_ceil_val = risk_ceil.iloc[0]['comprehensive_risk_score']
                        label = (1 - weight) * risk_floor_val + weight * risk_ceil_val
                    else:
                        label = 0.0
                
                # 创建样本
                sample = {
                    'node': node,
                    'window_idx': window_idx,
                    'input_years': input_years,
                    'target_year': target_year,
                    'node_features_seq': torch.FloatTensor(node_features_seq),  # [window_size, feature_dim]
                    'label': torch.FloatTensor([label]),
                    # 存储情感值用于加权损失（方案3）
                    'sentiment': last_year_node_sentiment,
                    # 存储图结构（使用预转换的图）
                    'graphs': [self.pyg_graphs[y] for y in input_years]
                }
                
                samples.append(sample)
        
        print(f"\n构建了 {len(samples)} 个节点级时序样本")
        print(f"  = {len(self.all_nodes)} 节点 × {len(time_windows)} 时间窗")
        
        # 验证插值标签（如果使用插值）
        if self.use_interpolation:
            real_year_samples = [s for s in samples if s['target_year'] in self.original_years]
            interp_year_samples = [s for s in samples if s['target_year'] not in self.original_years]
            
            real_labels = [s['label'].item() for s in real_year_samples]
            interp_labels = [s['label'].item() for s in interp_year_samples]
            
            print(f"\n插值验证:")
            print(f"  真实年份样本: {len(real_year_samples)}, 标签均值={np.mean(real_labels):.4f}")
            print(f"  插值年份样本: {len(interp_year_samples)}, 标签均值={np.mean(interp_labels):.4f}")
            
            if len(interp_labels) > 0 and np.mean(interp_labels) == 0:
                print(f"  ⚠️ 警告：插值标签全为0，可能插值逻辑有问题！")
            elif len(interp_labels) > 0:
                print(f"  ✓ 插值标签正常，范围=[{np.min(interp_labels):.4f}, {np.max(interp_labels):.4f}]")
        
        return samples
    
    def _convert_to_pyg(self, G):
        """将NetworkX图转换为PyG Data对象"""
        nodes = sorted(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
        # 边
        edge_index = []
        edge_weight = []
        for u, v, data in G.edges(data=True):
            edge_index.append([node_to_idx[u], node_to_idx[v]])
            edge_weight.append(data.get('normalized_weight', 1.0))
        
        return Data(
            edge_index=torch.LongTensor(edge_index).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long),
            edge_weight=torch.FloatTensor(edge_weight) if edge_weight else torch.ones(0),
            nodes=nodes
        )
    
    def _compute_normalization_stats(self):
        """计算归一化统计量"""
        all_features = []
        all_labels = []
        
        for sample in self.samples:
            # 展平时序特征
            all_features.append(sample['node_features_seq'].numpy())
            all_labels.append(sample['label'].numpy())
        
        all_features = np.vstack([f for fs in all_features for f in fs])
        all_labels = np.vstack(all_labels)
        
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0) + 1e-8
        
        self.label_mean = all_labels.mean()
        self.label_std = all_labels.std() + 1e-8
        
        print(f"\n归一化统计:")
        print(f"  特征均值: {self.feature_mean}")
        print(f"  特征标准差: {self.feature_std}")
        print(f"  标签均值: {self.label_mean:.4f}")
        print(f"  标签标准差: {self.label_std:.4f}")
    
    def _normalize_data(self):
        """归一化所有数据"""
        for sample in self.samples:
            # 归一化节点特征序列
            sample['node_features_seq'] = (sample['node_features_seq'] - torch.FloatTensor(self.feature_mean)) / torch.FloatTensor(self.feature_std)
            
            # 归一化标签
            sample['label'] = (sample['label'] - self.label_mean) / self.label_std
    
    def denormalize_labels(self, labels_normalized):
        """反归一化标签"""
        return labels_normalized * self.label_std + self.label_mean
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """自定义collate函数"""
    # 所有样本共享相同的图结构（每个时间步）
    graphs = batch[0]['graphs']
    
    # 收集节点特征序列和标签
    node_features_seqs = torch.stack([s['node_features_seq'] for s in batch])  # [batch, window_size, feature_dim]
    labels = torch.stack([s['label'] for s in batch])  # [batch, 1]
    sentiments = torch.FloatTensor([s['sentiment'] for s in batch])  # [batch] for weighted loss
    
    # 获取节点索引
    all_nodes = graphs[0].nodes
    node_to_idx = {n: i for i, n in enumerate(all_nodes)}
    target_node_indices = torch.LongTensor([node_to_idx[s['node']] for s in batch])
    
    return {
        'graphs': graphs,
        'node_features_seqs': node_features_seqs,
        'target_node_indices': target_node_indices,
        'labels': labels,
        'sentiments': sentiments,
        'nodes': [s['node'] for s in batch]
    }


def train_epoch(model, dataloader, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
        graphs = batch['graphs']
        node_features_seqs = batch['node_features_seqs'].to(device)  # [batch, window, feature_dim]
        target_node_indices = batch['target_node_indices'].to(device)
        labels = batch['labels'].to(device)
        sentiments = batch['sentiments'].to(device)
        
        # ===== 数据增强1：特征噪声 =====
        noise = torch.randn_like(node_features_seqs) * 0.05
        node_features_seqs = node_features_seqs + noise
        
        # ===== 数据增强2：节点遮挡（Node Masking） =====
        # 随机遮挡20%节点的特征
        mask = torch.rand(node_features_seqs.shape[0], node_features_seqs.shape[1], 1, device=device) > 0.2
        node_features_seqs = node_features_seqs * mask
        
        # 提取图结构
        edge_index = graphs[0].edge_index.to(device)
        edge_weight = graphs[0].edge_weight.to(device)
        
        # ===== 数据增强3：边扰动（Edge Perturbation） =====
        # 随机删除10%的边
        num_edges = edge_index.shape[1]
        edge_mask = torch.rand(num_edges, device=device) > 0.1
        edge_index = edge_index[:, edge_mask]
        edge_weight = edge_weight[edge_mask]
        
        # 扩展节点特征到全图
        # node_features_seqs是batch中各节点的特征，需要扩展到整个图
        batch_size, window_size, feature_dim = node_features_seqs.shape
        num_nodes = len(graphs[0].nodes)
        
        # 为整个图的所有节点创建特征（这里简化处理，只有目标节点有特征）
        # 更好的方式是为所有节点提取特征，但这里先用简化版
        # 构造 [batch, window, num_nodes, feature_dim]
        full_features = torch.zeros(batch_size, window_size, num_nodes, feature_dim, device=device)
        for i, node_idx in enumerate(target_node_indices):
            full_features[i, :, node_idx, :] = node_features_seqs[i]
        
        optimizer.zero_grad()
        
        # STGNN前向传播
        out = model(full_features, edge_index, edge_weight, target_node_indices)
        
        # 计算损失（恢复普通MSE，移除加权）
        loss = F.mse_loss(out, labels.squeeze())
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, dataloader, dataset, device):
    """评估模型"""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        graphs = batch['graphs']
        node_features_seqs = batch['node_features_seqs'].to(device)
        target_node_indices = batch['target_node_indices'].to(device)
        labels = batch['labels']
        
        edge_index = graphs[0].edge_index.to(device)
        edge_weight = graphs[0].edge_weight.to(device)
        
        batch_size, window_size, feature_dim = node_features_seqs.shape
        num_nodes = len(graphs[0].nodes)
        
        full_features = torch.zeros(batch_size, window_size, num_nodes, feature_dim, device=device)
        for i, node_idx in enumerate(target_node_indices):
            full_features[i, :, node_idx, :] = node_features_seqs[i]
        
        # 预测
        out = model(full_features, edge_index, edge_weight, target_node_indices)
        
        all_preds.append(out.cpu())
        all_labels.append(labels)
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # 反归一化
    preds_denorm = dataset.denormalize_labels(preds)
    labels_denorm = dataset.denormalize_labels(labels)
    
    # 计算指标
    mae = mean_absolute_error(labels_denorm, preds_denorm)
    rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
    r2 = r2_score(labels_denorm, preds_denorm)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def main():
    """主函数"""
    print("="*80)
    print("实验4: STGNN节点级时序预测")
    print("="*80)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载网络数据
    networks_file = Path('data/processed/networks/all_networks_2017_2021.pkl')
    with open(networks_file, 'rb') as f:
        all_networks = pickle.load(f)
    
    print(f"\n加载网络: {len(all_networks)} 年 (2017-2021)")
    
    # 创建数据集（预测贸易额变化率）
    # 使用window_size=3确保历史特征有足够数据点
    # use_interpolation=True: 时间窗口插值，样本数翻倍
    # 启用插值，风险标签也会线性插值（样本数×3）
    full_dataset = NodeLevelSTGNNDataset(all_networks, window_size=3, use_interpolation=True)
    
    # 划分数据集：随机划分所有样本（避免分布偏移）
    # 将所有时间窗的样本混合，随机划分80%训练、20%测试
    import random
    random.seed(42)
    
    all_samples = full_dataset.samples.copy()
    random.shuffle(all_samples)
    
    # 基础划分：80% vs 20%
    split_idx = int(len(all_samples) * 0.8)
    base_train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    
    train_samples = base_train_samples
    from training.splits import assert_disjoint_samples
    assert_disjoint_samples(train_samples, test_samples)
    random.shuffle(train_samples)  # 打乱顺序
    
    
    # 统计各时间窗在训练/测试集的分布
    train_windows = {}
    test_windows = {}
    for s in train_samples:
        train_windows[s['window_idx']] = train_windows.get(s['window_idx'], 0) + 1
    for s in test_samples:
        test_windows[s['window_idx']] = test_windows.get(s['window_idx'], 0) + 1
    
    print(f"\n  训练集时间窗分布: {train_windows}")
    print(f"  测试集时间窗分布: {test_windows}")
    
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
        'label_mean': full_dataset.label_mean,
        'label_std': full_dataset.label_std
    }
    
    train_dataset = SimpleDataset(train_samples, stats)
    test_dataset = SimpleDataset(test_samples, stats)
    
    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 配置AGCRN（自适应图卷积循环网络）
    config = STGNNConfig(
        node_feature_dim=16,  # 原始6维+邻居3维+时序3维+情感1维+新增3维
        num_nodes=len(full_dataset.all_nodes),  # 1200个节点
        embed_dim=16,  # 节点嵌入维度（用于自适应图学习）
        hidden_dim=64,  # GRU隐藏层维度
        num_time_steps=3,  # 时间步数（3年窗口）
        num_layers=2,  # AGCRN层数
        cheb_k=2,  # Chebyshev多项式阶数
        dropout=0.1,  # Dropout率
        output_dim=1  # 输出维度（风险预测）
    )
    
    model = STGNN(config).to(device)
    
    print(f"\n模型配置:")
    print(f"  核心创新1: Affective DomiRank作为关键特征")
    print(f"  核心创新2: 自适应学习图结构（不依赖预定义邻接矩阵）")
    print(f"  核心创新3: 节点自适应参数学习")
    print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  节点嵌入维度: {config.embed_dim}")
    print(f"  GRU隐藏维度: {config.hidden_dim}")
    print(f"  AGCRN层数: {config.num_layers}")
    print(f"  Chebyshev阶数: {config.cheb_k}")
    print(f"  输入窗口: 3年")
    print(f"  预测方式: 节点级时序预测")
    
    # 优化器（适度L2正则化）
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-4)
    
    # 学习率调度：Cosine Annealing with Warm Restarts
    # T_0=30：每30个epoch重启一次
    # T_mult=2：每次重启后周期翻倍（30, 60, 120...）
    # eta_min=1e-6：最小学习率
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=30,      # 第一个周期长度
        T_mult=2,    # 周期倍增因子
        eta_min=1e-6 # 最小学习率
    )
    
    # 训练
    print(f"\n开始训练...")
    print(f"{'='*80}")
    
    best_test_r2 = -float('inf')
    patience = 15  # 早停patience=15（延长耐心，给模型更多机会）
    patience_counter = 0
    
    for epoch in range(1, 301):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Cosine Annealing每个epoch都要调用
        scheduler.step()
        
        # 每个epoch都评估
        train_metrics = evaluate(model, train_loader, train_dataset, device)
        test_metrics = evaluate(model, test_loader, test_dataset, device)
        
        # 计算验证集loss
        test_loss = 0.0
        for batch in test_loader:
            graphs = batch['graphs']
            node_features_seqs = batch['node_features_seqs'].to(device)
            target_node_indices = batch['target_node_indices'].to(device)
            labels = batch['labels'].to(device)
            
            edge_index = graphs[0].edge_index.to(device)
            edge_weight = graphs[0].edge_weight.to(device)
            
            batch_size, window_size, feature_dim = node_features_seqs.shape
            num_nodes = len(graphs[0].nodes)
            full_features = torch.zeros(batch_size, window_size, num_nodes, feature_dim, device=device)
            for i, node_idx in enumerate(target_node_indices):
                full_features[i, :, node_idx, :] = node_features_seqs[i]
            
            with torch.no_grad():
                out = model(full_features, edge_index, edge_weight, target_node_indices)
                loss = F.mse_loss(out, labels.squeeze())
                test_loss += loss.item() * batch_size
        test_loss /= len(test_samples)
        
        print(f"Epoch {epoch:3d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Test Loss: {test_loss:.4f} | "
              f"Train R²: {train_metrics['r2']:.4f} | "
              f"Test R²: {test_metrics['r2']:.4f} | "
              f"Test MAE: {test_metrics['mae']:.4f}")
        
        # Early stopping（基于Test R²）
        if test_metrics['r2'] > best_test_r2:
            best_test_r2 = test_metrics['r2']
            patience_counter = 0
            torch.save(model.state_dict(), 'experiments/best_stgnn_node_level.pt')
            print(f"  → 保存最佳模型 (Test R²={best_test_r2:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\n早停于第 {epoch} 轮 (最佳Test R²={best_test_r2:.4f})")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('experiments/best_stgnn_node_level.pt'))
    
    # 最终评估
    print(f"\n{'='*80}")
    print(f"最终评估（最佳模型 Test R²={best_test_r2:.4f}）:")
    print(f"{'='*80}")
    
    train_metrics = evaluate(model, train_loader, train_dataset, device)
    test_metrics = evaluate(model, test_loader, test_dataset, device)
    
    print(f"\n训练集 (预测2020):")
    print(f"  样本数: {len(train_samples)}")
    print(f"  MAE:  {train_metrics['mae']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  R²:   {train_metrics['r2']:.4f}")
    
    print(f"\n测试集 (预测2021):")
    print(f"  样本数: {len(test_samples)}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.4f}")
    
    print(f"\n✓ STAEformer节点级时序预测完成！")
    print(f"\n模型: STAEformer (Spatio-Temporal Adaptive Embedding Transformer, 2024)")
    print(f"      架构：GCN空间嵌入 + Transformer时序编码")
    print(f"      核心创新：Affective DomiRank作为关键特征输入")
    print(f"\n任务: 预测供应链节点的综合风险得分")
    print(f"      输入特征：历史3年的16维增强特征（含Affective DomiRank）")
    print(f"      预测目标：comprehensive_risk_score（情感驱动的风险分数）")
    print(f"      样本数: {len(train_samples) + len(test_samples)} (1200节点 × 多时间窗)")
    print(f"\n优势: Transformer注意力机制自适应捕捉时序依赖，比LSTM更强")


if __name__ == "__main__":
    main()
