"""
参数敏感性分析实验
一次性运行所有参数组合的实验，保存结果供后续绘图使用

分析参数：
1. α (alpha): DomiRank阻尼因子 - 控制邻居贡献权重
2. τ (threshold): 情感门控阈值 - 控制情感信号的触发条件
3. w (window_size): 时间窗口大小 - 历史数据的时间跨度

实验设计：
- 单参数敏感性分析（固定其他参数，改变一个参数）
- 双参数交互效应（α×τ热力图）
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
import copy
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm, trange

sys.path.append(str(Path(__file__).parent.parent))

from models.stgnn import STGNN, STGNNConfig
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import networkx as nx

# ============================================================================
# 全局配置（针对32GB显存优化）
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RANDOM_SEED = 42
EPOCHS_PER_EXPERIMENT = 50   # 敏感性分析50轮足够看出趋势
PATIENCE = 5                  # 早停patience减小，快速跳过不收敛的配置
BATCH_SIZE = 256              # 32GB显存可以用大batch，加速4-8倍

# 参数范围定义（精简版，保证覆盖关键区间）
ALPHA_VALUES = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]  # DomiRank阻尼因子（6个）
THRESHOLD_VALUES = [0.02, 0.05, 0.08, 0.1, 0.15]     # 情感门控阈值（5个）
WINDOW_VALUES = [2, 3, 4]                             # 时间窗口大小（3个）

# 默认参数（基准配置）
DEFAULT_ALPHA = 0.01
DEFAULT_THRESHOLD = 0.05
DEFAULT_GAMMA = 0.1
DEFAULT_WINDOW = 3


# ============================================================================
# Affective DomiRank计算（支持自定义参数）
# ============================================================================
class ParameterizedAffectiveDomiRank:
    """
    参数化的Affective DomiRank算法
    支持动态调整alpha、threshold和gamma参数
    """
    
    def __init__(
        self,
        G: nx.DiGraph,
        alpha: float = 0.01,
        beta: float = 0.5,
        theta: float = 1.0,
        gamma: float = 0.1,
        threshold: float = 0.05
    ):
        """
        初始化参数化Affective DomiRank
        
        参数:
            G: NetworkX有向图
            alpha: 支配增益系数
            beta: 衰减系数
            theta: 支配阈值
            gamma: 情感调制强度
            threshold: 情感门控阈值（情感绝对值低于此值视为无信号）
        """
        self.G = G
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.gamma = gamma
        self.threshold = threshold
        
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
        # 预计算情感风险分数
        self.sentiment_risk_scores = self._compute_sentiment_risk()
    
    def _compute_sentiment_risk(self):
        """计算所有节点的情感风险分数"""
        risk_dict = {}
        
        for node in self.nodes:
            sentiment = self.G.nodes[node].get('sentiment_mean', 0.0)
            
            # 使用参数化的阈值
            if abs(sentiment) < self.threshold:
                sentiment_risk = 0.0
            elif sentiment < 0:
                # 负面情感 → 增加风险
                sentiment_risk = abs(sentiment) ** 0.8
            else:
                # 正面情感 → 减少风险
                sentiment_risk = -(sentiment ** 0.8) * 0.5
            
            risk_dict[node] = sentiment_risk
        
        return risk_dict
    
    def compute(self, max_iter=1000, tol=1e-6, time_step=0.01):
        """计算Affective DomiRank值"""
        gamma_vals = np.ones(self.n)
        
        for iteration in range(max_iter):
            dgamma = self._dgamma_dt(gamma_vals)
            gamma_new = gamma_vals + time_step * dgamma
            
            if np.linalg.norm(gamma_new - gamma_vals) < tol:
                break
            
            gamma_vals = gamma_new
        
        return {node: gamma_vals[i] for i, node in enumerate(self.nodes)}
    
    def _dgamma_dt(self, gamma):
        """Affective DomiRank微分方程"""
        dgamma = np.zeros(self.n)
        
        for i, node in enumerate(self.nodes):
            neighbor_sum = 0.0
            for neighbor in self.G.predecessors(node):
                j = self.node_to_idx[neighbor]
                neighbor_sum += (self.theta - gamma[j])
            
            sentiment_force = self.sentiment_risk_scores[node]
            
            dgamma[i] = (self.alpha * neighbor_sum 
                         - self.beta * gamma[i] 
                         + self.gamma * sentiment_force)
        
        return dgamma


def recompute_affective_domirank(networks, alpha, threshold, gamma=DEFAULT_GAMMA):
    """
    使用新参数重新计算所有年份网络的Affective DomiRank
    
    参数:
        networks: 年度网络字典
        alpha: DomiRank阻尼因子
        threshold: 情感门控阈值
        gamma: 情感调制强度
    
    返回:
        更新后的网络字典（深拷贝）
    """
    updated_networks = {}
    
    for year, G in networks.items():
        # 深拷贝网络
        G_new = copy.deepcopy(G)
        
        # 使用新参数计算Affective DomiRank
        affective_dr = ParameterizedAffectiveDomiRank(
            G_new,
            alpha=alpha,
            beta=0.5,
            theta=1.0,
            gamma=gamma,
            threshold=threshold
        )
        
        scores = affective_dr.compute(max_iter=2000, tol=1e-6)
        
        # 更新节点属性
        for node, score in scores.items():
            G_new.nodes[node]['affective_domirank'] = score
        
        # 重新计算邻居特征
        for node in G_new.nodes():
            predecessors = list(G_new.predecessors(node))
            if predecessors:
                domiranks = [G_new.nodes[p].get('affective_domirank', 0) for p in predecessors]
                G_new.nodes[node]['neighbor_avg_risk'] = np.mean(domiranks)
        
        updated_networks[year] = G_new
    
    return updated_networks


# ============================================================================
# 数据集构建（简化版，支持自定义窗口大小）
# ============================================================================
class SensitivityDataset:
    """敏感性分析用数据集"""
    
    def __init__(self, networks, window_size=3, risk_labels_path='data/processed/networks/risk_labels.pkl'):
        self.networks = networks
        self.years = sorted(networks.keys())
        self.window_size = window_size
        
        # 加载风险标签
        with open(risk_labels_path, 'rb') as f:
            self.risk_labels_df = pickle.load(f)
        
        self.all_nodes = sorted(list(networks[self.years[0]].nodes()))
        self.node_to_idx = {node: i for i, node in enumerate(self.all_nodes)}
        
        # 预计算特征缓存
        self._precompute_features_cache()
        
        # 预转换PyG图
        self._precompute_pyg_graphs()
        
        # 计算DomiRank归一化参数
        self._compute_domirank_normalization()
        
        # 构建样本
        self.samples = self._build_samples()
        
        # 归一化
        self._compute_normalization_stats()
        self._normalize_data()
    
    def _precompute_features_cache(self):
        """预计算特征缓存"""
        self.features_cache = {}
        feature_attrs = ['in_degree', 'out_degree', 'pagerank', 'sentiment_mean',
                        'news_count', 'affective_domirank', 'domirank', 'neighbor_avg_sentiment',
                        'neighbor_avg_risk', 'supplier_concentration', 'total_degree']
        
        for year in self.years:
            G = self.networks[year]
            year_features = {}
            for node in self.all_nodes:
                if node in G.nodes():
                    node_data = G.nodes[node]
                    year_features[node] = {attr: node_data.get(attr, 0) for attr in feature_attrs}
                else:
                    year_features[node] = {attr: 0 for attr in feature_attrs}
            self.features_cache[year] = year_features
    
    def _precompute_pyg_graphs(self):
        """预转换PyG图"""
        self.pyg_graphs = {}
        for year in self.years:
            self.pyg_graphs[year] = self._convert_to_pyg(self.networks[year])
    
    def _convert_to_pyg(self, G):
        """转换为PyG格式"""
        nodes = sorted(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        
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
    
    def _compute_domirank_normalization(self):
        """计算DomiRank归一化参数"""
        all_affective_dr = []
        for year in self.years:
            G = self.networks[year]
            for node in G.nodes():
                adr = G.nodes[node].get('affective_domirank', 0)
                all_affective_dr.append(adr)
        
        self.affective_dr_min = np.min(all_affective_dr)
        self.affective_dr_max = np.max(all_affective_dr)
    
    def _domirank_to_vulnerability(self, domirank_value):
        """将DomiRank转换为脆弱性指数"""
        dr_min, dr_max = self.affective_dr_min, self.affective_dr_max
        if dr_max - dr_min > 1e-8:
            normalized = (domirank_value - dr_min) / (dr_max - dr_min)
        else:
            normalized = 0.5
        return 1.0 - normalized
    
    def _build_samples(self):
        """构建样本"""
        samples = []
        time_windows = []
        
        for i in range(len(self.years) - self.window_size):
            input_years = self.years[i:i+self.window_size]
            target_year = self.years[i+self.window_size]
            time_windows.append((input_years, target_year))
        
        for node in self.all_nodes:
            for window_idx, (input_years, target_year) in enumerate(time_windows):
                node_features_seq = []
                
                # 计算历史变化率
                hist_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:
                        prev_year = input_years[i-1]
                        trade_prev = self.features_cache[prev_year][node]['total_degree']
                        trade_curr = self.features_cache[year][node]['total_degree']
                        if trade_prev > 0:
                            change = (trade_curr - trade_prev) / trade_prev
                            hist_changes.append(change)
                
                hist_volatility = np.std(hist_changes) if len(hist_changes) > 0 else 0
                hist_acceleration = (hist_changes[-1] - hist_changes[0]) if len(hist_changes) >= 2 else 0
                hist_avg_change = np.mean(hist_changes) if len(hist_changes) > 0 else 0
                
                # 情感变化率
                sentiment_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:
                        prev_year = input_years[i-1]
                        sent_prev = self.features_cache[prev_year][node]['sentiment_mean']
                        sent_curr = self.features_cache[year][node]['sentiment_mean']
                        sentiment_changes.append(sent_curr - sent_prev)
                
                sentiment_change_rate = np.mean(sentiment_changes) if len(sentiment_changes) > 0 else 0
                
                # 增强特征
                trade_momentum = hist_avg_change * hist_volatility
                
                degree_changes = []
                for i, year in enumerate(input_years):
                    if i > 0:
                        prev_year = input_years[i-1]
                        deg_prev = self.features_cache[prev_year][node]['total_degree']
                        deg_curr = self.features_cache[year][node]['total_degree']
                        if deg_prev > 0:
                            degree_changes.append((deg_curr - deg_prev) / deg_prev)
                degree_change_rate = np.mean(degree_changes) if len(degree_changes) > 0 else 0
                
                last_year_node_sentiment = self.features_cache[input_years[-1]][node]['sentiment_mean']
                last_year_neighbor_sentiment = self.features_cache[input_years[-1]][node]['neighbor_avg_sentiment']
                sentiment_risk_score = last_year_node_sentiment * last_year_neighbor_sentiment
                
                for year in input_years:
                    cached_features = self.features_cache[year][node]
                    affective_vulnerability = self._domirank_to_vulnerability(
                        cached_features['affective_domirank']
                    )
                    
                    features = [
                        cached_features['in_degree'],
                        cached_features['out_degree'],
                        cached_features['pagerank'],
                        cached_features['sentiment_mean'],
                        cached_features['news_count'],
                        affective_vulnerability,
                        cached_features['neighbor_avg_sentiment'],
                        cached_features['neighbor_avg_risk'],
                        cached_features['supplier_concentration'],
                        hist_avg_change,
                        hist_volatility,
                        hist_acceleration,
                        sentiment_change_rate,
                        trade_momentum,
                        degree_change_rate,
                        sentiment_risk_score,
                    ]
                    node_features_seq.append(features)
                
                # 获取标签
                target_idx = self.years.index(target_year)
                if target_idx > 0:
                    year_from = self.years[target_idx - 1]
                    year_to = target_year
                    risk_row = self.risk_labels_df[
                        (self.risk_labels_df['year_from'] == year_from) &
                        (self.risk_labels_df['year_to'] == year_to) &
                        (self.risk_labels_df['node'] == node)
                    ]
                    label = risk_row.iloc[0]['comprehensive_risk_score'] if len(risk_row) > 0 else 0.0
                else:
                    label = 0.0
                
                sample = {
                    'node': node,
                    'window_idx': window_idx,
                    'input_years': input_years,
                    'target_year': target_year,
                    'node_features_seq': torch.FloatTensor(node_features_seq),
                    'label': torch.FloatTensor([label]),
                    'sentiment': last_year_node_sentiment,
                    'graphs': [self.pyg_graphs[y] for y in input_years]
                }
                samples.append(sample)
        
        return samples
    
    def _compute_normalization_stats(self):
        """计算归一化统计量"""
        all_features = []
        all_labels = []
        
        for sample in self.samples:
            all_features.append(sample['node_features_seq'].numpy())
            all_labels.append(sample['label'].numpy())
        
        all_features = np.vstack([f for fs in all_features for f in fs])
        all_labels = np.vstack(all_labels)
        
        self.feature_mean = all_features.mean(axis=0)
        self.feature_std = all_features.std(axis=0) + 1e-8
        self.label_mean = all_labels.mean()
        self.label_std = all_labels.std() + 1e-8
    
    def _normalize_data(self):
        """归一化数据"""
        for sample in self.samples:
            sample['node_features_seq'] = (sample['node_features_seq'] - torch.FloatTensor(self.feature_mean)) / torch.FloatTensor(self.feature_std)
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
    graphs = batch[0]['graphs']
    node_features_seqs = torch.stack([s['node_features_seq'] for s in batch])
    labels = torch.stack([s['label'] for s in batch])
    sentiments = torch.FloatTensor([s['sentiment'] for s in batch])
    
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


# ============================================================================
# 训练和评估函数
# ============================================================================
def train_epoch(model, dataloader, optimizer, device, pbar=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in dataloader:
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
        
        optimizer.zero_grad()
        out = model(full_features, edge_index, edge_weight, target_node_indices)
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
        
        out = model(full_features, edge_index, edge_weight, target_node_indices)
        
        all_preds.append(out.cpu())
        all_labels.append(labels)
    
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    preds_denorm = dataset.denormalize_labels(preds)
    labels_denorm = dataset.denormalize_labels(labels)
    
    mae = mean_absolute_error(labels_denorm, preds_denorm)
    rmse = np.sqrt(mean_squared_error(labels_denorm, preds_denorm))
    r2 = r2_score(labels_denorm, preds_denorm)
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


def run_single_experiment(networks, window_size, num_nodes, device):
    """
    运行单个实验配置（32GB显存优化版）
    
    返回:
        dict: 包含train和test指标
    """
    import random
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # 构建数据集
    dataset = SensitivityDataset(networks, window_size=window_size)
    
    # 划分数据集
    all_samples = dataset.samples.copy()
    random.shuffle(all_samples)
    
    split_idx = int(len(all_samples) * 0.8)
    train_samples = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    from training.splits import assert_disjoint_samples
    assert_disjoint_samples(train_samples, test_samples)
    
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
    
    stats = {'label_mean': dataset.label_mean, 'label_std': dataset.label_std}
    train_dataset = SimpleDataset(train_samples, stats)
    test_dataset = SimpleDataset(test_samples, stats)
    
    # 32GB显存优化：大batch（Windows不支持多进程DataLoader）
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        collate_fn=collate_fn, num_workers=0, pin_memory=True
    )
    
    # 创建模型
    config = STGNNConfig(
        node_feature_dim=16,
        num_nodes=num_nodes,
        embed_dim=16,
        hidden_dim=64,
        num_time_steps=window_size,
        num_layers=2,
        cheb_k=2,
        dropout=0.1,
        output_dim=1
    )
    
    model = STGNN(config).to(device)
    
    # 32GB显存优化：使用更大学习率配合大batch
    optimizer = optim.AdamW(model.parameters(), lr=0.003, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=1e-6)
    
    # 启用cudnn优化
    torch.backends.cudnn.benchmark = True
    
    # 训练（带进度条）
    best_test_r2 = -float('inf')
    patience_counter = 0
    best_metrics = None
    
    pbar = tqdm(range(1, EPOCHS_PER_EXPERIMENT + 1), 
                desc="    训练中", leave=False, ncols=80,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for epoch in pbar:
        train_loss = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        # 每5个epoch评估一次（加快迭代）
        if epoch % 5 == 0:
            test_metrics = evaluate(model, test_loader, test_dataset, device)
            pbar.set_postfix({'R²': f"{test_metrics['r2']:.3f}"})
            
            if test_metrics['r2'] > best_test_r2:
                best_test_r2 = test_metrics['r2']
                best_metrics = test_metrics.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= PATIENCE:
                break
    
    pbar.close()
    
    # 最终评估
    train_metrics = evaluate(model, train_loader, train_dataset, device)
    test_metrics = evaluate(model, test_loader, test_dataset, device)
    
    # 清理显存
    del model
    torch.cuda.empty_cache()
    
    return {
        'train': train_metrics,
        'test': test_metrics,
        'best_test_r2': best_test_r2
    }


# ============================================================================
# 主实验函数
# ============================================================================
def run_sensitivity_analysis():
    """运行完整的敏感性分析（32GB显存优化版）"""
    print("="*80)
    print("参数敏感性分析实验 [32GB显存优化版]")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # 计算总实验数
    total_experiments = (len(ALPHA_VALUES) + len(THRESHOLD_VALUES) + 
                        len(WINDOW_VALUES) + len(ALPHA_VALUES) * len(THRESHOLD_VALUES))
    print(f"\n预计实验数: {total_experiments} 组")
    print(f"  - α敏感性: {len(ALPHA_VALUES)} 组")
    print(f"  - τ敏感性: {len(THRESHOLD_VALUES)} 组")
    print(f"  - w敏感性: {len(WINDOW_VALUES)} 组")
    print(f"  - α×τ热力图: {len(ALPHA_VALUES) * len(THRESHOLD_VALUES)} 组")
    print(f"\n优化配置:")
    print(f"  - Batch Size: {BATCH_SIZE}")
    print(f"  - Epochs/实验: {EPOCHS_PER_EXPERIMENT}")
    print(f"  - 早停Patience: {PATIENCE}")
    
    start_time = datetime.now()
    completed_experiments = 0
    
    # 总体进度条（固定在顶部）
    print("\n" + "="*60)
    overall_pbar = tqdm(total=total_experiments, desc="🔬 总体进度", ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                        colour='green')
    print("="*60 + "\n")
    
    # 加载原始网络
    print("\n[1/5] 加载原始网络数据...")
    networks_file = Path('data/processed/networks/all_networks_2017_2021.pkl')
    with open(networks_file, 'rb') as f:
        original_networks = pickle.load(f)
    
    num_nodes = len(list(original_networks[list(original_networks.keys())[0]].nodes()))
    print(f"  网络年份: {sorted(original_networks.keys())}")
    print(f"  节点数: {num_nodes}")
    
    # 存储所有结果
    results = {
        'alpha_sensitivity': [],      # α敏感性
        'threshold_sensitivity': [],  # τ敏感性
        'window_sensitivity': [],     # w敏感性
        'alpha_threshold_heatmap': [],  # α×τ交互效应
        'metadata': {
            'alpha_values': ALPHA_VALUES,
            'threshold_values': THRESHOLD_VALUES,
            'window_values': WINDOW_VALUES,
            'default_alpha': DEFAULT_ALPHA,
            'default_threshold': DEFAULT_THRESHOLD,
            'default_gamma': DEFAULT_GAMMA,
            'default_window': DEFAULT_WINDOW,
            'epochs': EPOCHS_PER_EXPERIMENT,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # ========================================================================
    # 实验1: α (DomiRank阻尼因子) 敏感性分析
    # ========================================================================
    print("\n[2/5] α (DomiRank阻尼因子) 敏感性分析")
    print(f"  测试值: {ALPHA_VALUES}")
    print(f"  固定参数: τ={DEFAULT_THRESHOLD}, w={DEFAULT_WINDOW}\n")
    
    alpha_pbar = tqdm(ALPHA_VALUES, desc="  α敏感性", ncols=100,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for alpha in alpha_pbar:
        alpha_pbar.set_postfix({'α': alpha})
        
        # 重新计算Affective DomiRank
        updated_networks = recompute_affective_domirank(
            original_networks, 
            alpha=alpha, 
            threshold=DEFAULT_THRESHOLD,
            gamma=DEFAULT_GAMMA
        )
        
        # 运行实验
        metrics = run_single_experiment(updated_networks, DEFAULT_WINDOW, num_nodes, DEVICE)
        
        results['alpha_sensitivity'].append({
            'alpha': alpha,
            'train_r2': metrics['train']['r2'],
            'train_mae': metrics['train']['mae'],
            'train_rmse': metrics['train']['rmse'],
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse'],
            'best_test_r2': metrics['best_test_r2']
        })
        
        completed_experiments += 1
        alpha_pbar.set_postfix({'α': alpha, 'R²': f"{metrics['test']['r2']:.3f}"})
        overall_pbar.update(1)
    
    # ========================================================================
    # 实验2: τ (情感门控阈值) 敏感性分析
    # ========================================================================
    print("\n[3/5] τ (情感门控阈值) 敏感性分析")
    print(f"  测试值: {THRESHOLD_VALUES}")
    print(f"  固定参数: α={DEFAULT_ALPHA}, w={DEFAULT_WINDOW}\n")
    
    threshold_pbar = tqdm(THRESHOLD_VALUES, desc="  τ敏感性", ncols=100,
                          bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for threshold in threshold_pbar:
        threshold_pbar.set_postfix({'τ': threshold})
        
        updated_networks = recompute_affective_domirank(
            original_networks,
            alpha=DEFAULT_ALPHA,
            threshold=threshold,
            gamma=DEFAULT_GAMMA
        )
        
        metrics = run_single_experiment(updated_networks, DEFAULT_WINDOW, num_nodes, DEVICE)
        
        results['threshold_sensitivity'].append({
            'threshold': threshold,
            'train_r2': metrics['train']['r2'],
            'train_mae': metrics['train']['mae'],
            'train_rmse': metrics['train']['rmse'],
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse'],
            'best_test_r2': metrics['best_test_r2']
        })
        
        completed_experiments += 1
        threshold_pbar.set_postfix({'τ': threshold, 'R²': f"{metrics['test']['r2']:.3f}"})
        overall_pbar.update(1)
    
    # ========================================================================
    # 实验3: w (时间窗口大小) 敏感性分析
    # ========================================================================
    print("\n[4/5] w (时间窗口大小) 敏感性分析")
    print(f"  测试值: {WINDOW_VALUES}")
    print(f"  固定参数: α={DEFAULT_ALPHA}, τ={DEFAULT_THRESHOLD}\n")
    
    # 使用默认参数的网络
    default_networks = recompute_affective_domirank(
        original_networks,
        alpha=DEFAULT_ALPHA,
        threshold=DEFAULT_THRESHOLD,
        gamma=DEFAULT_GAMMA
    )
    
    window_pbar = tqdm(WINDOW_VALUES, desc="  w敏感性", ncols=100,
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for window in window_pbar:
        window_pbar.set_postfix({'w': window})
        
        metrics = run_single_experiment(default_networks, window, num_nodes, DEVICE)
        
        results['window_sensitivity'].append({
            'window': window,
            'train_r2': metrics['train']['r2'],
            'train_mae': metrics['train']['mae'],
            'train_rmse': metrics['train']['rmse'],
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse'],
            'best_test_r2': metrics['best_test_r2']
        })
        
        completed_experiments += 1
        window_pbar.set_postfix({'w': window, 'R²': f"{metrics['test']['r2']:.3f}"})
        overall_pbar.update(1)
    
    # ========================================================================
    # 实验4: α×τ 双参数交互效应热力图
    # ========================================================================
    heatmap_total = len(ALPHA_VALUES) * len(THRESHOLD_VALUES)
    print(f"\n[5/5] α×τ 双参数交互效应分析 ({heatmap_total}组)")
    print(f"  α范围: {ALPHA_VALUES}")
    print(f"  τ范围: {THRESHOLD_VALUES}\n")
    
    # 创建参数组合列表
    param_combinations = [(a, t) for a in ALPHA_VALUES for t in THRESHOLD_VALUES]
    
    heatmap_pbar = tqdm(param_combinations, desc="  α×τ热力图", ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    for alpha, threshold in heatmap_pbar:
        heatmap_pbar.set_postfix({'α': alpha, 'τ': threshold})
        
        updated_networks = recompute_affective_domirank(
            original_networks,
            alpha=alpha,
            threshold=threshold,
            gamma=DEFAULT_GAMMA
        )
        
        metrics = run_single_experiment(updated_networks, DEFAULT_WINDOW, num_nodes, DEVICE)
        
        results['alpha_threshold_heatmap'].append({
            'alpha': alpha,
            'threshold': threshold,
            'test_r2': metrics['test']['r2'],
            'test_mae': metrics['test']['mae'],
            'test_rmse': metrics['test']['rmse']
        })
        
        completed_experiments += 1
        heatmap_pbar.set_postfix({'α': alpha, 'τ': threshold, 'R²': f"{metrics['test']['r2']:.3f}"})
        overall_pbar.update(1)
    
    # 关闭总体进度条
    overall_pbar.close()
    
    # ========================================================================
    # 保存结果
    # ========================================================================
    output_file = Path('experiments/sensitivity_results.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*80)
    print("敏感性分析完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果已保存: {output_file}")
    print("="*80)
    
    # 打印摘要
    print("\n实验摘要:")
    print(f"  1. α敏感性: {len(results['alpha_sensitivity'])} 组实验")
    print(f"  2. τ敏感性: {len(results['threshold_sensitivity'])} 组实验")
    print(f"  3. w敏感性: {len(results['window_sensitivity'])} 组实验")
    print(f"  4. α×τ热力图: {len(results['alpha_threshold_heatmap'])} 组实验")
    
    # 找出最优参数
    best_alpha = max(results['alpha_sensitivity'], key=lambda x: x['test_r2'])
    best_threshold = max(results['threshold_sensitivity'], key=lambda x: x['test_r2'])
    best_window = max(results['window_sensitivity'], key=lambda x: x['test_r2'])
    
    print(f"\n最优参数:")
    print(f"  最优α: {best_alpha['alpha']} (Test R²={best_alpha['test_r2']:.4f})")
    print(f"  最优τ: {best_threshold['threshold']} (Test R²={best_threshold['test_r2']:.4f})")
    print(f"  最优w: {best_window['window']} (Test R²={best_window['test_r2']:.4f})")
    
    return results


if __name__ == "__main__":
    results = run_sensitivity_analysis()
