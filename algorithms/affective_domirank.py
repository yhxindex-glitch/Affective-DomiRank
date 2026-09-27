"""
情感增强的DomiRank算法（Affective DomiRank）
核心创新：将情感门控机制融入DomiRank
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
from typing import Dict, Optional
from .domirank import DomiRank


class EmotionGate(nn.Module):
    """情感门控神经网络"""
    
    def __init__(self, emotion_dim: int = 7, hidden_dim: int = 16):
        """
        初始化情感门控
        
        参数:
            emotion_dim: 情感向量维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, emotion_vector: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            emotion_vector: 情感向量 [batch_size, emotion_dim]
            
        返回:
            门控值 [batch_size, 1]，范围[0, 1]
        """
        return self.gate(emotion_vector)


class AffectiveDomiRank(DomiRank):
    """情感增强的DomiRank算法"""
    
    def __init__(
        self,
        G: nx.DiGraph,
        emotion_vectors: Dict[str, np.ndarray],
        alpha: float = 1.0,
        beta: float = 0.1,
        theta: float = 1.0,
        gamma: float = 0.1,  # 情感调制强度
        device: str = 'cpu'
    ):
        """
        初始化Affective DomiRank（加性情感调制）
        
        核心创新：在DomiRank的ODE中加入情感驱动项
        dΓ_i/dt = α·Σ_j[θ - Γ_j] - β·Γ_i + γ·s(e_i)
        
        参数:
            G: NetworkX有向图
            emotion_vectors: 节点情感向量字典 {node_id: emotion_vector}
            alpha, beta, theta: DomiRank参数
            gamma: 情感调制强度（默认0.1）
            device: 计算设备
        """
        super().__init__(G, alpha, beta, theta)
        
        self.emotion_vectors = emotion_vectors
        self.gamma = gamma
        self.device = device
        
        # 获取情感向量维度
        sample_emotion = next(iter(emotion_vectors.values()))
        self.emotion_dim = len(sample_emotion) if isinstance(sample_emotion, (list, np.ndarray)) else 7
        
        # 预计算所有节点的情感风险分数 s(e)
        self.sentiment_risk_scores = self._compute_all_sentiment_risk()
    
    def _compute_all_sentiment_risk(self) -> Dict[str, float]:
        """
        【新方法】预计算所有节点的情感风险分数（用于加法融合）
        
        情感风险分数：
        - 负面情感（sentiment < 0）→ 高风险（正值）
        - 正面情感（sentiment > 0）→ 低风险（负值或0）
        - 归一化到合理范围
        
        返回:
            节点到情感风险分数的字典
        """
        risk_dict = {}
        
        for node in self.nodes:
            emotion = self._get_emotion_vector(node)
            
            # 提取情感极性（第一个维度）
            sentiment = emotion[0] if len(emotion) > 0 else 0.0
            
            # 转换为风险分数：
            # sentiment = -1（极度负面）→ risk = +1（高风险）
            # sentiment = 0（中性）→ risk = 0（无额外风险）
            # sentiment = +1（极度正面）→ risk = -1（降低风险）
            sentiment_risk = -sentiment  # 简单反转
            
            # 或者使用非线性映射增强区分度
            if abs(sentiment) < 0.05:
                # 情感太弱，视为无信号
                sentiment_risk = 0.0
            elif sentiment < 0:
                # 负面情感 → 增加风险（非线性放大）
                sentiment_risk = abs(sentiment) ** 0.8
            else:
                # 正面情感 → 减少风险（非线性削弱）
                sentiment_risk = -(sentiment ** 0.8) * 0.5  # 正面影响较弱
            
            risk_dict[node] = sentiment_risk
        
        return risk_dict
    
    def _compute_all_sigma(self) -> Dict[str, float]:
        """
        【旧方法】预计算所有节点的情感门控值 σ(e_j)（用于乘法门控）
        
        返回:
            节点到σ值的字典
        """
        sigma_dict = {}
        
        for node in self.nodes:
            emotion = self._get_emotion_vector(node)
            sigma = self._compute_sigma(emotion)
            sigma_dict[node] = sigma
        
        return sigma_dict
    
    def _get_emotion_vector(self, node: str) -> np.ndarray:
        """
        获取节点的情感向量
        
        参数:
            node: 节点ID
            
        返回:
            情感向量（如果节点没有情感，返回中性向量）
        """
        if node in self.emotion_vectors:
            emotion = self.emotion_vectors[node]
            if isinstance(emotion, dict):
                # 如果是字典，转换为数组
                emotion_array = np.array([
                    emotion.get('sentiment_polarity', 0.0),
                    emotion.get('sentiment_intensity', 0.0),
                    emotion.get('fear', 0.0),
                    emotion.get('trust', 0.0),
                    emotion.get('anger', 0.0),
                    emotion.get('joy', 0.0),
                    emotion.get('sadness', 0.0),
                ])
                return emotion_array
            else:
                return np.array(emotion)
        else:
            # 返回中性情感向量
            return np.zeros(self.emotion_dim)
    
    def _compute_sigma(self, emotion_vector: np.ndarray) -> float:
        """
        计算情感门控值 σ(e)（改进版）
        
        改进要点：
        1. 无情感节点返回1.0，不影响DomiRank
        2. 有情感节点根据极性调制：负面放大风险，正面抑制风险
        3. 非线性映射，增强区分度
        
        参数:
            emotion_vector: 情感向量
            
        返回:
            门控值，范围根据情感调整
        """
        if self.gate_type == 'learned':
            # 使用学习的神经网络
            with torch.no_grad():
                emotion_tensor = torch.FloatTensor(emotion_vector).unsqueeze(0).to(self.device)
                sigma = self.emotion_gate_nn(emotion_tensor).item()
            return sigma
        
        elif self.gate_type == 'handcrafted':
            # 改进的手工设计规则
            # 提取情感极性（第一个维度）
            sentiment = emotion_vector[0] if len(emotion_vector) > 0 else 0.0
            
            # 【关键改进1】检查是否有情感数据
            # 如果情感信号太弱（|sentiment| < 0.05），视为无情感
            if abs(sentiment) < 0.05:
                return 1.0  # 无情感=不调制，保持原DomiRank
            
            # 【关键改进2】强化的非线性映射（增大调制幅度）
            # σ(e)的含义：情感对风险传播的调制因子
            # 
            # 负面情感 → σ > 1.0 → 显著放大风险传播
            # 正面情感 → σ < 1.0 → 显著抑制风险传播
            # 中性情感 → σ = 1.0 → 不改变风险传播
            
            # 【增强版】使用指数映射，增强情感的影响
            # 这样即使情感数据稀疏，有情感的节点也会产生显著效果
            
            if sentiment < 0:
                # 负面情感：σ ∈ (1.0, 2.0]（增大范围）
                # sentiment=-1（极度负面）→ σ=2.0（放大100%风险）
                # sentiment=-0.5（中等负面）→ σ=1.5（放大50%风险）
                # sentiment=-0.2（轻微负面）→ σ=1.2（放大20%风险）
                sigma = 1.0 + abs(sentiment) ** 0.8  # 0.8次方，非线性放大
            else:
                # 正面情感：σ ∈ [0.3, 1.0)（增大范围）
                # sentiment=1（极度正面）→ σ=0.3（削弱70%风险）
                # sentiment=0.5（中等正面）→ σ=0.65（削弱35%风险）
                # sentiment=0.2（轻微正面）→ σ=0.86（削弱14%风险）
                sigma = 1.0 - 0.7 * (sentiment ** 0.8)  # 0.8次方，非线性削弱
            
            return sigma
        
        else:
            return 1.0  # 默认不调制
    
    # 直接使用父类的compute方法，通过_dgamma_dt实现加性调制
    
    def _dgamma_dt(self, gamma: np.ndarray) -> np.ndarray:
        """
        加性情感调制的Affective DomiRank微分方程
        
        dΓ_i/dt = α·Σ_j∈N(i)[θ - Γ_j] - β·Γ_i + γ·s(e_i)
        
        参数:
            gamma: 当前Γ值数组
            
        返回:
            dΓ/dt数组
        """
        dgamma = np.zeros(self.n)
        
        for i, node in enumerate(self.nodes):
            # 1. 标准DomiRank项：邻居贡献
            neighbor_sum = 0.0
            for neighbor in self.G.predecessors(node):
                j = self.node_to_idx[neighbor]
                neighbor_sum += (self.theta - gamma[j])
            
            # 2. 情感驱动项
            sentiment_force = self.sentiment_risk_scores[node]
            
            # 3. 组合：标准项 + 情感调制项
            dgamma[i] = (self.alpha * neighbor_sum 
                         - self.beta * gamma[i] 
                         + self.gamma * sentiment_force)
        
        return dgamma
    
    def train_gate(
        self,
        validation_metric: callable,
        epochs: int = 50,
        learning_rate: float = 0.001
    ):
        """
        训练情感门控参数（如果使用learned gate）
        
        参数:
            validation_metric: 验证指标函数
            epochs: 训练轮数
            learning_rate: 学习率
        """
        if self.gate_type != 'learned':
            print("  只有learned gate需要训练")
            return
        
        optimizer = torch.optim.Adam(
            self.emotion_gate_nn.parameters(),
            lr=learning_rate
        )
        
        print(f"  训练情感门控参数（{epochs}轮）...")
        
        for epoch in range(epochs):
            # 重新计算σ值
            self.sigma_values = self._compute_all_sigma()
            
            # 计算DomiRank
            gamma_values = self.compute(method='euler', max_iter=500)
            
            # 计算验证指标（例如：节点识别准确率）
            metric_value = validation_metric(gamma_values)
            
            # 反向传播（简化：这里需要根据具体任务设计损失）
            # 实际使用时需要定义合适的损失函数
            
            if (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1}/{epochs}, Metric: {metric_value:.4f}")


if __name__ == "__main__":
    # 测试代码
    import pickle
    import pandas as pd
    
    print("=" * 80)
    print("测试Affective DomiRank算法")
    print("=" * 80)
    
    # 加载网络
    networks_file = "data/processed/final/supply_chain_network_with_emotions.pkl"
    
    with open(networks_file, 'rb') as f:
        networks = pickle.load(f)
    
    # 选择一个时间段
    period = list(networks.keys())[0]
    G = networks[period]
    
    print(f"\n使用网络: {period}")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}\n")
    
    # 提取情感向量
    emotion_vectors = {}
    for node in G.nodes():
        attrs = G.nodes[node]
        emotion_vectors[node] = {
            'sentiment_polarity': attrs.get('sentiment_polarity', 0.0),
            'sentiment_intensity': attrs.get('sentiment_intensity', 0.0),
            'fear': attrs.get('fear', 0.0),
            'trust': attrs.get('trust', 0.0),
            'anger': attrs.get('anger', 0.0),
            'joy': attrs.get('joy', 0.0),
            'sadness': attrs.get('sadness', 0.0),
        }
    
    # 计算Affective DomiRank
    print("使用手工设计的情感门控...")
    affective_domirank = AffectiveDomiRank(
        G,
        emotion_vectors,
        alpha=1.0,
        beta=0.1,
        theta=1.0,
        gate_type='handcrafted'
    )
    
    values = affective_domirank.compute(method='euler')
    
    # 识别关键节点
    critical_nodes = affective_domirank.identify_critical_nodes(values, top_k=10)
    
    print("\n关键节点（Affective DomiRank Top 10）:")
    for i, (node, score) in enumerate(critical_nodes, 1):
        emotion = emotion_vectors[node]
        print(f"  {i}. {node}: {score:.4f}")
        print(f"      情感: polarity={emotion['sentiment_polarity']:.2f}, fear={emotion['fear']:.2f}")
    
    print("\n✓ 测试完成!")

