"""
网络可视化
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
import numpy as np


class NetworkVisualizer:
    """网络可视化器"""
    
    def __init__(self, output_dir: str = 'results/figures'):
        """
        初始化
        
        参数:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_network(
        self,
        G: nx.DiGraph,
        node_colors: Optional[Dict] = None,
        node_sizes: Optional[Dict] = None,
        title: str = "Supply Chain Network",
        filename: str = "network.png",
        layout: str = 'spring'
    ):
        """
        绘制网络图
        
        参数:
            G: NetworkX图
            node_colors: 节点颜色字典
            node_sizes: 节点大小字典
            title: 图标题
            filename: 保存文件名
            layout: 布局算法
        """
        plt.figure(figsize=(16, 12))
        
        # 计算布局
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.circular_layout(G)
        
        # 节点颜色（基于情感或其他属性）
        if node_colors is None:
            # 使用情感极性作为颜色
            node_colors = []
            for node in G.nodes():
                sentiment = G.nodes[node].get('sentiment_polarity', 0.0)
                node_colors.append(sentiment)
        else:
            node_colors = [node_colors.get(node, 0) for node in G.nodes()]
        
        # 节点大小（基于DomiRank或度）
        if node_sizes is None:
            node_sizes = [G.degree(node, weight='weight') for node in G.nodes()]
            # 归一化
            max_size = max(node_sizes) if node_sizes else 1
            node_sizes = [s / max_size * 3000 + 100 for s in node_sizes]
        else:
            node_sizes = [node_sizes.get(node, 300) for node in G.nodes()]
        
        # 绘制网络
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            cmap='RdYlGn',  # 红-黄-绿色图（负-中-正情感）
            vmin=-1, vmax=1,
            alpha=0.8
        )
        
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10,
            width=0.5
        )
        
        # 只标注关键节点（有情感的节点）
        labels = {}
        for node in G.nodes():
            if G.nodes[node].get('sentiment_intensity', 0) > 0:
                # 简化标签（只显示国家）
                labels[node] = node.split('_')[0]
        
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # 添加colorbar
        sm = plt.cm.ScalarMappable(cmap='RdYlGn', norm=plt.Normalize(vmin=-1, vmax=1))
        sm.set_array([])
        plt.colorbar(sm, label='Sentiment Polarity', shrink=0.8, ax=plt.gca())
        
        plt.axis('off')
        plt.tight_layout()
        
        # 保存
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 网络图已保存: {filepath}")
    
    def plot_centrality_comparison(
        self,
        centrality_dict: Dict[str, Dict[str, float]],
        top_k: int = 15,
        filename: str = "centrality_comparison.png"
    ):
        """
        对比不同中心性指标
        
        参数:
            centrality_dict: {metric_name: {node: value}}
            top_k: 显示前k个节点
            filename: 文件名
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, (metric_name, values) in enumerate(centrality_dict.items()):
            if idx >= len(axes):
                break
            
            # 排序取Top k
            sorted_nodes = sorted(values.items(), key=lambda x: x[1], reverse=True)[:top_k]
            nodes = [n[0].split('_')[0] for n in sorted_nodes]  # 简化名称
            scores = [n[1] for n in sorted_nodes]
            
            # 绘制条形图
            axes[idx].barh(nodes, scores, color='steelblue')
            axes[idx].set_xlabel('Score')
            axes[idx].invert_yaxis()
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 中心性对比图已保存: {filepath}")


def plot_emotion_heatmap(
    emotion_df,
    filename: str = "results/figures/emotion_heatmap.png"
):
    """绘制情感热力图"""
    plt.figure(figsize=(12, 8))
    
    # 选择情感特征列
    emotion_cols = ['sentiment_polarity', 'sentiment_intensity', 'fear', 'trust']
    pivot_data = emotion_df.pivot_table(
        index='node_id',
        columns='period',
        values='sentiment_polarity'
    )
    
    sns.heatmap(pivot_data, annot=True, cmap='RdYlGn', center=0, fmt='.2f')
    plt.tight_layout()
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 情感热力图已保存: {filename}")

