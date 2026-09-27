"""
基线中心性算法
实现PageRank、度中心性等传统方法
"""

import numpy as np
import networkx as nx
from typing import Dict, List


class CentralityMeasures:
    """中心性度量集合"""
    
    def __init__(self, G: nx.DiGraph):
        """
        初始化
        
        参数:
            G: NetworkX有向图
        """
        self.G = G
    
    def pagerank(
        self,
        alpha: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        计算PageRank中心性
        
        参数:
            alpha: 阻尼系数
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            节点到PageRank值的字典
        """
        print(f"  计算PageRank (alpha={alpha})...")
        
        try:
            pr = nx.pagerank(
                self.G,
                alpha=alpha,
                max_iter=max_iter,
                tol=tol,
                weight='weight'
            )
            return pr
        except:
            # 如果失败，使用无权重版本
            pr = nx.pagerank(self.G, alpha=alpha, max_iter=max_iter, tol=tol)
            return pr
    
    def degree_centrality(self, mode: str = 'total') -> Dict[str, float]:
        """
        计算度中心性
        
        参数:
            mode: 'in', 'out', 'total'
            
        返回:
            节点到度中心性的字典
        """
        print(f"  计算度中心性 (mode={mode})...")
        
        if mode == 'in':
            degree_dict = dict(self.G.in_degree(weight='weight'))
        elif mode == 'out':
            degree_dict = dict(self.G.out_degree(weight='weight'))
        elif mode == 'total':
            in_deg = dict(self.G.in_degree(weight='weight'))
            out_deg = dict(self.G.out_degree(weight='weight'))
            degree_dict = {
                node: in_deg.get(node, 0) + out_deg.get(node, 0)
                for node in self.G.nodes()
            }
        else:
            raise ValueError(f"不支持的mode: {mode}")
        
        # 归一化
        if degree_dict:
            max_degree = max(degree_dict.values())
            if max_degree > 0:
                degree_dict = {k: v/max_degree for k, v in degree_dict.items()}
        
        return degree_dict
    
    def betweenness_centrality(self, weight: str = 'weight') -> Dict[str, float]:
        """
        计算介数中心性
        
        参数:
            weight: 边权重属性名
            
        返回:
            节点到介数中心性的字典
        """
        print(f"  计算介数中心性...")
        
        try:
            bc = nx.betweenness_centrality(self.G, weight=weight)
        except:
            bc = nx.betweenness_centrality(self.G)
        
        return bc
    
    def closeness_centrality(self) -> Dict[str, float]:
        """
        计算接近中心性
        
        返回:
            节点到接近中心性的字典
        """
        print(f"  计算接近中心性...")
        
        cc = nx.closeness_centrality(self.G)
        return cc
    
    def eigenvector_centrality(
        self,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> Dict[str, float]:
        """
        计算特征向量中心性
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
            
        返回:
            节点到特征向量中心性的字典
        """
        print(f"  计算特征向量中心性...")
        
        try:
            ec = nx.eigenvector_centrality(
                self.G,
                max_iter=max_iter,
                tol=tol,
                weight='weight'
            )
        except:
            try:
                ec = nx.eigenvector_centrality(self.G, max_iter=max_iter, tol=tol)
            except:
                # 如果图不连通，返回全0
                ec = {node: 0.0 for node in self.G.nodes()}
        
        return ec
    
    def compute_all(self) -> Dict[str, Dict[str, float]]:
        """
        计算所有中心性指标
        
        返回:
            指标名称到{节点:值}字典的字典
        """
        print("\n计算所有基线中心性指标...")
        
        results = {}
        
        results['pagerank'] = self.pagerank()
        results['degree_in'] = self.degree_centrality('in')
        results['degree_out'] = self.degree_centrality('out')
        results['degree_total'] = self.degree_centrality('total')
        results['betweenness'] = self.betweenness_centrality()
        results['closeness'] = self.closeness_centrality()
        results['eigenvector'] = self.eigenvector_centrality()
        
        print("✓ 所有基线指标计算完成\n")
        
        return results
    
    def get_top_k_nodes(
        self,
        centrality_values: Dict[str, float],
        top_k: int = 20
    ) -> List[tuple]:
        """
        获取前k个关键节点
        
        参数:
            centrality_values: 中心性值字典
            top_k: 前k个
            
        返回:
            (节点, 分数)列表
        """
        sorted_nodes = sorted(
            centrality_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:top_k]


if __name__ == "__main__":
    # 测试代码
    import pickle
    
    print("=" * 80)
    print("测试基线中心性算法")
    print("=" * 80)
    
    # 加载网络
    networks_file = "data/processed/final/supply_chain_network_with_emotions.pkl"
    
    with open(networks_file, 'rb') as f:
        networks = pickle.load(f)
    
    period = list(networks.keys())[0]
    G = networks[period]
    
    print(f"\n使用网络: {period}")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}\n")
    
    # 计算所有中心性
    measures = CentralityMeasures(G)
    all_centralities = measures.compute_all()
    
    # 显示各指标的Top 5节点
    for metric_name, values in all_centralities.items():
        top_nodes = measures.get_top_k_nodes(values, top_k=5)
        print(f"{metric_name} Top 5:")
        for i, (node, score) in enumerate(top_nodes, 1):
            print(f"  {i}. {node}: {score:.4f}")
        print()
    
    print("✓ 测试完成!")

