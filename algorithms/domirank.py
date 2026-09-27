"""
经典DomiRank算法实现
参考论文: DomiRank中心性算法
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional, List
from scipy.integrate import odeint


class DomiRank:
    """经典DomiRank算法"""
    
    def __init__(
        self,
        G: nx.DiGraph,
        alpha: float = 1.0,
        beta: float = 0.1,
        theta: float = 1.0
    ):
        """
        初始化DomiRank
        
        参数:
            G: NetworkX有向图
            alpha: 支配增益系数
            beta: 衰减系数
            theta: 支配阈值
        """
        self.G = G
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        
        # 节点列表
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        
    def compute(
        self,
        method: str = 'euler',
        max_iter: int = 1000,
        tol: float = 1e-6,
        time_step: float = 0.01
    ) -> Dict[str, float]:
        """
        计算DomiRank值
        
        参数:
            method: 求解方法 ('euler', 'runge_kutta', 'scipy')
            max_iter: 最大迭代次数
            tol: 收敛容差
            time_step: 时间步长
            
        返回:
            节点到DomiRank值的字典
        """
        if method == 'euler':
            gamma = self._solve_euler(max_iter, tol, time_step)
        elif method == 'runge_kutta':
            gamma = self._solve_runge_kutta(max_iter, tol, time_step)
        elif method == 'scipy':
            gamma = self._solve_scipy()
        else:
            raise ValueError(f"不支持的求解方法: {method}")
        
        # 转换为字典
        return {node: gamma[i] for i, node in enumerate(self.nodes)}
    
    def _dgamma_dt(self, gamma: np.ndarray) -> np.ndarray:
        """
        计算DomiRank的微分方程
        
        dΓ_i/dt = α·Σ_j∈N(i)[θ - Γ_j] - β·Γ_i
        
        参数:
            gamma: 当前Γ值数组
            
        返回:
            dΓ/dt数组
        """
        dgamma = np.zeros(self.n)
        
        for i, node in enumerate(self.nodes):
            # 计算邻居贡献
            neighbor_sum = 0.0
            
            # 入边邻居（指向当前节点的节点）
            for neighbor in self.G.predecessors(node):
                j = self.node_to_idx[neighbor]
                neighbor_sum += (self.theta - gamma[j])
            
            # 计算dΓ_i/dt
            dgamma[i] = self.alpha * neighbor_sum - self.beta * gamma[i]
        
        return dgamma
    
    def _solve_euler(
        self,
        max_iter: int,
        tol: float,
        time_step: float
    ) -> np.ndarray:
        """
        Euler方法求解微分方程
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
            time_step: 时间步长
            
        返回:
            收敛后的Γ值
        """
        # 初始化Γ值
        gamma = np.ones(self.n)
        
        for iteration in range(max_iter):
            # 计算导数
            dgamma = self._dgamma_dt(gamma)
            
            # Euler更新
            gamma_new = gamma + time_step * dgamma
            
            # 检查收敛
            if np.linalg.norm(gamma_new - gamma) < tol:
                print(f"  DomiRank收敛于第 {iteration} 次迭代")
                return gamma_new
            
            gamma = gamma_new
        
        print(f"  警告: DomiRank未完全收敛（{max_iter}次迭代）")
        return gamma
    
    def _solve_runge_kutta(
        self,
        max_iter: int,
        tol: float,
        time_step: float
    ) -> np.ndarray:
        """
        4阶Runge-Kutta方法求解
        
        参数:
            max_iter: 最大迭代次数
            tol: 收敛容差
            time_step: 时间步长
            
        返回:
            收敛后的Γ值
        """
        gamma = np.ones(self.n)
        
        for iteration in range(max_iter):
            # RK4步骤
            k1 = self._dgamma_dt(gamma)
            k2 = self._dgamma_dt(gamma + 0.5 * time_step * k1)
            k3 = self._dgamma_dt(gamma + 0.5 * time_step * k2)
            k4 = self._dgamma_dt(gamma + time_step * k3)
            
            gamma_new = gamma + (time_step / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 检查收敛
            if np.linalg.norm(gamma_new - gamma) < tol:
                print(f"  DomiRank收敛于第 {iteration} 次迭代")
                return gamma_new
            
            gamma = gamma_new
        
        print(f"  警告: DomiRank未完全收敛（{max_iter}次迭代）")
        return gamma
    
    def _solve_scipy(self) -> np.ndarray:
        """
        使用scipy的ODE求解器
        
        返回:
            稳态Γ值
        """
        def f(gamma, t):
            return self._dgamma_dt(gamma)
        
        # 初始值
        gamma0 = np.ones(self.n)
        
        # 时间范围：根据网络规模自适应调整
        # 大网络需要更长的时间达到稳态
        max_time = max(100, self.n * 0.5)  # 至少100，大网络按节点数调整
        n_points = int(max_time * 10)  # 保证足够的时间步
        t = np.linspace(0, max_time, n_points)
        
        # 求解
        solution = odeint(f, gamma0, t)
        
        # 检查是否收敛（最后两个时间点的差异）
        final_diff = np.linalg.norm(solution[-1] - solution[-2])
        if final_diff > 1e-3:
            print(f"  警告: scipy可能未达到稳态 (diff={final_diff:.6f})")
        
        # 返回稳态值
        return solution[-1]
    
    def identify_critical_nodes(
        self,
        domirank_values: Dict[str, float],
        top_k: int = 20
    ) -> List[tuple]:
        """
        识别关键节点
        
        参数:
            domirank_values: DomiRank值字典
            top_k: 前k个节点
            
        返回:
            (节点, 分数)的列表，按分数降序排列
        """
        sorted_nodes = sorted(
            domirank_values.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_nodes[:top_k]


if __name__ == "__main__":
    # 测试代码
    import pickle
    
    print("=" * 80)
    print("测试经典DomiRank算法")
    print("=" * 80)
    
    # 加载网络
    networks_file = "data/processed/final/supply_chain_network_with_emotions.pkl"
    
    with open(networks_file, 'rb') as f:
        networks = pickle.load(f)
    
    # 选择一个时间段的网络
    period = list(networks.keys())[0]
    G = networks[period]
    
    print(f"\n使用网络: {period}")
    print(f"  节点数: {G.number_of_nodes()}")
    print(f"  边数: {G.number_of_edges()}\n")
    
    # 计算DomiRank
    domirank = DomiRank(G, alpha=1.0, beta=0.1, theta=1.0)
    values = domirank.compute(method='euler')
    
    # 识别关键节点
    critical_nodes = domirank.identify_critical_nodes(values, top_k=10)
    
    print("\n关键节点（Top 10）:")
    for i, (node, score) in enumerate(critical_nodes, 1):
        print(f"  {i}. {node}: {score:.4f}")
    
    print("\n✓ 测试完成!")

