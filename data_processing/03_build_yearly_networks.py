"""
构建年度供应链网络（2017-2021）
整合BACI贸易数据和新闻情感数据
"""
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import pickle
import sys
sys.path.append(str(Path(__file__).parent.parent))

from algorithms.domirank import DomiRank
from algorithms.affective_domirank import AffectiveDomiRank

class YearlyNetworkBuilder:
    """年度供应链网络构建器"""
    
    def __init__(self):
        self.trade_dir = Path('data/processed/baci')
        self.sentiment_dir = Path('data/processed/news_sentiment')
        self.output_dir = Path('data/processed/networks')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self.trade_data = self._load_trade_data()
        self.sentiment_data = self._load_sentiment_data()
        
        print(f"✓ 贸易数据: {len(self.trade_data)} 条")
        print(f"✓ 情感数据: {len(self.sentiment_data)} 条")
    
    def _load_trade_data(self):
        """加载BACI贸易数据"""
        trade_file = self.trade_dir / 'baci_trade_2017_2021.csv'
        df = pd.read_csv(trade_file)
        print(f"加载贸易数据: {len(df)} 条")
        return df
    
    def _load_sentiment_data(self):
        """加载新闻情感数据"""
        sentiment_file = self.sentiment_dir / 'news_sentiment_complete.csv'
        df = pd.read_csv(sentiment_file)
        print(f"加载情感数据: {len(df)} 条")
        return df
    
    def build_network_for_year(self, year: int) -> nx.DiGraph:
        """构建某一年的供应链网络"""
        print(f"\n{'='*60}")
        print(f"构建 {year} 年网络...")
        print(f"{'='*60}")
        
        # 过滤该年数据
        trade_year = self.trade_data[self.trade_data['year'] == year]
        sentiment_year = self.sentiment_data[self.sentiment_data['year'] == year]
        
        print(f"  贸易记录: {len(trade_year)}")
        print(f"  情感记录: {len(sentiment_year)}")
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点（国家-行业）
        nodes = set()
        for _, row in trade_year.iterrows():
            exp_node = f"{row['exporter']}_{row['industry']}"
            imp_node = f"{row['importer']}_{row['industry']}"
            nodes.add(exp_node)
            nodes.add(imp_node)
        
        for node in nodes:
            country, industry = node.split('_', 1)
            G.add_node(
                node,
                country=country,
                industry=industry,
                year=year
            )
        
        print(f"  节点数: {G.number_of_nodes()}")
        
        # 添加边（贸易流）
        for _, row in trade_year.iterrows():
            exp_node = f"{row['exporter']}_{row['industry']}"
            imp_node = f"{row['importer']}_{row['industry']}"
            
            if exp_node != imp_node:
                G.add_edge(
                    exp_node,
                    imp_node,
                    weight=row['trade_value'],  # 千美元
                    quantity=row['trade_quantity']
                )
        
        print(f"  边数: {G.number_of_edges()}")
        
        # 归一化边权重
        self._normalize_edge_weights(G)
        
        # 添加情感数据到节点
        for node in G.nodes():
            country, industry = node.split('_', 1)
            sentiment_row = sentiment_year[
                (sentiment_year['country'] == country) & 
                (sentiment_year['industry'] == industry)
            ]
            
            if len(sentiment_row) > 0:
                sent_mean = sentiment_row.iloc[0]['sentiment_mean']
                news_cnt = sentiment_row.iloc[0]['news_count']
                sent_std = sentiment_row.iloc[0]['sentiment_std']
                
                G.nodes[node]['sentiment_mean'] = sent_mean
                G.nodes[node]['news_count'] = news_cnt
                G.nodes[node]['sentiment_std'] = sent_std
                # 只有实际情感值显著且有足够新闻数才标记为有情感
                G.nodes[node]['has_sentiment'] = (abs(sent_mean) > 0.01 and news_cnt >= 3)
            else:
                G.nodes[node]['sentiment_mean'] = 0.0
                G.nodes[node]['news_count'] = 0
                G.nodes[node]['sentiment_std'] = 0.0
                G.nodes[node]['has_sentiment'] = False
        
        sentiment_nodes = sum(1 for n in G.nodes() if G.nodes[n]['has_sentiment'])
        print(f"  有情感数据的节点: {sentiment_nodes}/{G.number_of_nodes()} ({sentiment_nodes/G.number_of_nodes()*100:.1f}%)")
        
        # 填充缺失的情感数据（完美方案）
        self._impute_missing_sentiment(G, year)
        
        # 计算网络中心性
        self._compute_centralities(G)
        
        # 计算DomiRank
        self._compute_domirank(G)
        
        # 计算邻居聚合特征（方案2：供应链风险传播）
        self._compute_neighbor_features(G)
        
        return G
    
    def _impute_missing_sentiment(self, G: nx.DiGraph, year: int):
        """
        完美的情感数据填充方案（多策略组合）
        
        策略优先级：
        1. 图结构传播：从供应链邻居节点传播情感（加权平均）
        2. 统计填充：同国家或同行业的平均值
        3. 全局填充：全局平均 + 随机扰动
        """
        print(f"  填充缺失情感数据...")
        
        from collections import defaultdict
        
        # 统计原始情感覆盖率
        original_sentiment_count = sum(1 for n in G.nodes() if abs(G.nodes[n]['sentiment_mean']) > 0.01)
        
        # 第一轮：统计有情感节点的分布
        country_sentiments = defaultdict(list)
        industry_sentiments = defaultdict(list)
        all_sentiments = []
        
        for node in G.nodes():
            sent = G.nodes[node]['sentiment_mean']
            if abs(sent) > 0.01:
                country, industry = node.split('_', 1)
                country_sentiments[country].append(sent)
                industry_sentiments[industry].append(sent)
                all_sentiments.append(sent)
        
        global_avg = np.mean(all_sentiments) if all_sentiments else 0.0
        global_std = np.std(all_sentiments) if all_sentiments else 0.1
        
        # 第二轮：多策略填充
        filled_strategies = {'graph': 0, 'country': 0, 'industry': 0, 'global': 0}
        
        for node in G.nodes():
            if abs(G.nodes[node]['sentiment_mean']) > 0.01:
                continue  # 已有情感，跳过
            
            country, industry = node.split('_', 1)
            filled_value = None
            strategy = None
            
            # 策略1：图结构传播（最优）
            # 从供应链邻居（上游+下游）传播情感
            neighbors = list(G.predecessors(node)) + list(G.successors(node))
            neighbor_sentiments = []
            neighbor_weights = []
            
            for neighbor in neighbors:
                neighbor_sent = G.nodes[neighbor]['sentiment_mean']
                if abs(neighbor_sent) > 0.01:
                    neighbor_sentiments.append(neighbor_sent)
                    # 权重：边的贸易额（归一化）
                    if G.has_edge(neighbor, node):
                        weight = G[neighbor][node].get('normalized_weight', 0.1)
                    elif G.has_edge(node, neighbor):
                        weight = G[node][neighbor].get('normalized_weight', 0.1)
                    else:
                        weight = 0.1
                    neighbor_weights.append(weight)
            
            if neighbor_sentiments:
                # 加权平均邻居情感
                total_weight = sum(neighbor_weights)
                filled_value = sum(s * w for s, w in zip(neighbor_sentiments, neighbor_weights)) / total_weight
                strategy = 'graph'
            
            # 策略2：国家统计（次优）
            if filled_value is None and country in country_sentiments:
                filled_value = np.mean(country_sentiments[country])
                # 添加噪声，避免完全相同
                filled_value += np.random.normal(0, global_std * 0.1)
                strategy = 'country'
            
            # 策略3：行业统计
            if filled_value is None and industry in industry_sentiments:
                filled_value = np.mean(industry_sentiments[industry])
                filled_value += np.random.normal(0, global_std * 0.1)
                strategy = 'industry'
            
            # 策略4：全局统计（保底）
            if filled_value is None:
                # 全局平均 + 较大随机扰动（保持方差）
                filled_value = global_avg + np.random.normal(0, global_std * 0.3)
                strategy = 'global'
            
            # 应用填充值
            G.nodes[node]['sentiment_mean'] = filled_value
            G.nodes[node]['sentiment_filled'] = True
            G.nodes[node]['fill_strategy'] = strategy
            filled_strategies[strategy] += 1
        
        # 统计填充效果
        final_sentiment_count = sum(1 for n in G.nodes() if abs(G.nodes[n]['sentiment_mean']) > 0.01)
        total_filled = sum(filled_strategies.values())
        
        print(f"    原始有情感: {original_sentiment_count}/{G.number_of_nodes()} ({original_sentiment_count/G.number_of_nodes()*100:.1f}%)")
        print(f"    填充后有情感: {final_sentiment_count}/{G.number_of_nodes()} ({final_sentiment_count/G.number_of_nodes()*100:.1f}%)")
        print(f"    填充策略分布:")
        print(f"      图传播: {filled_strategies['graph']} ({filled_strategies['graph']/total_filled*100 if total_filled>0 else 0:.1f}%)")
        print(f"      国家统计: {filled_strategies['country']} ({filled_strategies['country']/total_filled*100 if total_filled>0 else 0:.1f}%)")
        print(f"      行业统计: {filled_strategies['industry']} ({filled_strategies['industry']/total_filled*100 if total_filled>0 else 0:.1f}%)")
        print(f"      全局统计: {filled_strategies['global']} ({filled_strategies['global']/total_filled*100 if total_filled>0 else 0:.1f}%)")
        
        # 验证填充质量
        all_sentiments_after = [G.nodes[n]['sentiment_mean'] for n in G.nodes()]
        print(f"    填充后情感分布: 均值={np.mean(all_sentiments_after):.4f}, 标准差={np.std(all_sentiments_after):.4f}")
    
    def _normalize_edge_weights(self, G: nx.DiGraph):
        """归一化边权重到[0, 1]"""
        weights = [data['weight'] for _, _, data in G.edges(data=True)]
        if weights:
            min_w = min(weights)
            max_w = max(weights)
            if max_w > min_w:
                for u, v, data in G.edges(data=True):
                    data['normalized_weight'] = (data['weight'] - min_w) / (max_w - min_w)
            else:
                for u, v, data in G.edges(data=True):
                    data['normalized_weight'] = 1.0
    
    def _compute_centralities(self, G: nx.DiGraph):
        """计算各种中心性指标"""
        # 度中心性（无权版本：连接数）
        # 使用连接数而非贸易额，避免数值过大
        in_degree = dict(G.in_degree())  # 无权度数
        out_degree = dict(G.out_degree())  # 无权度数
        
        for node in G.nodes():
            G.nodes[node]['in_degree'] = in_degree.get(node, 0)
            G.nodes[node]['out_degree'] = out_degree.get(node, 0)
            G.nodes[node]['total_degree'] = in_degree.get(node, 0) + out_degree.get(node, 0)
        
        # PageRank
        try:
            pagerank = nx.pagerank(G, weight='weight', max_iter=100)
            for node, pr in pagerank.items():
                G.nodes[node]['pagerank'] = pr
        except:
            for node in G.nodes():
                G.nodes[node]['pagerank'] = 1.0 / G.number_of_nodes()
    
    def _compute_domirank(self, G: nx.DiGraph):
        """计算DomiRank（作为标签）"""
        print(f"  计算DomiRank...")
        
        # 传统DomiRank（使用稳定参数，作为affective_domirank的基础）
        domirank = DomiRank(G, alpha=0.01, beta=0.5)  # 保守参数，保证数值稳定
        try:
            scores = domirank.compute(method='scipy')  # scipy求解器
        except:
            print("    警告: scipy求解失败，使用euler方法")
            scores = domirank.compute(max_iter=5000, tol=1e-6)
        
        for node, score in scores.items():
            G.nodes[node]['domirank'] = score
        
        print(f"    DomiRank范围: [{min(scores.values()):.4f}, {max(scores.values()):.4f}]")
        
        # Affective DomiRank（如果有情感数据）
        sentiment_nodes = [n for n in G.nodes() if G.nodes[n]['has_sentiment']]
        if len(sentiment_nodes) > 10:
            print(f"  计算Affective DomiRank...")
            
            # 构建情感向量
            emotion_vectors = {}
            for node in G.nodes():
                sentiment_mean = G.nodes[node]['sentiment_mean']
                emotion_vectors[node] = {
                    'sentiment_polarity': sentiment_mean,
                    'sentiment_intensity': abs(sentiment_mean),
                    'fear': max(-sentiment_mean, 0),  # 负面情感→恐惧
                    'trust': max(sentiment_mean, 0),  # 正面情感→信任
                    'anger': 0,
                    'joy': 0,
                    'sadness': 0
                }
            
            # 使用加性情感调制机制
            affective_domirank = AffectiveDomiRank(
                G,
                emotion_vectors,
                alpha=0.01,
                beta=0.5,
                gamma=0.1  # 情感调制强度
            )
            try:
                affective_scores = affective_domirank.compute(method='scipy')  # scipy求解器
            except:
                print("    警告: scipy求解失败，使用euler方法")
                affective_scores = affective_domirank.compute(max_iter=5000, tol=1e-6)
            
            for node, score in affective_scores.items():
                G.nodes[node]['affective_domirank'] = score
            
            print(f"    Affective DomiRank范围: [{min(affective_scores.values()):.4f}, {max(affective_scores.values()):.4f}]")
    
    def _compute_neighbor_features(self, G: nx.DiGraph):
        """
        计算邻居聚合特征（方案2：供应链风险传播）
        
        特征：
        1. neighbor_avg_sentiment: 上游供应商平均情感
        2. neighbor_avg_domirank: 上游供应商平均风险
        3. supplier_concentration: 供应商集中度（入度熵）
        """
        print(f"  计算邻居聚合特征...")
        
        import numpy as np
        
        for node in G.nodes():
            # 1. 上游邻居平均情感（predecessors = 供应商）
            predecessors = list(G.predecessors(node))
            if predecessors:
                sentiments = [G.nodes[p].get('sentiment_mean', 0) for p in predecessors]
                G.nodes[node]['neighbor_avg_sentiment'] = np.mean(sentiments)
                
                # 2. 上游邻居平均风险（Affective DomiRank）
                domiranks = [G.nodes[p].get('affective_domirank', 0) for p in predecessors]
                G.nodes[node]['neighbor_avg_risk'] = np.mean(domiranks)
                
                # 3. 供应商集中度：入边权重的熵
                in_weights = [G[p][node]['weight'] for p in predecessors]
                total_weight = sum(in_weights)
                if total_weight > 0:
                    probs = [w / total_weight for w in in_weights]
                    entropy = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)
                    # 归一化熵：除以最大熵log(n)
                    max_entropy = np.log(len(predecessors)) if len(predecessors) > 1 else 1
                    G.nodes[node]['supplier_concentration'] = 1 - (entropy / max_entropy)  # 1-熵=集中度
                else:
                    G.nodes[node]['supplier_concentration'] = 0
            else:
                G.nodes[node]['neighbor_avg_sentiment'] = 0
                G.nodes[node]['neighbor_avg_risk'] = 0
                G.nodes[node]['supplier_concentration'] = 0
        
        print(f"    ✓ 已计算邻居特征")
    
    def _temporal_sentiment_smoothing(self, networks: dict):
        """
        跨年份时间序列平滑
        
        对仍然缺失情感的节点，从相邻年份传播情感
        对已有情感的节点，进行时间序列平滑（减少年度波动）
        """
        years = sorted(networks.keys())
        smooth_count = 0
        
        for node in networks[years[0]].nodes():
            # 收集该节点在所有年份的情感
            sentiments_by_year = {}
            for year in years:
                if node in networks[year].nodes():
                    sent = networks[year].nodes[node]['sentiment_mean']
                    if abs(sent) > 0.01:
                        sentiments_by_year[year] = sent
            
            # 如果有至少1年的数据，进行插值和平滑
            if sentiments_by_year:
                for year in years:
                    if node not in networks[year].nodes():
                        continue
                    
                    current_sent = networks[year].nodes[node]['sentiment_mean']
                    
                    # 如果当前年缺失，用相邻年份插值
                    if abs(current_sent) < 0.01:
                        # 找最近的前后年份
                        prev_sent = None
                        next_sent = None
                        
                        for y in reversed(years):
                            if y < year and y in sentiments_by_year:
                                prev_sent = sentiments_by_year[y]
                                break
                        
                        for y in years:
                            if y > year and y in sentiments_by_year:
                                next_sent = sentiments_by_year[y]
                                break
                        
                        # 插值
                        if prev_sent is not None and next_sent is not None:
                            # 线性插值
                            networks[year].nodes[node]['sentiment_mean'] = (prev_sent + next_sent) / 2.0
                            networks[year].nodes[node]['sentiment_filled'] = True
                            networks[year].nodes[node]['fill_strategy'] = 'temporal_interp'
                            smooth_count += 1
                        elif prev_sent is not None:
                            # 只有历史值，衰减使用
                            networks[year].nodes[node]['sentiment_mean'] = prev_sent * 0.8
                            networks[year].nodes[node]['sentiment_filled'] = True
                            networks[year].nodes[node]['fill_strategy'] = 'temporal_decay'
                            smooth_count += 1
                        elif next_sent is not None:
                            # 只有未来值，衰减使用
                            networks[year].nodes[node]['sentiment_mean'] = next_sent * 0.8
                            networks[year].nodes[node]['sentiment_filled'] = True
                            networks[year].nodes[node]['fill_strategy'] = 'temporal_forward'
                            smooth_count += 1
        
        print(f"  时间序列平滑填充: {smooth_count} 个节点-年份对")
        
        # 统计最终覆盖率
        total_node_years = sum(G.number_of_nodes() for G in networks.values())
        final_sentiment_count = sum(
            sum(1 for n in G.nodes() if abs(G.nodes[n]['sentiment_mean']) > 0.01)
            for G in networks.values()
        )
        print(f"  最终情感覆盖率: {final_sentiment_count}/{total_node_years} ({final_sentiment_count/total_node_years*100:.1f}%)")
    
    def build_all_years(self):
        """构建所有年份的网络"""
        print("\n" + "="*80)
        print("构建年度供应链网络: 2017-2021")
        print("="*80)
        
        # 设置随机种子，保证可复现
        np.random.seed(42)
        
        networks = {}
        
        for year in range(2017, 2022):
            G = self.build_network_for_year(year)
            networks[year] = G
        
        # 跨年份时间序列平滑（第二轮填充）
        print(f"\n跨年份时间序列平滑...")
        self._temporal_sentiment_smoothing(networks)
        
        # 保存所有网络
        for year, G in networks.items():
            # 保存网络
            output_file = self.output_dir / f'network_{year}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(G, f)
            print(f"  ✓ 已保存: {output_file}")
            
            # 也保存为GraphML（可视化用）
            # 先转换numpy类型为Python原生类型
            for node in G.nodes():
                for key, value in list(G.nodes[node].items()):
                    if isinstance(value, (np.bool_, np.generic)):
                        G.nodes[node][key] = value.item()
            
            graphml_file = self.output_dir / f'network_{year}.graphml'
            try:
                nx.write_graphml(G, graphml_file)
            except Exception as e:
                print(f"  ⚠️ GraphML保存失败: {e}")
        
        # 保存所有网络
        all_networks_file = self.output_dir / 'all_networks_2017_2021.pkl'
        with open(all_networks_file, 'wb') as f:
            pickle.dump(networks, f)
        print(f"\n✓ 所有网络已保存: {all_networks_file}")
        
        # 统计
        print(f"\n" + "="*80)
        print("网络统计:")
        for year, G in sorted(networks.items()):
            print(f"  {year}年:")
            print(f"    节点: {G.number_of_nodes()}")
            print(f"    边: {G.number_of_edges()}")
            print(f"    密度: {nx.density(G):.4f}")
            
            sentiment_nodes = sum(1 for n in G.nodes() if G.nodes[n]['has_sentiment'])
            print(f"    有情感节点: {sentiment_nodes} ({sentiment_nodes/G.number_of_nodes()*100:.1f}%)")
        
        # 总体统计
        total_nodes = sum(G.number_of_nodes() for G in networks.values())
        total_edges = sum(G.number_of_edges() for G in networks.values())
        
        print(f"\n总体统计:")
        print(f"  年份数: {len(networks)}")
        print(f"  总节点实例: {total_nodes} (平均 {total_nodes/len(networks):.0f}/年)")
        print(f"  总边实例: {total_edges} (平均 {total_edges/len(networks):.0f}/年)")
        print(f"  潜在训练样本: {total_nodes} (每个节点=1个样本)")
        
        print("\n✓ 年度网络构建完成！")


def main():
    """主函数"""
    builder = YearlyNetworkBuilder()
    
    print("\n是否开始构建？")
    choice = input("继续？[y/N]: ").strip().lower()
    
    if choice == 'y':
        builder.build_all_years()
    else:
        print("已取消")


if __name__ == "__main__":
    main()

