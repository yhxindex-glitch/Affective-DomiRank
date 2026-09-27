"""
构建供应链风险事件标签

基于现有数据的多个维度构建风险得分：
1. 贸易量下降幅度
2. 供应商流失率
3. 负面情感强度
4. 度中心性下降
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def load_networks():
    """加载网络数据"""
    network_file = Path('data/processed/networks/all_networks_2017_2021.pkl')
    with open(network_file, 'rb') as f:
        networks = pickle.load(f)
    return networks

def compute_risk_components(networks):
    """计算风险得分的各个组成部分"""
    years = sorted(networks.keys())
    
    risk_data = []
    
    for i in range(len(years) - 1):
        year_curr = years[i]
        year_next = years[i + 1]
        
        G_curr = networks[year_curr]
        G_next = networks[year_next]
        
        print(f"\n处理 {year_curr} → {year_next}...")
        
        for node in G_curr.nodes():
            if node not in G_next.nodes():
                continue
            
            # 1. 贸易量变化（已有的标签）
            trade_last = G_curr.nodes[node].get('trade_volume', 0)
            trade_target = G_next.nodes[node].get('trade_volume', 0)
            
            if trade_last > 0:
                trade_change_rate = (trade_target - trade_last) / trade_last
            else:
                trade_change_rate = 0.0
            
            # 贸易下降风险：只关注下降，归一化到[0,1]
            trade_decline_risk = max(0, -trade_change_rate)  # 下降越多风险越高
            trade_decline_risk = min(trade_decline_risk, 1.0)  # 截断到1.0
            
            # 2. 供应商流失率
            suppliers_curr = set(G_curr.predecessors(node))
            suppliers_next = set(G_next.predecessors(node))
            
            if len(suppliers_curr) > 0:
                lost_suppliers = suppliers_curr - suppliers_next
                supplier_loss_rate = len(lost_suppliers) / len(suppliers_curr)
            else:
                supplier_loss_rate = 0.0
            
            # 3. 负面情感强度
            sentiment_curr = G_curr.nodes[node].get('sentiment_mean', 0.0)
            sentiment_next = G_next.nodes[node].get('sentiment_mean', 0.0)
            
            # 负面情感风险：取负值并归一化
            negative_sentiment_risk = max(0, -sentiment_next)  # 负面情感越强风险越高
            
            # 情感恶化：如果情感变得更负面
            sentiment_deterioration = max(0, sentiment_curr - sentiment_next)
            sentiment_deterioration = min(sentiment_deterioration, 1.0)
            
            # 4. 度中心性下降
            in_degree_curr = G_curr.in_degree(node)
            in_degree_next = G_next.in_degree(node)
            out_degree_curr = G_curr.out_degree(node)
            out_degree_next = G_next.out_degree(node)
            
            total_degree_curr = in_degree_curr + out_degree_curr
            total_degree_next = in_degree_next + out_degree_next
            
            if total_degree_curr > 0:
                degree_decline_rate = max(0, (total_degree_curr - total_degree_next) / total_degree_curr)
            else:
                degree_decline_rate = 0.0
            
            # 5. Affective DomiRank变化（作为特征，不作为风险组件）
            adr_curr = G_curr.nodes[node].get('affective_domirank', 0.5)
            adr_next = G_next.nodes[node].get('affective_domirank', 0.5)
            adr_increase = adr_next - adr_curr  # ADR上升可能表示风险传播增强
            
            # 构建综合风险得分（方案B：客观为主，情感为辅）
            # 核心理念：风险 = 供应链实际恶化程度（客观）+ 情感预警信号（主观）
            comprehensive_risk = (
                0.40 * supplier_loss_rate +         # 供应商流失（最直接的风险）
                0.25 * degree_decline_rate +        # 网络连接退化
                0.20 * negative_sentiment_risk +    # 负面情感（预警信号）
                0.15 * sentiment_deterioration +    # 情感恶化趋势
                0.00 * trade_decline_risk           # 失效（全增长期）
            )
            # 新权重分配：客观70%，情感35%（但贸易失效，实际为客观65%，情感35%）
            
            # 构建贸易中断二分类标签（方案1）
            trade_disruption = 1 if trade_change_rate < -0.3 else 0
            
            risk_data.append({
                'year_from': year_curr,
                'year_to': year_next,
                'node': node,
                # 风险组件
                'trade_decline_risk': trade_decline_risk,
                'supplier_loss_rate': supplier_loss_rate,
                'negative_sentiment_risk': negative_sentiment_risk,
                'sentiment_deterioration': sentiment_deterioration,
                'degree_decline_rate': degree_decline_rate,
                # 综合风险得分
                'comprehensive_risk_score': comprehensive_risk,
                # 二分类标签
                'trade_disruption': trade_disruption,
                # 原始值（用于分析）
                'trade_change_rate': trade_change_rate,
                'sentiment_curr': sentiment_curr,
                'sentiment_next': sentiment_next,
                'adr_curr': adr_curr,
                'adr_next': adr_next,
                'adr_increase': adr_increase,
            })
    
    return pd.DataFrame(risk_data)

def analyze_risk_distribution(df):
    """分析风险标签分布"""
    print("\n" + "="*80)
    print("风险标签分布分析")
    print("="*80)
    
    print(f"\n总样本数: {len(df)}")
    
    print("\n【综合风险得分】（0-1连续值）")
    print(f"  均值: {df['comprehensive_risk_score'].mean():.4f}")
    print(f"  标准差: {df['comprehensive_risk_score'].std():.4f}")
    print(f"  最小值: {df['comprehensive_risk_score'].min():.4f}")
    print(f"  最大值: {df['comprehensive_risk_score'].max():.4f}")
    print(f"  中位数: {df['comprehensive_risk_score'].median():.4f}")
    
    print("\n  风险等级分布:")
    print(f"    低风险 (0-0.2): {(df['comprehensive_risk_score'] < 0.2).sum()} ({(df['comprehensive_risk_score'] < 0.2).mean()*100:.1f}%)")
    print(f"    中风险 (0.2-0.5): {((df['comprehensive_risk_score'] >= 0.2) & (df['comprehensive_risk_score'] < 0.5)).sum()} ({((df['comprehensive_risk_score'] >= 0.2) & (df['comprehensive_risk_score'] < 0.5)).mean()*100:.1f}%)")
    print(f"    高风险 (0.5+): {(df['comprehensive_risk_score'] >= 0.5).sum()} ({(df['comprehensive_risk_score'] >= 0.5).mean()*100:.1f}%)")
    
    print("\n【贸易中断事件】（二分类）")
    print(f"  中断事件 (1): {df['trade_disruption'].sum()} ({df['trade_disruption'].mean()*100:.1f}%)")
    print(f"  正常 (0): {(df['trade_disruption'] == 0).sum()} ({(df['trade_disruption'] == 0).mean()*100:.1f}%)")
    
    print("\n【风险组件统计】")
    for col in ['trade_decline_risk', 'supplier_loss_rate', 'negative_sentiment_risk', 
                'sentiment_deterioration', 'degree_decline_rate']:
        print(f"  {col}: 均值={df[col].mean():.4f}, 非零比例={((df[col] > 0).mean()*100):.1f}%")
    
    # 绘制分布图
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 综合风险得分分布
    axes[0, 0].hist(df['comprehensive_risk_score'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(df['comprehensive_risk_score'].mean(), color='r', linestyle='--', label='均值')
    axes[0, 0].set_xlabel('综合风险得分')
    axes[0, 0].set_ylabel('频数')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 各风险组件分布
    components = [
        ('trade_decline_risk', '贸易下降风险'),
        ('supplier_loss_rate', '供应商流失率'),
        ('negative_sentiment_risk', '负面情感风险'),
        ('sentiment_deterioration', '情感恶化'),
        ('degree_decline_rate', '度中心性下降')
    ]
    
    for idx, (col, title) in enumerate(components):
        ax = axes[(idx+1)//3, (idx+1)%3]
        ax.hist(df[col], bins=30, alpha=0.7, edgecolor='black', color='orange')
        ax.axvline(df[col].mean(), color='r', linestyle='--', label='均值')
        ax.set_xlabel(title)
        ax.set_ylabel('频数')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('risk_labels_distribution.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ 已保存分布图: risk_labels_distribution.png")

def save_risk_labels(df):
    """保存风险标签"""
    output_file = Path('data/processed/networks/risk_labels.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(df, f)
    print(f"\n✓ 已保存风险标签: {output_file}")
    
    # 也保存为CSV方便查看
    csv_file = Path('data/processed/networks/risk_labels.csv')
    df.to_csv(csv_file, index=False)
    print(f"✓ 已保存CSV: {csv_file}")

def main():
    print("="*80)
    print("构建供应链风险事件标签")
    print("="*80)
    
    # 加载网络
    networks = load_networks()
    print(f"\n加载了 {len(networks)} 年的网络数据")
    
    # 计算风险组件
    risk_df = compute_risk_components(networks)
    
    # 分析分布
    analyze_risk_distribution(risk_df)
    
    # 保存
    save_risk_labels(risk_df)
    
    print("\n" + "="*80)
    print("✓ 风险标签构建完成！")
    print("="*80)
    print("\n推荐使用：comprehensive_risk_score 作为新的预测目标")
    print("  - 连续值（0-1），适合回归任务")
    print("  - 融合多维度风险信息")
    print("  - 更适合Affective DomiRank的情感风险捕捉能力")

if __name__ == '__main__':
    main()

