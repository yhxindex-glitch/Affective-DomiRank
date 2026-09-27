"""
顶会水平论文图表制作脚本 (中英文双版本)
绘制供应链网络脆弱性分析的核心图表

包含：
1. 供应链网络拓扑图
2. 风险标签分布图  
3. 情感特征分布图

设计标准：顶会级别，专业配色，清晰排版
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from figure_export_utils import save_figure_png_and_pdf

# 顶会配色方案
COLORS = {
    'primary': '#2E86AB',      # 深蓝
    'secondary': '#A23B72',     # 深紫
    'accent': '#F18F01',        # 橙色
    'success': '#C73E1D',       # 红色
    'neutral': '#6C757D',       # 灰色
    'light': '#F8F9FA',         # 浅灰
    'dark': '#212529'           # 深灰
}

# 专业调色板
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D', '#28A745', '#FFC107', '#17A2B8']

def setup_english_style():
    """设置英文绑图风格"""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.unicode_minus': True
    })

def setup_chinese_style():
    """设置中文绑图风格"""
    plt.rcParams.update({
        'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'STSong', 'SimSun', 'Arial Unicode MS'],
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.linewidth': 1.2,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        'legend.frameon': False,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.unicode_minus': False
    })

def load_data():
    """加载数据"""
    print("Loading data...")
    
    # 加载网络数据
    with open('data/processed/networks/all_networks_2017_2021.pkl', 'rb') as f:
        networks = pickle.load(f)
    
    # 加载风险标签 - 检查数据结构
    with open('data/processed/networks/risk_labels.pkl', 'rb') as f:
        risk_data = pickle.load(f)
    
    print(f"Risk data type: {type(risk_data)}")
    if isinstance(risk_data, dict):
        print(f"Risk data keys sample: {list(risk_data.keys())[:5]}")
        print(f"Risk data values sample: {list(risk_data.values())[:5]}")
        risk_labels = risk_data
    elif isinstance(risk_data, pd.DataFrame):
        print(f"Risk DataFrame columns: {risk_data.columns.tolist()}")
        print(f"Risk DataFrame shape: {risk_data.shape}")
        # 如果是DataFrame，转换为字典格式
        risk_labels = {}
        for idx, row in risk_data.iterrows():
            node = row.get('node', f'node_{idx}')
            risk = row.get('comprehensive_risk_score', 0)  # 使用综合风险分数
            risk_labels[node] = risk
    else:
        print(f"Unexpected risk data type: {type(risk_data)}")
        risk_labels = {}
    
    # 加载情感数据
    sentiment_df = pd.read_csv('data/processed/news_sentiment/news_sentiment_aggregated.csv')
    
    return networks, risk_labels, sentiment_df

def create_network_topology_figure(networks, risk_labels, lang='en'):
    """
    创建供应链网络拓扑图
    顶会级别设计：节点大小=贸易量，颜色=风险等级，布局=力导向
    """
    is_cn = (lang == 'cn')
    
    if is_cn:
        setup_chinese_style()
        print("Creating network topology figure (CN)...")
    else:
        setup_english_style()
        print("Creating network topology figure...")
    
    # 选择2020年网络（疫情年，更有代表性）
    G = networks[2020]
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # 计算布局（使用力导向算法）
    pos = nx.spring_layout(G, k=3, iterations=100, seed=42)
    
    # 准备节点属性
    node_sizes = []
    node_colors = []
    node_names = []
    
    for node in G.nodes():
        # 节点大小基于总贸易量
        total_trade = G.nodes[node].get('total_degree', 0)
        node_sizes.append(max(50, min(500, total_trade * 0.1)))
        
        # 节点颜色基于风险等级
        risk = risk_labels.get(node, 0)
        try:
            risk_value = float(risk)
        except (ValueError, TypeError):
            risk_value = 0.0
        node_colors.append(risk_value)
        
        # 节点名称（只显示主要国家）
        if total_trade > 1000:  # 只显示贸易量大的国家
            node_names.append(node)
        else:
            node_names.append('')
    
    # 绘制边（透明度较低，避免过于复杂）
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.5, 
                          edge_color=COLORS['neutral'], ax=ax)
    
    # 绘制节点
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_size=node_sizes,
                                  node_color=node_colors,
                                  cmap='RdYlBu_r',  # 红-黄-蓝，红色=高风险
                                  alpha=0.8,
                                  ax=ax)
    
    # 添加节点标签（只显示主要国家）
    labels = {node: name for node, name in zip(G.nodes(), node_names) if name}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold', ax=ax)
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', 
                              norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
    
    if is_cn:
        cbar.set_label('风险等级', fontsize=12, fontweight='bold')
        legend_elements = [
            plt.scatter([], [], s=100, c=COLORS['primary'], alpha=0.8, label='小贸易量'),
            plt.scatter([], [], s=300, c=COLORS['primary'], alpha=0.8, label='大贸易量'),
            plt.scatter([], [], s=200, c='red', alpha=0.8, label='高风险'),
            plt.scatter([], [], s=200, c='blue', alpha=0.8, label='低风险')
        ]
        save_path = 'figures_cn/network_topology.png'
    else:
        cbar.set_label('Risk Level', fontsize=12, fontweight='bold')
        legend_elements = [
            plt.scatter([], [], s=100, c=COLORS['primary'], alpha=0.8, label='Small Trade Volume'),
            plt.scatter([], [], s=300, c=COLORS['primary'], alpha=0.8, label='Large Trade Volume'),
            plt.scatter([], [], s=200, c='red', alpha=0.8, label='High Risk'),
            plt.scatter([], [], s=200, c='blue', alpha=0.8, label='Low Risk')
        ]
        save_path = 'figures/network_topology.png'
    
    cbar.ax.tick_params(labelsize=10)
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    # 移除坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 保存图形
    plt.tight_layout()
    save_figure_png_and_pdf(save_path, dpi_png=300, dpi_pdf=400, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Network topology figure saved: {save_path}")

def create_risk_distribution_figure(risk_labels, lang='en'):
    """
    创建风险值分布直方图
    顶会级别设计：单一精美图表展示风险分布
    """
    is_cn = (lang == 'cn')
    
    if is_cn:
        setup_chinese_style()
        print("Creating risk distribution figure (CN)...")
    else:
        setup_english_style()
        print("Creating risk distribution figure...")
    
    # 转换为DataFrame - 修复数据类型问题
    risk_data = []
    for country, risk in risk_labels.items():
        # 确保risk是数值类型
        try:
            risk_value = float(risk)
            risk_data.append({'Country': country, 'Risk': risk_value})
        except (ValueError, TypeError):
            continue
    
    df = pd.DataFrame(risk_data)
    
    # 确保Risk列是数值类型
    df['Risk'] = pd.to_numeric(df['Risk'], errors='coerce')
    df = df.dropna(subset=['Risk'])  # 删除无效值
    
    # 创建单一图表
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # 绘制直方图
    n, bins, patches = ax.hist(df['Risk'], bins=40, alpha=0.7, 
                              color=COLORS['primary'], edgecolor='white', linewidth=1.2)
    
    # 添加颜色渐变效果（基于风险等级）
    for i, (patch, freq) in enumerate(zip(patches, n)):
        alpha = 0.4 + 0.6 * (freq / n.max())
        patch.set_alpha(alpha)
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < 0.05:
            patch.set_facecolor(COLORS['primary'])
        elif bin_center < 0.15:
            patch.set_facecolor(COLORS['accent'])
        else:
            patch.set_facecolor(COLORS['success'])
    
    # 添加统计信息
    mean_risk = df['Risk'].mean()
    std_risk = df['Risk'].std()
    median_risk = df['Risk'].median()
    
    low_count = len(df[df['Risk'] < 0.05])
    med_count = len(df[(df['Risk'] >= 0.05) & (df['Risk'] < 0.15)])
    high_count = len(df[df['Risk'] >= 0.15])
    
    if is_cn:
        ax.axvline(mean_risk, color=COLORS['accent'], linestyle='-', linewidth=3, 
                   label=f'均值: {mean_risk:.3f}')
        ax.axvline(median_risk, color=COLORS['dark'], linestyle='--', linewidth=2, 
                   label=f'中位数: {median_risk:.3f}')
        ax.axvspan(0, 0.05, alpha=0.1, color=COLORS['primary'], label='低风险')
        ax.axvspan(0.05, 0.15, alpha=0.1, color=COLORS['accent'], label='中风险')
        ax.axvspan(0.15, df['Risk'].max(), alpha=0.1, color=COLORS['success'], label='高风险')
        
        stats_text = f"""
    风险统计:
    • 均值: {mean_risk:.3f}
    • 标准差: {std_risk:.3f}
    • 中位数: {median_risk:.3f}
    • 范围: [{df['Risk'].min():.3f}, {df['Risk'].max():.3f}]
    • 样本数: {len(df):,}
    
    风险分类:
    • 低风险 (<0.05): {low_count:,} ({low_count/len(df)*100:.1f}%)
    • 中风险 (0.05-0.15): {med_count:,} ({med_count/len(df)*100:.1f}%)
    • 高风险 (>0.15): {high_count:,} ({high_count/len(df)*100:.1f}%)
        """
        ax.set_xlabel('风险等级', fontsize=14, fontweight='bold')
        ax.set_ylabel('频率', fontsize=14, fontweight='bold')
        save_path = 'figures_cn/risk_distribution.png'
    else:
        ax.axvline(mean_risk, color=COLORS['accent'], linestyle='-', linewidth=3, 
                   label=f'Mean: {mean_risk:.3f}')
        ax.axvline(median_risk, color=COLORS['dark'], linestyle='--', linewidth=2, 
                   label=f'Median: {median_risk:.3f}')
        ax.axvspan(0, 0.05, alpha=0.1, color=COLORS['primary'], label='Low Risk')
        ax.axvspan(0.05, 0.15, alpha=0.1, color=COLORS['accent'], label='Medium Risk')
        ax.axvspan(0.15, df['Risk'].max(), alpha=0.1, color=COLORS['success'], label='High Risk')
        
        stats_text = f"""
    Risk Statistics:
    • Mean: {mean_risk:.3f}
    • Std: {std_risk:.3f}
    • Median: {median_risk:.3f}
    • Range: [{df['Risk'].min():.3f}, {df['Risk'].max():.3f}]
    • Samples: {len(df):,}
    
    Risk Categories:
    • Low Risk (<0.05): {low_count:,} ({low_count/len(df)*100:.1f}%)
    • Medium Risk (0.05-0.15): {med_count:,} ({med_count/len(df)*100:.1f}%)
    • High Risk (>0.15): {high_count:,} ({high_count/len(df)*100:.1f}%)
        """
        ax.set_xlabel('Risk Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        save_path = 'figures/risk_distribution.png'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    save_figure_png_and_pdf(save_path, dpi_png=300, dpi_pdf=400, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Risk distribution figure saved: {save_path}")

def create_high_risk_geographic_map(risk_labels, lang='en'):
    """
    创建高风险节点地理分布图
    顶会级别设计：世界地图可视化高风险节点
    """
    is_cn = (lang == 'cn')
    
    if is_cn:
        setup_chinese_style()
        print("Creating high risk geographic map (CN)...")
    else:
        setup_english_style()
        print("Creating high risk geographic map...")
    
    # 先尝试安装geopandas
    try:
        import geopandas as gpd
        print("✓ GeoPandas already installed")
    except ImportError:
        print("Installing GeoPandas...")
        import subprocess
        import sys
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "geopandas"])
            import geopandas as gpd
            print("✓ GeoPandas installed successfully")
        except Exception as e:
            print(f"Failed to install GeoPandas: {e}")
            print("Using fallback visualization...")
            gpd = None
    
    # 转换为DataFrame
    risk_data = []
    for country, risk in risk_labels.items():
        try:
            risk_value = float(risk)
            # 提取国家名称（去掉行业信息）
            country_name = country.split('_')[0]
            risk_data.append({'Country': country_name, 'Risk': risk_value, 'Full_Name': country})
        except (ValueError, TypeError):
            continue
    
    df = pd.DataFrame(risk_data)
    df['Risk'] = pd.to_numeric(df['Risk'], errors='coerce')
    df = df.dropna(subset=['Risk'])
    
    # 按国家聚合风险（取最高风险）
    country_risk = df.groupby('Country')['Risk'].max().reset_index()
    
    # 国家名称映射
    country_mapping = {
        'USA': 'United States of America',
        'UK': 'United Kingdom',
        'UAE': 'United Arab Emirates',
        'South_Korea': 'South Korea',
        'North_Korea': 'North Korea',
        'Czech_Republic': 'Czech Republic',
        'Bosnia_Herzegovina': 'Bosnia and Herzegovina',
        'Dominican_Republic': 'Dominican Republic',
        'El_Salvador': 'El Salvador',
        'Costa_Rica': 'Costa Rica',
        'Puerto_Rico': 'Puerto Rico',
        'Trinidad_Tobago': 'Trinidad and Tobago',
        'Antigua_Barbuda': 'Antigua and Barbuda',
        'Saint_Kitts_Nevis': 'Saint Kitts and Nevis',
        'Saint_Vincent_Grenadines': 'Saint Vincent and the Grenadines',
        'Saint_Lucia': 'Saint Lucia',
        'Burkina_Faso': 'Burkina Faso',
        'Central_African_Republic': 'Central African Republic',
        'Democratic_Republic_Congo': 'Democratic Republic of the Congo',
        'Republic_Congo': 'Republic of the Congo',
        'Ivory_Coast': 'Côte d\'Ivoire',
        'Equatorial_Guinea': 'Equatorial Guinea',
        'Guinea_Bissau': 'Guinea-Bissau',
        'Sao_Tome_Principe': 'São Tomé and Príncipe',
        'Sierra_Leone': 'Sierra Leone',
        'South_Africa': 'South Africa',
        'South_Sudan': 'South Sudan',
        'Western_Sahara': 'Western Sahara',
        'Cape_Verde': 'Cape Verde',
        'Marshall_Islands': 'Marshall Islands',
        'Papua_New_Guinea': 'Papua New Guinea',
        'Solomon_Islands': 'Solomon Islands',
        'Timor_Leste': 'Timor-Leste',
        'New_Zealand': 'New Zealand',
        'Sri_Lanka': 'Sri Lanka',
        'United_Arab_Emirates': 'United Arab Emirates',
        'Saudi_Arabia': 'Saudi Arabia',
        'United_States': 'United States of America',
        'United_Kingdom': 'United Kingdom'
    }
    
    # 应用映射
    country_risk['Country_Mapped'] = country_risk['Country'].map(country_mapping).fillna(country_risk['Country'])
    
    # 创建matplotlib版本的地理分布图
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # 创建世界地图背景
    if gpd is not None:
        try:
            # 使用在线地图数据
            world_url = 'https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson'
            world = gpd.read_file(world_url)
            
            # 合并数据
            world_risk = world.merge(country_risk, left_on='name', right_on='Country_Mapped', how='left')
            world_risk['Risk'] = world_risk['Risk'].fillna(0)
            
            # 绘制地图
            world_risk.plot(column='Risk', cmap='RdYlBu_r', ax=ax, 
                           legend=True, edgecolor='black', linewidth=0.5,
                           legend_kwds={'label': 'Risk Level', 'orientation': 'horizontal', 'pad': 0.02})
            
            # 添加高风险国家标注
            high_risk_countries = country_risk.nlargest(10, 'Risk')
            for _, row in high_risk_countries.iterrows():
                country_geom = world_risk[world_risk['Country_Mapped'] == row['Country_Mapped']]
                if not country_geom.empty:
                    centroid = country_geom.geometry.centroid.iloc[0]
                    ax.annotate(f"{row['Country']}\n{row['Risk']:.3f}", 
                               xy=(centroid.x, centroid.y), 
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
            
            if is_cn:
                ax.set_xlabel('经度', fontsize=12, fontweight='bold')
                ax.set_ylabel('纬度', fontsize=12, fontweight='bold')
            else:
                ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
                ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
            
        except Exception as e:
            print(f"Geographic plotting failed: {e}")
            print("Using enhanced scatter plot instead...")
            gpd = None
    
    if gpd is None:
        # 备用方案：创建更美观的散点图
        # 按风险等级排序
        country_risk_sorted = country_risk.sort_values('Risk', ascending=False)
        
        # 创建散点图
        scatter = ax.scatter(range(len(country_risk_sorted)), country_risk_sorted['Risk'], 
                            c=country_risk_sorted['Risk'], cmap='RdYlBu_r', s=150, alpha=0.8,
                            edgecolors='black', linewidth=0.5)
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        if is_cn:
            cbar.set_label('风险等级', fontsize=12, fontweight='bold')
        else:
            cbar.set_label('Risk Level', fontsize=12, fontweight='bold')
        
        # 标注高风险国家
        top_10 = country_risk_sorted.head(10)
        for i, (_, row) in enumerate(top_10.iterrows()):
            ax.annotate(f"{row['Country']}\n{row['Risk']:.3f}", 
                       xy=(i, row['Risk']), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        if is_cn:
            ax.set_xlabel('国家排名 (按风险等级)', fontsize=12, fontweight='bold')
            ax.set_ylabel('风险等级', fontsize=12, fontweight='bold')
        else:
            ax.set_xlabel('Country Rank (by Risk Level)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Risk Level', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # 保存图形
    if is_cn:
        save_path = 'figures_cn/high_risk_geographic_map.png'
    else:
        save_path = 'figures/high_risk_geographic_map.png'
    
    plt.tight_layout()
    save_figure_png_and_pdf(save_path, dpi_png=300, dpi_pdf=400, bbox_inches='tight')
    plt.close()
    
    print(f"✓ High risk geographic map saved: {save_path}")

def create_sentiment_distribution_figure(sentiment_df, lang='en'):
    """
    创建情感特征分布图
    顶会级别设计：单一图表展示情感分布的核心特征
    """
    is_cn = (lang == 'cn')
    
    if is_cn:
        setup_chinese_style()
        print("Creating sentiment distribution figure (CN)...")
    else:
        setup_english_style()
        print("Creating sentiment distribution figure...")
    
    # 确保数据类型正确
    sentiment_df = sentiment_df.copy()
    sentiment_df['sentiment_mean'] = pd.to_numeric(sentiment_df['sentiment_mean'], errors='coerce')
    sentiment_df['sentiment_std'] = pd.to_numeric(sentiment_df['sentiment_std'], errors='coerce')
    sentiment_df['news_count'] = pd.to_numeric(sentiment_df['news_count'], errors='coerce')
    
    # 创建单一图表
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # 情感极性分布
    sentiment_values = sentiment_df['sentiment_mean'].dropna()
    
    # 绘制直方图
    n, bins, patches = ax.hist(sentiment_values, bins=50, alpha=0.7, 
                              color=COLORS['secondary'], edgecolor='white', linewidth=1.2)
    
    # 添加颜色渐变效果（基于频率）
    for i, (patch, freq) in enumerate(zip(patches, n)):
        alpha = 0.3 + 0.7 * (freq / n.max())
        patch.set_alpha(alpha)
        bin_center = (bins[i] + bins[i+1]) / 2
        if bin_center < -0.1:
            patch.set_facecolor(COLORS['success'])
        elif bin_center > 0.1:
            patch.set_facecolor(COLORS['primary'])
        else:
            patch.set_facecolor(COLORS['neutral'])
    
    # 添加统计信息
    mean_sentiment = sentiment_values.mean()
    std_sentiment = sentiment_values.std()
    median_sentiment = sentiment_values.median()
    
    if is_cn:
        ax.axvline(mean_sentiment, color=COLORS['accent'], linestyle='-', linewidth=3, 
                   label=f'均值: {mean_sentiment:.3f}')
        ax.axvline(median_sentiment, color=COLORS['dark'], linestyle='--', linewidth=2, 
                   label=f'中位数: {median_sentiment:.3f}')
        ax.axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.7, 
                   label='中性 (0.0)')
        
        stats_text = f"""
    统计信息:
    • 均值: {mean_sentiment:.3f}
    • 标准差: {std_sentiment:.3f}
    • 中位数: {median_sentiment:.3f}
    • 范围: [{sentiment_values.min():.3f}, {sentiment_values.max():.3f}]
    • 样本数: {len(sentiment_values):,}
        """
        ax.set_xlabel('情感极性', fontsize=14, fontweight='bold')
        ax.set_ylabel('频率', fontsize=14, fontweight='bold')
        ax.axvspan(-1.0, -0.1, alpha=0.1, color=COLORS['success'], label='负面区域')
        ax.axvspan(-0.1, 0.1, alpha=0.1, color=COLORS['neutral'], label='中性区域')
        ax.axvspan(0.1, 1.0, alpha=0.1, color=COLORS['primary'], label='正面区域')
        save_path = 'figures_cn/sentiment_distribution.png'
    else:
        ax.axvline(mean_sentiment, color=COLORS['accent'], linestyle='-', linewidth=3, 
                   label=f'Mean: {mean_sentiment:.3f}')
        ax.axvline(median_sentiment, color=COLORS['dark'], linestyle='--', linewidth=2, 
                   label=f'Median: {median_sentiment:.3f}')
        ax.axvline(0, color='black', linestyle=':', linewidth=2, alpha=0.7, 
                   label='Neutral (0.0)')
        
        stats_text = f"""
    Statistics:
    • Mean: {mean_sentiment:.3f}
    • Std: {std_sentiment:.3f}
    • Median: {median_sentiment:.3f}
    • Range: [{sentiment_values.min():.3f}, {sentiment_values.max():.3f}]
    • Samples: {len(sentiment_values):,}
        """
        ax.set_xlabel('Sentiment Polarity', fontsize=14, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
        ax.axvspan(-1.0, -0.1, alpha=0.1, color=COLORS['success'], label='Negative Region')
        ax.axvspan(-0.1, 0.1, alpha=0.1, color=COLORS['neutral'], label='Neutral Region')
        ax.axvspan(0.1, 1.0, alpha=0.1, color=COLORS['primary'], label='Positive Region')
        save_path = 'figures/sentiment_distribution.png'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(-1.1, 1.1)
    
    plt.tight_layout()
    save_figure_png_and_pdf(save_path, dpi_png=300, dpi_pdf=400, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Sentiment distribution figure saved: {save_path}")

def create_combined_figure(networks, risk_labels, sentiment_df):
    """
    创建组合图：三个子图在一个图中展示
    顶会级别的组合设计
    """
    print("Creating combined figure...")
    
    fig = plt.figure(figsize=(20, 6))
    
    # 子图1：网络拓扑（简化版）
    ax1 = plt.subplot(1, 3, 1)
    G = networks[2020]
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # 只显示主要节点 - 降低阈值确保有节点显示
    major_nodes = [node for node in G.nodes() if G.nodes[node].get('total_degree', 0) > 100]
    if len(major_nodes) < 10:  # 如果节点太少，进一步降低阈值
        major_nodes = [node for node in G.nodes() if G.nodes[node].get('total_degree', 0) > 50]
    if len(major_nodes) < 5:  # 如果还是太少，显示所有节点
        major_nodes = list(G.nodes())[:50]  # 最多显示50个节点
    
    subG = G.subgraph(major_nodes)
    subpos = {node: pos[node] for node in major_nodes}
    
    node_sizes = [max(100, min(300, G.nodes[node].get('total_degree', 0) * 0.05)) for node in major_nodes]
    node_colors = []
    for node in major_nodes:
        risk = risk_labels.get(node, 0)
        try:
            risk_value = float(risk)
        except (ValueError, TypeError):
            risk_value = 0.0
        node_colors.append(risk_value)
    
    nx.draw_networkx_nodes(subG, subpos, node_size=node_sizes, node_color=node_colors,
                          cmap='RdYlBu_r', alpha=0.8, ax=ax1)
    nx.draw_networkx_edges(subG, subpos, alpha=0.3, width=0.5, ax=ax1)
    
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 子图2：风险分布
    ax2 = plt.subplot(1, 3, 2)
    risk_data = []
    for risk in risk_labels.values():
        try:
            risk_data.append(float(risk))
        except (ValueError, TypeError):
            continue
    ax2.hist(risk_data, bins=20, alpha=0.7, color=COLORS['primary'], edgecolor='white')
    ax2.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 子图3：情感分布
    ax3 = plt.subplot(1, 3, 3)
    sentiment_values = sentiment_df['sentiment_mean'].dropna()
    ax3.hist(sentiment_values, bins=20, alpha=0.7, color=COLORS['secondary'], edgecolor='white')
    ax3.set_xlabel('Sentiment Polarity', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axvline(0, color=COLORS['accent'], linestyle='--', linewidth=2)
    
    # 保存图形
    plt.tight_layout()
    save_figure_png_and_pdf('figures/combined_overview.png', dpi_png=300, dpi_pdf=400, bbox_inches='tight')
    plt.close()
    
    print("✓ Combined overview figure saved: figures/combined_overview.png")

def main():
    """主函数"""
    print("="*80)
    print("Creating Conference-Level Paper Figures (EN & CN)")
    print("="*80)
    
    # 创建输出目录
    Path('figures').mkdir(exist_ok=True)
    Path('figures_cn').mkdir(exist_ok=True)
    
    # 加载数据
    networks, risk_labels, sentiment_df = load_data()
    
    # 创建英文版图表
    create_network_topology_figure(networks, risk_labels, lang='en')
    create_risk_distribution_figure(risk_labels, lang='en')
    create_high_risk_geographic_map(risk_labels, lang='en')
    create_sentiment_distribution_figure(sentiment_df, lang='en')
    
    # 创建中文版图表
    create_network_topology_figure(networks, risk_labels, lang='cn')
    create_risk_distribution_figure(risk_labels, lang='cn')
    create_high_risk_geographic_map(risk_labels, lang='cn')
    create_sentiment_distribution_figure(sentiment_df, lang='cn')
    
    print("\n" + "="*80)
    print("All figures created successfully!")
    print("="*80)
    print("Generated files (English):")
    print("  ✓ figures/network_topology.png")
    print("  ✓ figures/risk_distribution.png") 
    print("  ✓ figures/high_risk_geographic_map.png")
    print("  ✓ figures/sentiment_distribution.png")
    print("\nGenerated files (Chinese):")
    print("  ✓ figures_cn/network_topology.png")
    print("  ✓ figures_cn/risk_distribution.png") 
    print("  ✓ figures_cn/high_risk_geographic_map.png")
    print("  ✓ figures_cn/sentiment_distribution.png")

if __name__ == "__main__":
    main()
