"""
Affective DomiRank Framework Architecture Diagram
专业论文框架图 - 详细版
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D
import numpy as np

# 设置字体
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'cm'

def draw_rounded_box(ax, x, y, w, h, text, subtext=None, facecolor='white', 
                     edgecolor='#333333', fontsize=10, subtextsize=8, 
                     textcolor='black', linewidth=1.5, alpha=1.0, bold=True):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15",
                         facecolor=facecolor, edgecolor=edgecolor, 
                         linewidth=linewidth, alpha=alpha)
    ax.add_patch(box)
    
    if subtext:
        ax.text(x + w/2, y + h*0.65, text, fontsize=fontsize, 
                fontweight='bold' if bold else 'normal', ha='center', va='center', color=textcolor)
        ax.text(x + w/2, y + h*0.3, subtext, fontsize=subtextsize, 
                ha='center', va='center', color='#555555', style='italic')
    else:
        ax.text(x + w/2, y + h/2, text, fontsize=fontsize, 
                fontweight='bold' if bold else 'normal', ha='center', va='center', color=textcolor)
    return box

def draw_arrow(ax, start, end, color='#555555', style='->', lw=1.5, 
               connectionstyle="arc3,rad=0", shrinkA=2, shrinkB=2):
    """绘制箭头"""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                               connectionstyle=connectionstyle,
                               shrinkA=shrinkA, shrinkB=shrinkB))

def draw_module_background(ax, x, y, w, h, title, color, titlecolor):
    """绘制模块背景"""
    bg = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03,rounding_size=0.2",
                        facecolor=color, edgecolor=titlecolor, linewidth=2.5, alpha=0.95)
    ax.add_patch(bg)
    # 标题栏
    title_bar = FancyBboxPatch((x, y+h-0.8), w, 0.8, 
                               boxstyle="round,pad=0.02,rounding_size=0.15",
                               facecolor=titlecolor, edgecolor=titlecolor, linewidth=0)
    ax.add_patch(title_bar)
    ax.text(x + w/2, y + h - 0.4, title, fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')

def create_framework_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('#FAFAFA')
    
    # ==================== 颜色定义 ====================
    colors = {
        'input': '#ECEFF1',        # 输入层灰色
        'sentiment_bg': '#E3F2FD', # 情感模块背景
        'sentiment': '#1565C0',    # 情感模块主色
        'domirank_bg': '#FFF8E1',  # DomiRank模块背景
        'domirank': '#F57C00',     # DomiRank模块主色
        'stgnn_bg': '#E8F5E9',     # STGNN模块背景
        'stgnn': '#2E7D32',        # STGNN模块主色
        'output': '#FFEBEE',       # 输出层
        'output_border': '#C62828',
        'innovation': '#D32F2F',   # 创新点高亮
        'arrow': '#616161',
        'formula_bg': '#FFFFFF',
    }
    
    # ==================== 输入层 (顶部) ====================
    # 数据源背景
    input_bg = FancyBboxPatch((0.5, 10.2), 17, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=colors['input'], edgecolor='#90A4AE', linewidth=1.5)
    ax.add_patch(input_bg)
    ax.text(9, 11.5, 'INPUT DATA SOURCES', fontsize=10, fontweight='bold', 
            ha='center', color='#546E7A')
    
    # 三个数据源
    draw_rounded_box(ax, 1, 10.4, 3.5, 1.1, 'News & Media', 
                     'Financial news, Reuters, Bloomberg',
                     facecolor='white', edgecolor='#78909C', fontsize=9)
    draw_rounded_box(ax, 5.5, 10.4, 3.5, 1.1, 'Policy Documents',
                     'Trade policies, Regulations',
                     facecolor='white', edgecolor='#78909C', fontsize=9)
    draw_rounded_box(ax, 10, 10.4, 3.5, 1.1, 'Trade Database',
                     'BACI (240 countries, 2015-2022)',
                     facecolor='white', edgecolor='#78909C', fontsize=9)
    draw_rounded_box(ax, 14, 10.4, 3, 1.1, 'Network',
                     'G = (V, E, A)',
                     facecolor='white', edgecolor='#78909C', fontsize=9)
    
    # ==================== Module 1: Sentiment Computing ====================
    draw_module_background(ax, 0.5, 4.8, 5.5, 5.2, 'MODULE 1: Sentiment Computing', 
                          colors['sentiment_bg'], colors['sentiment'])
    
    # Step 1.1: Text Preprocessing
    draw_rounded_box(ax, 0.8, 8.5, 4.9, 0.9, 'Step 1.1: Text Preprocessing',
                     'Tokenization, Cleaning, Entity Recognition',
                     facecolor='white', edgecolor=colors['sentiment'], fontsize=9, subtextsize=7)
    
    draw_arrow(ax, (3.25, 8.4), (3.25, 7.85), color=colors['sentiment'])
    
    # Step 1.2: BERT Encoding
    draw_rounded_box(ax, 0.8, 6.95, 4.9, 0.9, 'Step 1.2: BERT Encoder',
                     'bert-base-multilingual-cased',
                     facecolor='white', edgecolor=colors['sentiment'], fontsize=9, subtextsize=7)
    ax.text(3.25, 7.15, r'$h_d = BERT(d)[CLS]$', fontsize=8, ha='center', color='#333')
    
    draw_arrow(ax, (3.25, 6.85), (3.25, 6.35), color=colors['sentiment'])
    
    # Step 1.3: Emotion Classification
    draw_rounded_box(ax, 0.8, 5.4, 4.9, 0.95, 'Step 1.3: Multi-task Emotion Classifier',
                     None, facecolor='white', edgecolor=colors['sentiment'], fontsize=9)
    ax.text(3.25, 5.65, '7-Dimensional Emotion Vector:', fontsize=7, ha='center', 
            color='#555', fontweight='bold')
    
    # 7维向量详细展示
    emotions = ['pol', 'int', 'fear', 'trust', 'anger', 'joy', 'sad']
    emotion_colors = ['#E53935', '#FB8C00', '#8E24AA', '#43A047', '#D32F2F', '#FDD835', '#1E88E5']
    for i, (em, ec) in enumerate(zip(emotions, emotion_colors)):
        x_pos = 1.0 + i * 0.68
        rect = Rectangle((x_pos, 5.08), 0.6, 0.25, facecolor=ec, alpha=0.4, edgecolor=ec, linewidth=1)
        ax.add_patch(rect)
        ax.text(x_pos + 0.3, 5.2, em, fontsize=5.5, ha='center', va='center', color='#333', fontweight='bold')
    
    # ==================== Module 2: Affective DomiRank ====================
    draw_module_background(ax, 6.25, 4.8, 5.5, 5.2, 'MODULE 2: Affective DomiRank', 
                          colors['domirank_bg'], colors['domirank'])
    
    # Step 2.1: Sentiment-Risk Mapping
    draw_rounded_box(ax, 6.55, 8.3, 4.9, 1.1, 'Step 2.1: Sentiment-Risk Mapping',
                     None, facecolor='white', edgecolor=colors['domirank'], fontsize=9)
    # 公式 - 简化版
    ax.text(9, 8.6, 'Asymmetric Non-linear Transform:', fontsize=7, ha='center', color='#555')
    ax.text(9, 8.25, r'$s(e) = |e|^{0.8}$ if e<0', fontsize=8, ha='center', color='#333')
    ax.text(9, 7.95, r'$s(e) = -0.5 \cdot e^{0.8}$ if e>0', fontsize=8, ha='center', color='#333')
    
    draw_arrow(ax, (9, 8.2), (9, 7.55), color=colors['domirank'])
    
    # Step 2.2: Network Construction
    draw_rounded_box(ax, 6.55, 6.65, 4.9, 0.9, 'Step 2.2: Weighted Adjacency Matrix',
                     None, facecolor='white', edgecolor=colors['domirank'], fontsize=9)
    ax.text(9, 6.9, r'$A_{ij} = \log(1 + trade_{ij}) \cdot w_{ij}$', 
            fontsize=8, ha='center', color='#333')
    
    draw_arrow(ax, (9, 6.55), (9, 6.0), color=colors['domirank'])
    
    # Step 2.3: ODE Solver (创新点 - 红色高亮)
    box_ode = FancyBboxPatch((6.55, 5.0), 4.9, 1.0, boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor='#FFEBEE', edgecolor=colors['innovation'], linewidth=2.5)
    ax.add_patch(box_ode)
    ax.text(9, 5.75, 'Step 2.3: ODE Solver (Euler Method)', fontsize=9, fontweight='bold',
            ha='center', color=colors['innovation'])
    ax.text(9, 5.35, r'$d\Gamma_i/dt = \alpha \Sigma A_{ij}(\theta - \Gamma_j) - \beta\Gamma_i + \gamma \cdot s(e_i)$', 
            fontsize=7, ha='center', color='#333')
    # 创新标签
    innovation_label = FancyBboxPatch((10.5, 5.65), 0.9, 0.3, boxstyle="round,pad=0.02",
                                      facecolor=colors['innovation'], edgecolor='none')
    ax.add_patch(innovation_label)
    ax.text(10.95, 5.8, 'NEW', fontsize=6, fontweight='bold', ha='center', va='center', color='white')
    
    # ==================== Module 3: STGNN Prediction ====================
    draw_module_background(ax, 12, 4.8, 5.5, 5.2, 'MODULE 3: STGNN Prediction', 
                          colors['stgnn_bg'], colors['stgnn'])
    
    # Step 3.1: Feature Engineering
    draw_rounded_box(ax, 12.3, 8.3, 4.9, 1.1, 'Step 3.1: Node Feature Engineering',
                     None, facecolor='white', edgecolor=colors['stgnn'], fontsize=9)
    ax.text(14.75, 8.6, '16-dim Feature Vector per Node:', fontsize=7, ha='center', color='#555')
    # 特征分组
    ax.text(14.75, 8.15, 'Structural(4) + Trade(4) + Emotion(7) + Gamma(1)', 
            fontsize=7, ha='center', color='#333', fontweight='bold')
    
    draw_arrow(ax, (14.75, 8.2), (14.75, 7.55), color=colors['stgnn'])
    
    # Step 3.2: Adaptive Graph Learning
    draw_rounded_box(ax, 12.3, 6.55, 4.9, 1.0, 'Step 3.2: Adaptive Graph Learning',
                     None, facecolor='white', edgecolor=colors['stgnn'], fontsize=9)
    ax.text(14.75, 6.85, r'$\tilde{A} = softmax(ReLU(E_1 \cdot E_2^T))$', 
            fontsize=8, ha='center', color='#333')
    ax.text(14.75, 6.55, 'E1, E2: learnable node embeddings', 
            fontsize=7, ha='center', color='#666', style='italic')
    
    draw_arrow(ax, (14.75, 6.45), (14.75, 5.95), color=colors['stgnn'])
    
    # Step 3.3: AGCRN
    draw_rounded_box(ax, 12.3, 5.0, 4.9, 0.95, 'Step 3.3: AGCRN (2 Layers)',
                     None, facecolor='white', edgecolor=colors['stgnn'], fontsize=9)
    ax.text(14.75, 5.55, 'Chebyshev GCN (K=2) + GRU', fontsize=7, ha='center', color='#333', fontweight='bold')
    ax.text(14.75, 5.25, r'$H = \sigma(\Sigma T_k(\tilde{L}) \cdot X \cdot W_k)$', 
            fontsize=7, ha='center', color='#333')
    
    # ==================== 输出层 ====================
    output_bg = FancyBboxPatch((0.5, 0.3), 17, 1.5, boxstyle="round,pad=0.02,rounding_size=0.1",
                               facecolor=colors['output'], edgecolor=colors['output_border'], linewidth=2)
    ax.add_patch(output_bg)
    ax.text(9, 1.5, 'OUTPUT: Vulnerability Prediction', fontsize=11, fontweight='bold', 
            ha='center', color=colors['output_border'])
    
    # 输出指标
    metrics = [('MAE', '0.0228'), ('RMSE', '0.0340'), ('R^2', '0.8578')]
    for i, (name, val) in enumerate(metrics):
        x_pos = 4 + i * 4
        draw_rounded_box(ax, x_pos, 0.5, 2.5, 0.8, f'{name}: {val}', None,
                        facecolor='white', edgecolor=colors['output_border'], 
                        fontsize=9, bold=True, textcolor=colors['output_border'])
    
    # ==================== 中间处理层 ====================
    # Node Aggregation
    draw_rounded_box(ax, 1.5, 3.5, 3.5, 0.9, 'Node-Level Aggregation',
                     'Average pooling per node',
                     facecolor='white', edgecolor=colors['sentiment'], fontsize=9, subtextsize=7)
    
    # Convergence Check
    draw_rounded_box(ax, 7, 3.5, 4, 0.9, 'Convergence Check',
                     'Iterate until stable',
                     facecolor='white', edgecolor=colors['domirank'], fontsize=9, subtextsize=7)
    
    # Temporal Window
    draw_rounded_box(ax, 12.5, 3.5, 4.5, 0.9, 'Sliding Window (T=12)',
                     'Predict next step',
                     facecolor='white', edgecolor=colors['stgnn'], fontsize=9, subtextsize=7)
    
    # ==================== 连接箭头 ====================
    # 输入到模块
    draw_arrow(ax, (2.75, 10.3), (2.75, 9.5), color=colors['arrow'], lw=2)
    draw_arrow(ax, (7.25, 10.3), (7.25, 9.5), color=colors['arrow'], lw=2)
    draw_arrow(ax, (11.75, 10.3), (9, 9.5), color=colors['arrow'], lw=2)
    draw_arrow(ax, (15.5, 10.3), (14.75, 9.5), color=colors['arrow'], lw=2)
    
    # 模块到中间层
    draw_arrow(ax, (3.25, 4.8), (3.25, 4.5), color=colors['sentiment'], lw=2)
    draw_arrow(ax, (9, 4.9), (9, 4.5), color=colors['domirank'], lw=2)
    draw_arrow(ax, (14.75, 4.8), (14.75, 4.5), color=colors['stgnn'], lw=2)
    
    # 中间层到输出
    draw_arrow(ax, (3.25, 3.4), (3.25, 2.0), color=colors['arrow'], lw=1.5)
    draw_arrow(ax, (9, 3.4), (9, 2.0), color=colors['arrow'], lw=1.5)
    draw_arrow(ax, (14.75, 3.4), (14.75, 2.0), color=colors['arrow'], lw=1.5)
    
    # 模块间横向连接
    # Module 1 → Module 2 (情感向量传递)
    draw_arrow(ax, (5.8, 5.75), (6.5, 5.75), color=colors['sentiment'], lw=2)
    ax.text(6.15, 6.05, 's(e)', fontsize=8, ha='center', color=colors['sentiment'], fontweight='bold')
    
    # Module 2 → Module 3 (DomiRank传递)
    draw_arrow(ax, (11.5, 5.5), (12.25, 5.5), color=colors['domirank'], lw=2)
    ax.text(11.85, 5.8, 'Gamma*', fontsize=8, ha='center', color=colors['domirank'], fontweight='bold')
    
    # Module 1 直接到 Module 3 (情感向量跨模块传递)
    # 使用虚线表示
    ax.annotate('', xy=(12.3, 8.5), xytext=(5.8, 5.3),
                arrowprops=dict(arrowstyle='->', color=colors['sentiment'], lw=1.5,
                               connectionstyle="arc3,rad=-0.15", linestyle='--'))
    ax.text(8.5, 7.5, 'e_i (7-dim)', fontsize=8, ha='center', 
            color=colors['sentiment'], style='italic', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='none', alpha=0.8))
    
    # ==================== 图例 ====================
    legend_elements = [
        mpatches.Patch(facecolor=colors['sentiment_bg'], edgecolor=colors['sentiment'], 
                      linewidth=2, label='Sentiment Computing'),
        mpatches.Patch(facecolor=colors['domirank_bg'], edgecolor=colors['domirank'], 
                      linewidth=2, label='Affective DomiRank'),
        mpatches.Patch(facecolor=colors['stgnn_bg'], edgecolor=colors['stgnn'], 
                      linewidth=2, label='STGNN Prediction'),
        mpatches.Patch(facecolor='#FFEBEE', edgecolor=colors['innovation'], 
                      linewidth=2, label='Innovation (New)'),
        Line2D([0], [0], linestyle='--', color=colors['sentiment'], lw=1.5, 
               label='Cross-module Data Flow'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, 
              framealpha=0.95, edgecolor='#CCCCCC')
    
    # ==================== 标题 ====================
    ax.text(9, 11.9, 'Affective DomiRank: A Sentiment-Enhanced Framework for Supply Chain Vulnerability Prediction', 
            fontsize=13, fontweight='bold', ha='center', color='#212121')
    
    plt.savefig('figures/framework_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    plt.savefig('figures/framework_architecture.pdf', bbox_inches='tight',
                facecolor='#FAFAFA', edgecolor='none')
    print("Framework diagram saved to:")
    print("  - figures/framework_architecture.png")
    print("  - figures/framework_architecture.pdf")
    plt.show()


if __name__ == "__main__":
    create_framework_diagram()
