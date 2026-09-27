"""
敏感性分析结果可视化 - 顶级期刊风格
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import networkx as nx

from figure_export_utils import save_figure_png_and_pdf

# ============================================================================
# 配色方案
# ============================================================================
PALETTE = {
    'primary': '#4A7298',
    'secondary': '#B85C4C', 
    'tertiary': '#5B8A72',
    'accent': '#D4A84B',
    'light': '#9CA3AF',
    'dark': '#374151',
}

PANEL_BG = {
    'alpha': '#F2F5FA',
    'threshold': '#F3F7F4',
    'window': '#F7F3EE',
    'impact': '#F5F2F7',
    'heatmap': '#F2F4F8',
}

# ============================================================================
# 字体设置
# ============================================================================
def get_font_prop(lang):
    """获取字体属性"""
    if lang == 'cn':
        for fp in ['C:/Windows/Fonts/simhei.ttf', 'C:/Windows/Fonts/msyh.ttc']:
            if Path(fp).exists():
                return FontProperties(fname=fp)
    return None

def setup_ax_style(ax, grid_axis='y'):
    """设置坐标轴样式"""
    for spine in ax.spines.values():
        spine.set_color(PALETTE['light'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if grid_axis == 'y':
        ax.yaxis.grid(True, linestyle='-', alpha=0.1, linewidth=0.5)
    elif grid_axis == 'x':
        ax.xaxis.grid(True, linestyle='-', alpha=0.1, linewidth=0.5)
    elif grid_axis == 'both':
        ax.grid(True, linestyle='-', alpha=0.1, linewidth=0.5)

def smooth_curve(x, y, num=240, logx=False):
    """平滑趋势曲线并返回软带宽度"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if logx and np.any(x <= 0):
        logx = False

    degree = 3 if len(x) >= 4 else 2 if len(x) >= 3 else 1

    if logx:
        x_fit = np.log10(x)
        x_new = np.linspace(x_fit.min(), x_fit.max(), num)
        coeffs = np.polyfit(x_fit, y, degree)
        y_new = np.polyval(coeffs, x_new)
        x_plot = 10 ** x_new
        y_fit = np.polyval(coeffs, x_fit)
    else:
        x_fit = x
        x_new = np.linspace(x.min(), x.max(), num)
        coeffs = np.polyfit(x_fit, y, degree)
        y_new = np.polyval(coeffs, x_new)
        x_plot = x_new
        y_fit = np.polyval(coeffs, x_fit)

    span = y.max() - y.min()
    if len(x) >= 3:
        resid = y - y_fit
        band = max(np.std(resid), span * 0.04)
    else:
        band = span * 0.06 if span > 0 else 0.01
    band = max(band, 0.002)

    y_new = np.clip(y_new, y.min() - band * 1.5, y.max() + band * 1.5)
    return x, y, x_plot, y_new, band

def format_log_ticks(values):
    """格式化对数轴刻度显示"""
    labels = []
    for v in values:
        if v < 0.01:
            labels.append(f'{v:.3f}')
        elif v < 0.1:
            labels.append(f'{v:.2f}')
        else:
            labels.append(f'{v:.1f}')
    return labels

def scale_sizes(values, min_size=180, max_size=760):
    """把数值映射为气泡大小"""
    values = np.asarray(values, dtype=float)
    if values.max() == values.min():
        return np.full_like(values, (min_size + max_size) / 2)
    return min_size + (values - values.min()) / (values.max() - values.min()) * (max_size - min_size)

def response_ribbon(ax, x, y, color, marker='o', logx=False):
    """响应带 + 点阵"""
    x_sorted, y_sorted, x_smooth, y_smooth, band = smooth_curve(x, y, logx=logx)
    ax.fill_between(x_smooth, y_smooth - band, y_smooth + band,
                    color=color, alpha=0.18, linewidth=0)
    ax.scatter(x_sorted, y_sorted, s=70, facecolors='white',
               edgecolors=color, linewidths=1.6, marker=marker, zorder=3)

    best_idx = int(np.argmax(y_sorted))
    ax.scatter([x_sorted[best_idx]], [y_sorted[best_idx]],
               s=140, color=PALETTE['accent'], edgecolors='white',
               linewidths=1.4, marker=marker, zorder=4)
    ax.scatter([x_sorted[best_idx]], [y_sorted[best_idx]],
               s=300, color=PALETTE['accent'], alpha=0.18,
               marker=marker, zorder=2)
    return x_sorted, y_sorted, best_idx

def bubble_plot(ax, x, y, base_color, highlight_color):
    """气泡图表达离散点"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sizes = scale_sizes(y, min_size=240, max_size=900)
    best_idx = int(np.argmax(y))
    colors = [base_color] * len(y)
    colors[best_idx] = highlight_color

    ax.scatter(x, y, s=sizes, c=colors, alpha=0.85,
               edgecolors='white', linewidths=1.6, zorder=3)
    ax.scatter([x[best_idx]], [y[best_idx]], s=sizes[best_idx] * 1.6,
               facecolors='none', edgecolors=highlight_color, linewidths=1.6, zorder=2)
    return best_idx, sizes

def dot_rank_plot(ax, labels, values, colors, fp=None, show_values=True, text_size=9):
    """排名点图"""
    y_pos = np.arange(len(values))
    max_val = max(values) if values else 0
    offset = max_val * 0.04 if max_val > 0 else 0.01

    for i, (label, val) in enumerate(zip(labels, values)):
        ax.hlines(i, 0, val, color=PALETTE['light'], linewidth=2, alpha=0.6)
        ax.scatter(val, i, s=160, color=colors[i], edgecolors='white',
                   linewidths=1.4, zorder=3)
        if show_values:
            ax.text(val + offset, i, f'{val:.3f}', va='center',
                    fontsize=text_size, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontproperties=fp)

def apply_panel_card(ax, label=None, facecolor='#F6F7FB'):
    """添加面板卡片背景和编号"""
    ax.set_facecolor('none')
    card = FancyBboxPatch(
        (0, 0), 1, 1, transform=ax.transAxes,
        boxstyle='round,pad=0.02,rounding_size=0.03',
        linewidth=0, facecolor=facecolor, zorder=-10, clip_on=False
    )
    ax.add_patch(card)
    if label:
        ax.text(0.02, 0.98, f'({label})', transform=ax.transAxes,
                ha='left', va='top', fontsize=11, fontweight='bold',
                color=PALETTE['dark'])

def get_norm_from_dict(values_dict):
    """根据字典值创建Normalize"""
    values = np.array(list(values_dict.values()), dtype=float)
    if values.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin, vmax = float(values.min()), float(values.max())
    if np.isclose(vmin, vmax):
        vmin -= 1.0
        vmax += 1.0
    return Normalize(vmin=vmin, vmax=vmax)

def load_year_network(year=2021):
    """加载指定年份网络"""
    base_dir = Path('data/processed/networks')
    direct_file = base_dir / f'network_{year}.pkl'
    if direct_file.exists():
        with open(direct_file, 'rb') as f:
            return pickle.load(f)

    all_file = base_dir / 'all_networks_2017_2021.pkl'
    if all_file.exists():
        with open(all_file, 'rb') as f:
            networks = pickle.load(f)
        if year in networks:
            return networks[year]

    graphml_file = base_dir / f'network_{year}.graphml'
    if graphml_file.exists():
        return nx.read_graphml(graphml_file)

    raise FileNotFoundError(f'找不到 {year} 年网络文件')

def select_subgraph(G, top_k=60):
    """选取代表性子图"""
    if G.number_of_nodes() <= top_k:
        G_sub = G.copy()
    else:
        if all('total_degree' in G.nodes[n] for n in G.nodes()):
            nodes_sorted = sorted(G.nodes(), key=lambda n: G.nodes[n].get('total_degree', 0.0), reverse=True)
        else:
            degree_vals = dict(G.degree(weight='weight'))
            nodes_sorted = sorted(G.nodes(), key=lambda n: degree_vals.get(n, 0.0), reverse=True)
        top_nodes = nodes_sorted[:top_k]
        G_sub = G.subgraph(top_nodes).copy()

    G_und = G_sub.to_undirected()
    if G_und.number_of_nodes() == 0:
        return G_sub

    components = list(nx.connected_components(G_und))
    if components:
        largest = max(components, key=len)
        G_sub = G_sub.subgraph(largest).copy()

    return G_sub

def compute_affective_domirank_scores(G, alpha=0.01, threshold=0.05, beta=0.5, theta=1.0, gamma=0.1,
                                      max_iter=1200, tol=1e-6, time_step=0.01):
    """计算参数化Affective DomiRank分数（轻量版）"""
    nodes = list(G.nodes())
    n = len(nodes)
    node_to_idx = {node: i for i, node in enumerate(nodes)}

    sentiment_risk = {}
    for node in nodes:
        sentiment = G.nodes[node].get('sentiment_mean', 0.0)
        if abs(sentiment) < threshold:
            risk = 0.0
        elif sentiment < 0:
            risk = abs(sentiment) ** 0.8
        else:
            risk = -(sentiment ** 0.8) * 0.5
        sentiment_risk[node] = risk

    gamma_vals = np.ones(n)
    for _ in range(max_iter):
        dgamma = np.zeros(n)
        for i, node in enumerate(nodes):
            neighbor_sum = 0.0
            for neighbor in G.predecessors(node):
                j = node_to_idx[neighbor]
                neighbor_sum += (theta - gamma_vals[j])
            dgamma[i] = alpha * neighbor_sum - beta * gamma_vals[i] + gamma * sentiment_risk[node]

        gamma_new = gamma_vals + time_step * dgamma
        if np.linalg.norm(gamma_new - gamma_vals) < tol:
            gamma_vals = gamma_new
            break
        gamma_vals = gamma_new

    return {node: gamma_vals[i] for i, node in enumerate(nodes)}

# ============================================================================
# 图1: α敏感性
# ============================================================================
def create_alpha_sensitivity_figure(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    xlabel = r'阻尼因子 $\alpha$' if lang == 'cn' else r'Damping Factor $\alpha$'
    ylabel = r'预测精度 ($R^2$)' if lang == 'cn' else r'Prediction Accuracy ($R^2$)'
    opt_text = '最优' if lang == 'cn' else 'Optimal'
    
    data = results['alpha_sensitivity']
    alphas = [d['alpha'] for d in data]
    r2_values = [d['test_r2'] for d in data]
    
    fig, ax = plt.subplots(figsize=(5.6, 4))
    apply_panel_card(ax, facecolor=PANEL_BG['alpha'])
    
    x_sorted, y_sorted, best_idx = response_ribbon(
        ax, alphas, r2_values, PALETTE['primary'], marker='o', logx=True
    )
    best_alpha, best_r2 = x_sorted[best_idx], y_sorted[best_idx]
    
    ax.annotate(f'{opt_text}\n$R^2$={best_r2:.3f}',
                xy=(best_alpha, best_r2), xytext=(14, 14), textcoords='offset points',
                fontsize=8, ha='left', fontproperties=fp,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=PALETTE['light'], linewidth=0.5))
    
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.set_ylabel(ylabel, fontproperties=fp)
    ax.set_xscale('log')
    ax.set_xticks(sorted(alphas))
    ax.set_xticklabels(format_log_ticks(sorted(alphas)))
    
    y_min, y_max = float(np.min(y_sorted)), float(np.max(y_sorted))
    margin = (y_max - y_min) * 0.25 if y_max > y_min else 0.01
    ax.set_ylim([y_min - margin * 0.6, y_max + margin])
    
    setup_ax_style(ax, grid_axis='y')
    plt.tight_layout()
    save_figure_png_and_pdf(output_dir / 'sensitivity_alpha.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_alpha.png'}")

# ============================================================================
# 图2: τ敏感性
# ============================================================================
def create_threshold_sensitivity_figure(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    xlabel = r'情感阈值 $\tau$' if lang == 'cn' else r'Emotion Threshold $\tau$'
    ylabel = r'预测精度 ($R^2$)' if lang == 'cn' else r'Prediction Accuracy ($R^2$)'
    opt_text = '最优' if lang == 'cn' else 'Optimal'
    
    data = results['threshold_sensitivity']
    thresholds = [d['threshold'] for d in data]
    r2_values = [d['test_r2'] for d in data]
    
    fig, ax = plt.subplots(figsize=(5.6, 4))
    apply_panel_card(ax, facecolor=PANEL_BG['threshold'])
    
    x_sorted, y_sorted, best_idx = response_ribbon(
        ax, thresholds, r2_values, PALETTE['tertiary'], marker='s', logx=False
    )
    best_t, best_r2 = x_sorted[best_idx], y_sorted[best_idx]
    
    ax.annotate(f'{opt_text}\n$R^2$={best_r2:.3f}',
                xy=(best_t, best_r2), xytext=(14, 14), textcoords='offset points',
                fontsize=8, ha='left', fontproperties=fp,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                          edgecolor=PALETTE['light'], linewidth=0.5))
    
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.set_ylabel(ylabel, fontproperties=fp)
    
    y_min, y_max = float(np.min(y_sorted)), float(np.max(y_sorted))
    margin = (y_max - y_min) * 0.25 if y_max > y_min else 0.01
    ax.set_ylim([y_min - margin * 0.6, y_max + margin])
    
    setup_ax_style(ax, grid_axis='y')
    plt.tight_layout()
    save_figure_png_and_pdf(output_dir / 'sensitivity_threshold.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_threshold.png'}")

# ============================================================================
# 图3: 时间窗口
# ============================================================================
def create_window_sensitivity_figure(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    xlabel = '时间窗口 (年)' if lang == 'cn' else 'Time Window (Years)'
    ylabel = r'预测精度 ($R^2$)' if lang == 'cn' else r'Prediction Accuracy ($R^2$)'
    
    data = results['window_sensitivity']
    windows = [d['window'] for d in data]
    r2_values = [d['test_r2'] for d in data]
    
    fig, ax = plt.subplots(figsize=(4.8, 4))
    apply_panel_card(ax, facecolor=PANEL_BG['window'])
    
    x = np.asarray(windows, dtype=float)
    y = np.asarray(r2_values, dtype=float)
    best_idx, _ = bubble_plot(ax, x, y, PALETTE['primary'], PALETTE['accent'])
    
    for i, val in enumerate(y):
        weight = 'bold' if i == best_idx else 'normal'
        ax.text(x[i], val + 0.004, f'{val:.3f}', ha='center', fontsize=9, fontweight=weight)
    
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.set_ylabel(ylabel, fontproperties=fp)
    ax.set_xticks(windows)
    ax.set_xticklabels(windows)
    ax.set_xlim([min(x) - 0.4, max(x) + 0.4])
    
    y_min, y_max = float(np.min(y)), float(np.max(y))
    margin = (y_max - y_min) * 0.35 if y_max > y_min else 0.02
    ax.set_ylim([y_min - margin * 0.4, y_max + margin])
    
    setup_ax_style(ax, grid_axis='y')
    plt.tight_layout()
    save_figure_png_and_pdf(output_dir / 'sensitivity_window.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_window.png'}")

# ============================================================================
# 图4: 参数影响力
# ============================================================================
def create_tornado_chart(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    xlabel = r'R$^2$ 变化范围' if lang == 'cn' else r'R$^2$ Range'
    labels = {
        'window': '时间窗口' if lang == 'cn' else 'Time Window',
        'alpha': '阻尼因子' if lang == 'cn' else 'Damping Factor',
        'threshold': '情感阈值' if lang == 'cn' else 'Emotion Threshold',
    }
    
    impacts = []
    for key, data_key in [('alpha', 'alpha_sensitivity'), 
                          ('threshold', 'threshold_sensitivity'),
                          ('window', 'window_sensitivity')]:
        r2s = [d['test_r2'] for d in results[data_key]]
        impacts.append((key, max(r2s) - min(r2s)))
    impacts.sort(key=lambda x: x[1], reverse=True)
    
    fig, ax = plt.subplots(figsize=(5.6, 3))
    apply_panel_card(ax, facecolor=PANEL_BG['impact'])
    
    colors = [PALETTE['primary'], PALETTE['tertiary'], PALETTE['secondary']]
    labels_list = [labels[p[0]] for p in impacts]
    values_list = [p[1] for p in impacts]
    
    dot_rank_plot(ax, labels_list, values_list, colors, fp=fp, show_values=True, text_size=9)
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.set_xlim([0, max(values_list) * 1.2])
    
    setup_ax_style(ax, grid_axis='x')
    plt.tight_layout()
    save_figure_png_and_pdf(output_dir / 'sensitivity_tornado.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_tornado.png'}")

# ============================================================================
# 图5: 热力图
# ============================================================================
def create_heatmap_figure(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    xlabel = r'情感阈值 $\tau$' if lang == 'cn' else r'Emotion Threshold $\tau$'
    ylabel = r'阻尼因子 $\alpha$' if lang == 'cn' else r'Damping Factor $\alpha$'
    
    data = results['alpha_threshold_heatmap']
    alphas = sorted(set(d['alpha'] for d in data))
    thresholds = sorted(set(d['threshold'] for d in data))
    
    matrix = np.zeros((len(alphas), len(thresholds)))
    for d in data:
        matrix[alphas.index(d['alpha']), thresholds.index(d['threshold'])] = d['test_r2']
    
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    apply_panel_card(ax, facecolor=PANEL_BG['heatmap'])
    
    thresholds_arr = np.asarray(thresholds, dtype=float)
    alphas_arr = np.asarray(alphas, dtype=float)
    
    th_grid = np.linspace(thresholds_arr.min(), thresholds_arr.max(), 200)
    alpha_log = np.log10(alphas_arr)
    alpha_grid_log = np.linspace(alpha_log.min(), alpha_log.max(), 200)
    
    interp_rows = np.array([np.interp(th_grid, thresholds_arr, row) for row in matrix])
    interp_grid = np.array([
        np.interp(alpha_grid_log, alpha_log, interp_rows[:, j])
        for j in range(len(th_grid))
    ]).T
    alpha_grid = 10 ** alpha_grid_log
    
    X, Y = np.meshgrid(th_grid, alpha_grid)
    levels = np.linspace(interp_grid.min(), interp_grid.max(), 18)
    cf = ax.contourf(X, Y, interp_grid, levels=levels, cmap='cividis')
    ax.contour(X, Y, interp_grid, levels=6, colors='white', linewidths=0.6, alpha=0.6)
    
    Xg, Yg = np.meshgrid(thresholds_arr, alphas_arr)
    ax.scatter(Xg.flatten(), Yg.flatten(), s=14, color='white', alpha=0.6, edgecolors='none')
    
    best_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
    best_alpha = alphas_arr[best_idx[0]]
    best_threshold = thresholds_arr[best_idx[1]]
    ax.scatter([best_threshold], [best_alpha], s=140, color=PALETTE['accent'],
               edgecolors='white', linewidths=1.5, zorder=4)
    ax.scatter([best_threshold], [best_alpha], s=300, color=PALETTE['accent'],
               alpha=0.18, zorder=3)
    
    cbar = fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.03)
    cbar.set_label('R²', fontproperties=fp)
    
    ax.set_xlabel(xlabel, fontproperties=fp)
    ax.set_ylabel(ylabel, fontproperties=fp)
    ax.set_xticks(thresholds_arr)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds_arr])
    ax.set_yticks(alphas_arr)
    ax.set_yticklabels([f'{a:.3f}' for a in alphas_arr])
    ax.set_yscale('log')
    
    setup_ax_style(ax, grid_axis='none')
    plt.tight_layout()
    save_figure_png_and_pdf(output_dir / 'sensitivity_heatmap.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_heatmap.png'}")

# ============================================================================
# 图6: 参数节点表征面板
# ============================================================================
def create_node_representation_panels(results, lang='en', year=2021, top_k=60):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)

    G_full = load_year_network(year)
    G = select_subgraph(G_full, top_k=top_k)
    G_und = G.to_undirected()
    nodes = list(G.nodes())

    # 固定布局
    pos = nx.kamada_kawai_layout(G_und, scale=1.0)
    xs = np.array([pos[n][0] for n in nodes])
    ys = np.array([pos[n][1] for n in nodes])
    x_margin = (xs.max() - xs.min()) * 0.12 if xs.size > 0 else 0.1
    y_margin = (ys.max() - ys.min()) * 0.12 if ys.size > 0 else 0.1
    xlim = (xs.min() - x_margin, xs.max() + x_margin)
    ylim = (ys.min() - y_margin, ys.max() + y_margin)

    # 节点大小（基于度，范围收敛）
    degree_vals = dict(G_und.degree(weight='weight'))
    size_vals = scale_sizes(np.array([degree_vals.get(n, 1.0) for n in nodes]), min_size=45, max_size=140)
    node_sizes = {n: size_vals[i] for i, n in enumerate(nodes)}

    # 参数设置
    alpha_vals = [0.005, 0.03, 0.2]
    threshold_vals = [0.02, 0.08, 0.15]
    default_alpha = 0.01
    default_threshold = 0.05

    # 静态指标
    betweenness = nx.betweenness_centrality(G_und, weight='weight', normalized=True)
    closeness = nx.closeness_centrality(G_und)
    pagerank = nx.pagerank(G, weight='weight')

    metric_order = ['DomiRank', 'Betweenness', 'Closeness', 'PageRank']
    metric_colors = {
        'DomiRank': PALETTE['primary'],
        'Betweenness': PALETTE['dark'],
        'Closeness': PALETTE['dark'],
        'PageRank': PALETTE['dark'],
    }

    # 计算不同参数下的DomiRank
    domi_alpha = [compute_affective_domirank_scores(G, alpha=a, threshold=default_threshold) for a in alpha_vals]
    domi_tau = [compute_affective_domirank_scores(G, alpha=default_alpha, threshold=t) for t in threshold_vals]
    domi_all = domi_alpha + domi_tau

    norms = {
        'DomiRank': get_norm_from_dict({k: v for scores in domi_all for k, v in scores.items()}),
        'Betweenness': get_norm_from_dict(betweenness),
        'Closeness': get_norm_from_dict(closeness),
        'PageRank': get_norm_from_dict(pagerank),
    }

    cmap = plt.get_cmap('cividis')
    column_bg = ['#EEF3F9', '#F1F6F2', '#F6F2EC']
    panel_labels = [['a', 'b', 'c', 'd'], ['e', 'f', 'g', 'h']]
    row_titles = ['α sensitivity' if lang == 'en' else 'α 敏感性',
                  'τ sensitivity' if lang == 'en' else 'τ 敏感性']

    fig = plt.figure(figsize=(12.8, 7.6))
    outer_gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.1, 1.1, 1.1],
                                wspace=0.22, hspace=0.26)

    # 左侧曲线图
    for row_idx, data_key in enumerate(['alpha_sensitivity', 'threshold_sensitivity']):
        ax_curve = fig.add_subplot(outer_gs[row_idx, 0])
        apply_panel_card(ax_curve, label=panel_labels[row_idx][0], facecolor='#F7F7FB')
        data = results[data_key]
        x_vals = [d['alpha'] for d in data] if data_key == 'alpha_sensitivity' else [d['threshold'] for d in data]
        y_vals = [d['test_r2'] for d in data]
        logx = data_key == 'alpha_sensitivity'
        response_ribbon(ax_curve, x_vals, y_vals, PALETTE['primary'], marker='o', logx=logx)
        ax_curve.set_xlabel(r'$\alpha$' if data_key == 'alpha_sensitivity' else r'$\tau$')
        ax_curve.set_ylabel(r'$R^2$')
        if logx:
            ax_curve.set_xscale('log')
            ax_curve.set_xticks(sorted(x_vals))
            ax_curve.set_xticklabels(format_log_ticks(sorted(x_vals)))
        setup_ax_style(ax_curve, grid_axis='y')
        ax_curve.set_title(row_titles[row_idx], fontsize=9, color=PALETTE['dark'], pad=6, fontproperties=fp)

    # 参数列
    for col_idx in range(3):
        # 添加列背景和标签
        for row_idx in range(2):
            cell = outer_gs[row_idx, col_idx + 1]
            bbox = cell.get_position(fig)
            rect = FancyBboxPatch(
                (bbox.x0, bbox.y0), bbox.width, bbox.height,
                transform=fig.transFigure,
                boxstyle='round,pad=0.01,rounding_size=0.02',
                linewidth=0, facecolor=column_bg[col_idx], zorder=-5, clip_on=False
            )
            fig.add_artist(rect)
            fig.text(bbox.x0 + 0.006, bbox.y1 - 0.02,
                     f'({panel_labels[row_idx][col_idx + 1]})',
                     ha='left', va='top', fontsize=10, fontweight='bold',
                     color=PALETTE['dark'])

        # row 0: alpha sweep
        alpha = alpha_vals[col_idx]
        threshold = default_threshold
        domi_scores = domi_alpha[col_idx]
        label_text = f'α={alpha:.3f}, τ={threshold:.2f}'

        # row 1: threshold sweep
        alpha_tau = default_alpha
        threshold_tau = threshold_vals[col_idx]
        domi_scores_tau = domi_tau[col_idx]
        label_text_tau = f'α={alpha_tau:.2f}, τ={threshold_tau:.2f}'

        for row_idx, domi_scores_use, label_text_use in [
            (0, domi_scores, label_text),
            (1, domi_scores_tau, label_text_tau)
        ]:
            sub_gs = outer_gs[row_idx, col_idx + 1].subgridspec(2, 2, wspace=0.05, hspace=0.05)
            metrics = {
                'DomiRank': domi_scores_use,
                'Betweenness': betweenness,
                'Closeness': closeness,
                'PageRank': pagerank,
            }

            for idx, metric in enumerate(metric_order):
                ax = fig.add_subplot(sub_gs[idx // 2, idx % 2])
                ax.set_axis_off()
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)

                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edge_color=PALETTE['light'], alpha=0.45, width=0.6
                )

                values = metrics[metric]
                node_colors = [values[n] for n in nodes]
                norm = norms[metric]
                nx.draw_networkx_nodes(
                    G, pos, ax=ax,
                    node_color=node_colors, cmap=cmap, vmin=norm.vmin, vmax=norm.vmax,
                    node_size=[node_sizes[n] for n in nodes],
                    linewidths=0.4, edgecolors='white', alpha=0.95
                )

                ax.text(0.02, 0.98, metric, transform=ax.transAxes,
                        ha='left', va='top', fontsize=8,
                        color=metric_colors[metric], fontproperties=fp)

            # 参数标签
            fig.text(
                outer_gs[row_idx, col_idx + 1].get_position(fig).x1 - 0.008,
                outer_gs[row_idx, col_idx + 1].get_position(fig).y0 + 0.012,
                label_text_use, ha='right', va='bottom', fontsize=8,
                color='#C44E3B', fontproperties=fp
            )

    save_figure_png_and_pdf(output_dir / 'sensitivity_node_panels.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_node_panels.png'}")

# ============================================================================
# 综合图
# ============================================================================
def create_combined_figure(results, lang='en'):
    setup_style()
    fp = get_font_prop(lang)
    output_dir = Path('figures_cn' if lang == 'cn' else 'figures')
    output_dir.mkdir(exist_ok=True)
    
    labels = {
        'alpha_x': r'阻尼因子 $\alpha$' if lang == 'cn' else r'Damping Factor $\alpha$',
        'threshold_x': r'情感阈值 $\tau$' if lang == 'cn' else r'Emotion Threshold $\tau$',
        'window_x': '时间窗口 (年)' if lang == 'cn' else 'Time Window (Years)',
        'y': r'预测精度 ($R^2$)' if lang == 'cn' else r'Prediction Accuracy ($R^2$)',
        'range_x': r'R$^2$ 变化范围' if lang == 'cn' else r'R$^2$ Range',
        'window': '时间窗口' if lang == 'cn' else 'Time Window',
        'alpha': '阻尼因子' if lang == 'cn' else 'Damping Factor',
        'threshold': '情感阈值' if lang == 'cn' else 'Emotion Threshold',
    }
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.3)
    
    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    apply_panel_card(ax1, label='a', facecolor=PANEL_BG['alpha'])
    alpha_data = results['alpha_sensitivity']
    alphas = [d['alpha'] for d in alpha_data]
    alpha_r2 = [d['test_r2'] for d in alpha_data]
    response_ribbon(ax1, alphas, alpha_r2, PALETTE['primary'], marker='o', logx=True)
    ax1.set_xlabel(labels['alpha_x'], fontsize=9, fontproperties=fp)
    ax1.set_ylabel(labels['y'], fontsize=9, fontproperties=fp)
    ax1.set_xscale('log')
    ax1.set_xticks(sorted(alphas))
    ax1.set_xticklabels(format_log_ticks(sorted(alphas)))
    setup_ax_style(ax1, grid_axis='y')
    
    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    apply_panel_card(ax2, label='b', facecolor=PANEL_BG['threshold'])
    threshold_data = results['threshold_sensitivity']
    thresholds = [d['threshold'] for d in threshold_data]
    threshold_r2 = [d['test_r2'] for d in threshold_data]
    response_ribbon(ax2, thresholds, threshold_r2, PALETTE['tertiary'], marker='s', logx=False)
    ax2.set_xlabel(labels['threshold_x'], fontsize=9, fontproperties=fp)
    ax2.set_ylabel(labels['y'], fontsize=9, fontproperties=fp)
    setup_ax_style(ax2, grid_axis='y')
    
    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    apply_panel_card(ax3, label='c', facecolor=PANEL_BG['window'])
    window_data = results['window_sensitivity']
    windows = [d['window'] for d in window_data]
    window_r2 = [d['test_r2'] for d in window_data]
    x = np.asarray(windows, dtype=float)
    y = np.asarray(window_r2, dtype=float)
    best_idx, _ = bubble_plot(ax3, x, y, PALETTE['primary'], PALETTE['accent'])
    for i, val in enumerate(y):
        ax3.text(x[i], val + 0.003, f'{val:.3f}', ha='center', fontsize=8,
                 fontweight='bold' if i == best_idx else 'normal')
    ax3.set_xlabel(labels['window_x'], fontsize=9, fontproperties=fp)
    ax3.set_ylabel(labels['y'], fontsize=9, fontproperties=fp)
    ax3.set_xticks(windows)
    ax3.set_xticklabels(windows)
    y_min, y_max = float(np.min(y)), float(np.max(y))
    margin = (y_max - y_min) * 0.35 if y_max > y_min else 0.02
    ax3.set_ylim([y_min - margin * 0.4, y_max + margin])
    ax3.set_xlim([min(x) - 0.4, max(x) + 0.4])
    setup_ax_style(ax3, grid_axis='y')
    
    # Panel D
    ax4 = fig.add_subplot(gs[1, 1])
    apply_panel_card(ax4, label='d', facecolor=PANEL_BG['impact'])
    impacts = [
        ('alpha', max(alpha_r2) - min(alpha_r2)),
        ('threshold', max(threshold_r2) - min(threshold_r2)),
        ('window', max(window_r2) - min(window_r2)),
    ]
    impacts.sort(key=lambda x: x[1], reverse=True)
    labels_list = [labels[p[0]] for p in impacts]
    values_list = [p[1] for p in impacts]
    colors = [PALETTE['primary'], PALETTE['tertiary'], PALETTE['secondary']]
    dot_rank_plot(ax4, labels_list, values_list, colors, fp=fp, show_values=True, text_size=8)
    ax4.set_xlabel(labels['range_x'], fontsize=9, fontproperties=fp)
    ax4.set_xlim([0, max(values_list) * 1.2])
    setup_ax_style(ax4, grid_axis='x')
    
    save_figure_png_and_pdf(output_dir / 'sensitivity_combined.png', dpi_png=300, dpi_pdf=400)
    plt.close()
    print(f"  [OK] {output_dir / 'sensitivity_combined.png'}")

# ============================================================================
# 工具函数
# ============================================================================
def setup_style():
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'axes.linewidth': 0.8,
        'axes.edgecolor': PALETTE['light'],
        'axes.labelcolor': PALETTE['dark'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'xtick.color': PALETTE['dark'],
        'ytick.color': PALETTE['dark'],
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'legend.fontsize': 8,
        'legend.frameon': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.dpi': 300,
        'axes.grid': False,
    })

# ============================================================================
# 主函数
# ============================================================================
def main():
    print("=" * 50)
    print("敏感性分析可视化 - 顶级期刊风格")
    print("=" * 50)
    
    results_file = Path('experiments/sensitivity_results_full.pkl')
    if not results_file.exists():
        print(f"错误: 找不到 {results_file}")
        return
    
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    
    print("\n生成英文版本...")
    create_alpha_sensitivity_figure(results, 'en')
    create_threshold_sensitivity_figure(results, 'en')
    create_window_sensitivity_figure(results, 'en')
    create_tornado_chart(results, 'en')
    create_heatmap_figure(results, 'en')
    create_combined_figure(results, 'en')
    create_node_representation_panels(results, 'en')
    
    print("\n生成中文版本...")
    create_alpha_sensitivity_figure(results, 'cn')
    create_threshold_sensitivity_figure(results, 'cn')
    create_window_sensitivity_figure(results, 'cn')
    create_tornado_chart(results, 'cn')
    create_heatmap_figure(results, 'cn')
    create_combined_figure(results, 'cn')
    create_node_representation_panels(results, 'cn')
    
    print("\n完成!")

if __name__ == "__main__":
    main()
