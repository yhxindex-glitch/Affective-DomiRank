"""
实验结果可视化
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List


def plot_training_history(
    history: Dict[str, List[float]],
    filename: str = "results/figures/training_history.png"
):
    """
    绘制训练历史
    
    参数:
        history: 训练历史字典
        filename: 保存文件名
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    if 'train_loss' in history:
        ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 准确率曲线
    if 'train_acc' in history:
        ax2.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history:
        ax2.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 训练历史图已保存: {filename}")


def plot_comparison(
    results_dict: Dict[str, Dict[str, float]],
    filename: str = "results/figures/model_comparison.png"
):
    """
    绘制模型对比图
    
    参数:
        results_dict: {model_name: {metric: value}}
        filename: 保存文件名
    """
    # 转换为DataFrame
    df = pd.DataFrame(results_dict).T
    
    fig, axes = plt.subplots(1, len(df.columns), figsize=(5*len(df.columns), 5))
    
    if len(df.columns) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(df.columns):
        values = df[metric].values
        models = df.index.tolist()
        
        axes[idx].bar(models, values, color='steelblue', alpha=0.8)
        axes[idx].set_ylabel(metric.upper(), fontsize=12)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 模型对比图已保存: {filename}")


def plot_ablation_study(
    ablation_results: Dict[str, float],
    baseline_score: float,
    filename: str = "results/figures/ablation_study.png"
):
    """
    绘制消融实验结果
    
    参数:
        ablation_results: 消融实验结果 {variant_name: score}
        baseline_score: 完整模型得分
        filename: 文件名
    """
    plt.figure(figsize=(10, 6))
    
    # 添加完整模型
    variants = ['Full Model'] + list(ablation_results.keys())
    scores = [baseline_score] + list(ablation_results.values())
    
    # 颜色（完整模型用绿色，其他用灰色）
    colors = ['green'] + ['gray'] * len(ablation_results)
    
    bars = plt.barh(variants, scores, color=colors, alpha=0.7)
    
    # 添加数值标签
    for i, (variant, score) in enumerate(zip(variants, scores)):
        plt.text(score + 0.01, i, f'{score:.3f}', va='center')
    
    plt.xlabel('F1 Score', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ 消融实验图已保存: {filename}")

