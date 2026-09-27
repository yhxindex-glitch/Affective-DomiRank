"""
评估器实现
计算各种评估指标
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, mean_squared_error, mean_absolute_error, r2_score
)
from typing import Dict, List
from torch.utils.data import DataLoader


class Evaluator:
    """模型评估器"""
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        """
        初始化评估器
        
        参数:
            model: 要评估的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader: DataLoader,
        task: str = 'binary_classification'
    ) -> Dict[str, float]:
        """
        评估模型
        
        参数:
            data_loader: 数据加载器
            task: 任务类型
            
        返回:
            评估指标字典
        """
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                node_features, edge_index, edge_weight, target = batch
                
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(node_features, edge_index, edge_weight)
                
                if task == 'binary_classification':
                    # 二分类
                    probs = torch.sigmoid(output)
                    preds = (probs > 0.5).float()
                    
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                else:
                    # 回归
                    all_preds.extend(output.cpu().numpy())
                
                all_targets.extend(target.cpu().numpy())
        
        # 转换为numpy数组
        all_preds = np.array(all_preds).flatten()
        all_targets = np.array(all_targets).flatten()
        
        # 计算指标
        if task == 'binary_classification':
            all_probs = np.array(all_probs).flatten()
            
            metrics = {
                'accuracy': accuracy_score(all_targets, all_preds),
                'precision': precision_score(all_targets, all_preds, zero_division=0),
                'recall': recall_score(all_targets, all_preds, zero_division=0),
                'f1_score': f1_score(all_targets, all_preds, zero_division=0),
            }
            
            # ROC-AUC（如果有正负样本）
            if len(np.unique(all_targets)) > 1:
                metrics['roc_auc'] = roc_auc_score(all_targets, all_probs)
            else:
                metrics['roc_auc'] = 0.0
        
        else:
            # 回归指标
            metrics = {
                'mse': mean_squared_error(all_targets, all_preds),
                'mae': mean_absolute_error(all_targets, all_preds),
                'r2': r2_score(all_targets, all_preds),
            }
        
        return metrics
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        模型预测
        
        参数:
            data_loader: 数据加载器
            
        返回:
            预测结果数组
        """
        all_preds = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in data_loader:
                node_features, edge_index, edge_weight, _ = batch
                
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                
                output = self.model(node_features, edge_index, edge_weight)
                probs = torch.sigmoid(output)
                
                all_preds.extend(probs.cpu().numpy())
        
        return np.array(all_preds).flatten()
    
    def compare_models(
        self,
        models_dict: Dict[str, torch.nn.Module],
        test_loader: DataLoader
    ) -> Dict[str, Dict[str, float]]:
        """
        对比多个模型的性能
        
        参数:
            models_dict: 模型字典 {model_name: model}
            test_loader: 测试数据加载器
            
        返回:
            {model_name: metrics}字典
        """
        results = {}
        
        print("\n" + "=" * 80)
        print("模型性能对比")
        print("=" * 80)
        
        for model_name, model in models_dict.items():
            self.model = model.to(self.device)
            metrics = self.evaluate(test_loader)
            results[model_name] = metrics
            
            print(f"\n{model_name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")
        
        return results

