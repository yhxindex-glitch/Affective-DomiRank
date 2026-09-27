"""
训练器实现
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import numpy as np


class Trainer:
    """STGNN模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = 'results/checkpoints'
    ):
        """
        初始化训练器
        
        参数:
            model: 要训练的模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            learning_rate: 学习率
            device: 计算设备
            checkpoint_dir: 检查点保存目录
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 优化器
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # 损失函数（二分类）
        self.criterion = nn.BCEWithLogitsLoss()
        
        # 检查点目录
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        返回:
            训练指标字典
        """
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            node_features, edge_index, edge_weight, target = batch
            
            # 移动到设备
            node_features = node_features.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_weight = edge_weight.to(self.device)
            target = target.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            output = self.model(node_features, edge_index, edge_weight)
            
            # 计算损失
            loss = self.criterion(output, target)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            
            # 计算准确率
            pred = (torch.sigmoid(output) > 0.5).float()
            correct += (pred == target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def validate(self) -> Dict[str, float]:
        """
        验证模型
        
        返回:
            验证指标字典
        """
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                node_features, edge_index, edge_weight, target = batch
                
                node_features = node_features.to(self.device)
                edge_index = edge_index.to(self.device)
                edge_weight = edge_weight.to(self.device)
                target = target.to(self.device)
                
                # 前向传播
                output = self.model(node_features, edge_index, edge_weight)
                loss = self.criterion(output, target)
                
                # 统计
                total_loss += loss.item()
                pred = (torch.sigmoid(output) > 0.5).float()
                correct += (pred == target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total if total > 0 else 0.0
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def train(
        self,
        epochs: int = 200,
        early_stopping_patience: int = 20,
        save_best: bool = True
    ):
        """
        训练模型
        
        参数:
            epochs: 训练轮数
            early_stopping_patience: 早停耐心值
            save_best: 是否保存最佳模型
        """
        print("\n" + "=" * 80)
        print("开始训练STGNN模型")
        print("=" * 80)
        print(f"  训练轮数: {epochs}")
        print(f"  设备: {self.device}")
        print(f"  学习率: {self.optimizer.param_groups[0]['lr']}")
        print()
        
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self.train_epoch()
            
            # 验证
            val_metrics = self.validate()
            
            # 记录历史
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['accuracy'])
            
            if val_metrics:
                self.history['val_loss'].append(val_metrics['loss'])
                self.history['val_acc'].append(val_metrics['accuracy'])
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs}")
                print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
                if val_metrics:
                    print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
            
            # 学习率调度
            if val_metrics:
                self.scheduler.step(val_metrics['loss'])
                
                # 早停检查
                if val_metrics['loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    
                    if save_best:
                        self.save_checkpoint(epoch, is_best=True)
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\n早停于epoch {epoch+1}")
                    break
        
        print("\n✓ 训练完成!")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点
        
        参数:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, path)
            print(f"    ✓ 保存最佳模型: {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        参数:
            checkpoint_path: 检查点文件路径
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        print(f"✓ 加载检查点: {checkpoint_path}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")

