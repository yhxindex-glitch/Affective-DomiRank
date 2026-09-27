"""
传统机器学习Baseline对比实验
目标：证明深度学习（STGNN）相比传统ML方法的优势

包含模型：
1. XGBoost - 梯度提升树（工业界标准）
2. Random Forest - 随机森林（鲁棒性好）
3. LightGBM - 轻量级梯度提升（速度快）
"""

import pickle
import numpy as np
import torch
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm

# 导入exp4的数据集类
import importlib.util
spec = importlib.util.spec_from_file_location("exp4", "experiments/exp4_stgnn_temporal.py")
exp4_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp4_module)
NodeLevelSTGNNDataset = exp4_module.NodeLevelSTGNNDataset


def prepare_ml_data(dataset):
    """
    准备传统ML模型的数据
    将时序图数据转换为扁平特征向量
    
    Args:
        dataset: NodeLevelSTGNNDataset实例
    
    Returns:
        X: [num_samples, feature_dim] - 扁平化的特征
        y: [num_samples] - 标签
        node_names: [num_samples] - 节点名称（用于分析）
    """
    print("\n准备传统ML数据...")
    X = []
    y = []
    node_names = []
    
    for sample in tqdm(dataset.samples, desc="  转换样本"):
        # 提取节点的时序特征 [seq_len, feature_dim]
        node_features_seq = sample['node_features_seq'].numpy()  # [3, 16]
        
        # 方案1：展平时序特征 [seq_len * feature_dim]
        # 这样保留了所有时序信息
        flat_features = node_features_seq.flatten()  # [48] = 3*16
        
        # 方案2（可选）：添加统计特征
        # 计算时序维度的统计量
        temporal_stats = []
        for feat_idx in range(node_features_seq.shape[1]):
            feat_values = node_features_seq[:, feat_idx]
            temporal_stats.extend([
                feat_values.mean(),   # 均值
                feat_values.std(),    # 标准差
                feat_values.max() - feat_values.min()  # 范围
            ])
        temporal_stats = np.array(temporal_stats)  # [48] = 16*3
        
        # 方案3：结合原始序列和统计特征
        combined_features = np.concatenate([flat_features, temporal_stats])  # [96]
        
        X.append(combined_features)
        # 🔥 关键修复：使用归一化后的标签（与深度学习一致）
        normalized_label = sample['label'].item()  # 已经是归一化的
        y.append(normalized_label)
        node_names.append(sample['node'])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  ✓ 特征矩阵: {X.shape}")
    print(f"  ✓ 标签向量: {y.shape}")
    print(f"  特征统计: mean={X.mean():.4f}, std={X.std():.4f}")
    print(f"  标签统计: mean={y.mean():.4f}, std={y.std():.4f}")
    
    return X, y, node_names


def denormalize_labels(normalized_labels, dataset):
    """
    反归一化标签（与深度学习保持一致）
    
    Args:
        normalized_labels: 归一化后的标签
        dataset: NodeLevelSTGNNDataset实例
    
    Returns:
        反归一化后的标签
    """
    return normalized_labels * dataset.label_std + dataset.label_mean


def train_xgboost(X_train, y_train, X_test, y_test, dataset):
    """
    训练XGBoost模型
    """
    print("\n" + "="*80)
    print("1. XGBoost Baseline")
    print("="*80)
    print("  论文: XGBoost: A Scalable Tree Boosting System (KDD 2016)")
    print("  特点: 梯度提升决策树，供应链风险预测的工业标准")
    
    # 参数调优（针对回归任务）
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    print(f"  参数配置:")
    print(f"    - max_depth: {params['max_depth']}")
    print(f"    - learning_rate: {params['learning_rate']}")
    print(f"    - n_estimators: {params['n_estimators']}")
    
    # 训练模型（带early stopping）
    model = xgb.XGBRegressor(**params)
    
    print("\n  开始训练...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    print(f"  ✓ 训练完成！")
    
    # 预测
    train_pred_norm = model.predict(X_train)
    test_pred_norm = model.predict(X_test)
    
    # 🔥 关键修复：反归一化预测结果（与深度学习保持一致）
    train_pred = denormalize_labels(train_pred_norm, dataset)
    test_pred = denormalize_labels(test_pred_norm, dataset)
    y_train_denorm = denormalize_labels(y_train, dataset)
    y_test_denorm = denormalize_labels(y_test, dataset)
    
    # 评估（使用反归一化后的数据）
    train_mae = mean_absolute_error(y_train_denorm, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_denorm, train_pred))
    train_r2 = r2_score(y_train_denorm, train_pred)
    
    test_mae = mean_absolute_error(y_test_denorm, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_denorm, test_pred))
    test_r2 = r2_score(y_test_denorm, test_pred)
    
    print(f"\n  训练集性能:")
    print(f"    MAE:  {train_mae:.4f}")
    print(f"    RMSE: {train_rmse:.4f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"\n  测试集性能:")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    
    # 特征重要性（Top 10）
    feature_importance = model.feature_importances_
    top_indices = np.argsort(feature_importance)[-10:][::-1]
    print(f"\n  Top-10 重要特征:")
    for i, idx in enumerate(top_indices, 1):
        print(f"    {i}. 特征{idx}: {feature_importance[idx]:.4f}")
    
    return {
        'model': model,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_pred': test_pred
    }


def train_random_forest(X_train, y_train, X_test, y_test, dataset):
    """
    训练Random Forest模型
    """
    print("\n" + "="*80)
    print("2. Random Forest Baseline")
    print("="*80)
    print("  论文: Random Forests (Machine Learning 2001)")
    print("  特点: 集成学习，鲁棒性强，不易过拟合")
    
    params = {
        'n_estimators': 200,
        'max_depth': 15,
        'min_samples_split': 10,
        'min_samples_leaf': 4,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }
    
    print(f"  参数配置:")
    print(f"    - n_estimators: {params['n_estimators']}")
    print(f"    - max_depth: {params['max_depth']}")
    print(f"    - max_features: {params['max_features']}")
    
    print("\n  开始训练...")
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    print(f"  ✓ 训练完成！")
    
    # 预测
    train_pred_norm = model.predict(X_train)
    test_pred_norm = model.predict(X_test)
    
    # 🔥 关键修复：反归一化预测结果（与深度学习保持一致）
    train_pred = denormalize_labels(train_pred_norm, dataset)
    test_pred = denormalize_labels(test_pred_norm, dataset)
    y_train_denorm = denormalize_labels(y_train, dataset)
    y_test_denorm = denormalize_labels(y_test, dataset)
    
    # 评估（使用反归一化后的数据）
    train_mae = mean_absolute_error(y_train_denorm, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_denorm, train_pred))
    train_r2 = r2_score(y_train_denorm, train_pred)
    
    test_mae = mean_absolute_error(y_test_denorm, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_denorm, test_pred))
    test_r2 = r2_score(y_test_denorm, test_pred)
    
    print(f"\n  训练集性能:")
    print(f"    MAE:  {train_mae:.4f}")
    print(f"    RMSE: {train_rmse:.4f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"\n  测试集性能:")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    
    return {
        'model': model,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_pred': test_pred
    }


def train_lightgbm(X_train, y_train, X_test, y_test, dataset):
    """
    训练LightGBM模型
    """
    print("\n" + "="*80)
    print("3. LightGBM Baseline")
    print("="*80)
    print("  论文: LightGBM: A Highly Efficient Gradient Boosting Decision Tree (NIPS 2017)")
    print("  特点: 轻量级梯度提升，训练速度快，内存占用小")
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 20,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': -1
    }
    
    print(f"  参数配置:")
    print(f"    - num_leaves: {params['num_leaves']}")
    print(f"    - learning_rate: {params['learning_rate']}")
    print(f"    - n_estimators: {params['n_estimators']}")
    
    print("\n  开始训练...")
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='rmse'
    )
    print(f"  ✓ 训练完成！")
    
    # 预测
    train_pred_norm = model.predict(X_train)
    test_pred_norm = model.predict(X_test)
    
    #  关键修复：反归一化预测结果（与深度学习保持一致）
    train_pred = denormalize_labels(train_pred_norm, dataset)
    test_pred = denormalize_labels(test_pred_norm, dataset)
    y_train_denorm = denormalize_labels(y_train, dataset)
    y_test_denorm = denormalize_labels(y_test, dataset)
    
    # 评估（使用反归一化后的数据）
    train_mae = mean_absolute_error(y_train_denorm, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train_denorm, train_pred))
    train_r2 = r2_score(y_train_denorm, train_pred)
    
    test_mae = mean_absolute_error(y_test_denorm, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test_denorm, test_pred))
    test_r2 = r2_score(y_test_denorm, test_pred)
    
    print(f"\n  训练集性能:")
    print(f"    MAE:  {train_mae:.4f}")
    print(f"    RMSE: {train_rmse:.4f}")
    print(f"    R²:   {train_r2:.4f}")
    
    print(f"\n  测试集性能:")
    print(f"    MAE:  {test_mae:.4f}")
    print(f"    RMSE: {test_rmse:.4f}")
    print(f"    R²:   {test_r2:.4f}")
    
    return {
        'model': model,
        'train_mae': train_mae,
        'train_r2': train_r2,
        'test_mae': test_mae,
        'test_r2': test_r2,
        'test_pred': test_pred
    }


def main():
    print("="*80)
    print("传统机器学习Baseline对比实验")
    print("="*80)
    print("\n对比模型:")
    print("  1. XGBoost - 梯度提升树（工业界标准）")
    print("  2. Random Forest - 随机森林（鲁棒性好）")
    print("  3. LightGBM - 轻量级梯度提升（速度快）")
    
    # 加载数据
    print("\n" + "="*80)
    print("加载数据...")
    print("="*80)
    
    with open('data/processed/networks/all_networks_2017_2021.pkl', 'rb') as f:
        networks = pickle.load(f)
    
    print("构建数据集（与exp4一致）...")
    dataset = NodeLevelSTGNNDataset(
        networks=networks,
        window_size=3,
        use_interpolation=True,
        risk_labels_path='data/processed/networks/risk_labels.pkl'
    )
    
    # 准备ML数据
    X, y, node_names = prepare_ml_data(dataset)
    
    # 划分训练/测试集（与exp4一致：80/20随机划分）
    print("\n划分训练/测试集...")
    np.random.seed(42)
    indices = np.random.permutation(len(X))
    split_idx = int(0.8 * len(X))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    from training.splits import assert_disjoint_indices
    assert_disjoint_indices(train_indices, test_indices, len(X))
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"  训练样本: {len(X_train)}")
    print(f"  测试样本: {len(X_test)}")
    
    # 训练各个模型
    results = {}
    
    # 1. XGBoost
    xgb_results = train_xgboost(X_train, y_train, X_test, y_test, dataset)
    results['XGBoost'] = xgb_results
    
    # 2. Random Forest
    rf_results = train_random_forest(X_train, y_train, X_test, y_test, dataset)
    results['Random Forest'] = rf_results
    
    # 3. LightGBM
    lgb_results = train_lightgbm(X_train, y_train, X_test, y_test, dataset)
    results['LightGBM'] = lgb_results
    
    # 汇总结果
    print("\n" + "="*80)
    print("传统机器学习方法对比结果汇总")
    print("="*80)
    print(f"\n{'方法':<20} | {'Train R²':>10} | {'Test R²':>10} | {'Test MAE':>10} | {'说明':<30}")
    print("-"*95)
    
    for method_name, res in results.items():
        description = {
            'XGBoost': '工业标准，树提升',
            'Random Forest': '集成学习，鲁棒',
            'LightGBM': '轻量快速'
        }.get(method_name, '')
        
        print(f"{method_name:<20} | {res['train_r2']:>10.4f} | {res['test_r2']:>10.4f} | {res['test_mae']:>10.4f} | {description:<30}")
    print(f"\n传统ML最佳结果:")
    best_ml_method = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"  {best_ml_method[0]}:  Test R² = {best_ml_method[1]['test_r2']:.4f}, MAE = {best_ml_method[1]['test_mae']:.4f}")
    
    


if __name__ == "__main__":
    main()

