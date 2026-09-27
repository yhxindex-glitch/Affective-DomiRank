"""
消融实验 - 版本3：去掉所有DomiRank特征（affective和传统）
"""
# 复制exp4的代码，注释掉affective_domirank那一行
import sys
from pathlib import Path

# 读取exp4代码
with open('experiments/exp4_stgnn_temporal.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 注释掉affective_vulnerability计算和使用 - 修正缩进
# 先注释掉计算部分
code = code.replace(
    "                    # 【方案A】将Affective DomiRank转换为脆弱性指数\n                    affective_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['affective_domirank'], \n                        is_affective=True\n                    )",
    "                    # 【方案A】将Affective DomiRank转换为脆弱性指数\n                    # affective_vulnerability = self._domirank_to_vulnerability(\n                    #     cached_features['affective_domirank'], \n                    #     is_affective=True\n                    # )  # 版本3：不使用DomiRank"
)
# 再注释掉特征使用
code = code.replace(
    "                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）",
    "                        # affective_vulnerability,  # 版本3：不使用DomiRank"
)

# 修改特征维度 16 -> 15
code = code.replace(
    "node_feature_dim=16,",
    "node_feature_dim=15,  # 去掉1个DomiRank特征"
)

# 修改标题
code = code.replace(
    '实验4: STGNN节点级时序预测（情感风险预测版）',
    '消融实验 - 版本3：无DomiRank特征（风险预测任务）'
)

# 修改模型保存路径
code = code.replace(
    "'experiments/best_stgnn_node_level.pt'",
    "'experiments/best_stgnn_v3_no_domirank.pt'"
)

# 执行修改后的代码
exec(code)

