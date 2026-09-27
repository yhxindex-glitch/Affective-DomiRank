"""
消融实验 - 版本4：纯拓扑（去掉所有情感特征）
"""
# 复制exp4的代码，去掉所有情感相关特征
import sys
from pathlib import Path

# 读取exp4代码
with open('experiments/exp4_stgnn_temporal.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 去掉sentiment_mean (注释掉整行) - 修正缩进为24个空格
code = code.replace(
    "                        cached_features['sentiment_mean'],\n",
    ""
)

# 去掉news_count (也是情感相关)
code = code.replace(
    "                        cached_features['news_count'],\n",
    ""
)

# 去掉affective_vulnerability计算（情感驱动的DomiRank）
code = code.replace(
    "                    # 【方案A】将Affective DomiRank转换为脆弱性指数\n                    affective_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['affective_domirank'], \n                        is_affective=True\n                    )\n                    \n",
    "                    # V4：不使用情感驱动的DomiRank\n                    \n"
)

# 去掉affective_vulnerability特征
code = code.replace(
    "                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）\n",
    ""
)

# 去掉sentiment_change_rate
code = code.replace(
    "                        sentiment_change_rate,\n",
    ""
)

# 去掉sentiment_risk_score
code = code.replace(
    "                        sentiment_risk_score,\n",
    ""
)

# 修改特征维度 16 -> 11
# 去掉: sentiment_mean, news_count, affective_domirank, sentiment_change_rate, sentiment_risk_score = 5个
# 16 - 5 = 11
code = code.replace(
    "node_feature_dim=16,",
    "node_feature_dim=11,  # 去掉5个情感特征"
)

# 修改标题
code = code.replace(
    '实验4: STGNN节点级时序预测（情感风险预测版）',
    '消融实验 - 版本4：纯拓扑（无情感，风险预测任务）'
)

# 修改模型保存路径
code = code.replace(
    "'experiments/best_stgnn_node_level.pt'",
    "'experiments/best_stgnn_v4_no_sentiment.pt'"
)

# 执行修改后的代码
exec(code)

