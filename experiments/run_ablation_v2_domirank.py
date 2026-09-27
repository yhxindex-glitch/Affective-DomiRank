"""
消融实验 - 版本2：使用传统DomiRank替代Affective DomiRank
"""
# 复制exp4的代码，只修改affective_domirank那一行
import sys
from pathlib import Path

# 读取exp4代码
with open('experiments/exp4_stgnn_temporal.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 替换affective_vulnerability为traditional_vulnerability - 修正缩进
# 先替换函数调用
code = code.replace(
    "affective_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['affective_domirank'], \n                        is_affective=True\n                    )",
    "traditional_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['domirank'], \n                        is_affective=False\n                    )  # 使用传统DomiRank"
)
# 再替换特征名
code = code.replace(
    "                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）",
    "                        traditional_vulnerability,  # 传统DomiRank脆弱性"
)

# 修改标题
code = code.replace(
    '实验4: STGNN节点级时序预测（情感风险预测版）',
    '消融实验 - 版本2：传统DomiRank（风险预测任务）'
)

# 修改模型保存路径（避免冲突）
code = code.replace(
    "'experiments/best_stgnn_node_level.pt'",
    "'experiments/best_stgnn_v2_domirank.pt'"
)

# 执行修改后的代码
exec(code)

