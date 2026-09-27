"""
乘性情感门控版本的多随机种子实验脚本。

用途：
1. 复用 exp4_stgnn_temporal.py 的主训练逻辑；
2. 将 Affective DomiRank 的加性情感项替换为乘性门控版本；
3. 仅运行少量 seeds（默认 42/52/62）做 Reviewer 2 Comment 7 的对照实验。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any


HELPER_CODE = r"""
import copy
import networkx as nx


class MultiplicativeGatedDomiRank:
    # 乘性门控版本：
    # dΓ_i/dt = α * σ(e_i) * Σ_j(θ - Γ_j) - β * Γ_i
    # 其中 σ(e_i) 根据节点自身情感极性调制结构传播项。

    def __init__(
        self,
        G: nx.DiGraph,
        emotion_vectors,
        alpha: float = 0.01,
        beta: float = 0.5,
        theta: float = 1.0,
    ):
        self.G = G
        self.emotion_vectors = emotion_vectors
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.sigma_values = self._compute_all_sigma()

    def _get_sentiment(self, node):
        emotion = self.emotion_vectors.get(node, {})
        if isinstance(emotion, dict):
            return float(emotion.get('sentiment_polarity', 0.0))
        if hasattr(emotion, '__len__') and len(emotion) > 0:
            return float(emotion[0])
        return 0.0

    def _compute_sigma(self, sentiment: float) -> float:
        if abs(sentiment) < 0.05:
            return 1.0
        if sentiment < 0:
            return 1.0 + abs(sentiment) ** 0.8
        return 1.0 - 0.7 * (sentiment ** 0.8)

    def _compute_all_sigma(self):
        sigma_dict = {}
        for node in self.nodes:
            sigma_dict[node] = self._compute_sigma(self._get_sentiment(node))
        return sigma_dict

    def _dgamma_dt(self, gamma):
        dgamma = np.zeros(self.n)
        for i, node in enumerate(self.nodes):
            neighbor_sum = 0.0
            for neighbor in self.G.predecessors(node):
                j = self.node_to_idx[neighbor]
                neighbor_sum += (self.theta - gamma[j])
            sigma = self.sigma_values[node]
            dgamma[i] = self.alpha * sigma * neighbor_sum - self.beta * gamma[i]
        return dgamma

    def compute(self, max_iter=5000, tol=1e-6, time_step=0.01):
        gamma = np.ones(self.n)
        for _ in range(max_iter):
            dgamma = self._dgamma_dt(gamma)
            gamma_new = gamma + time_step * dgamma
            if np.linalg.norm(gamma_new - gamma) < tol:
                return {node: gamma_new[i] for i, node in enumerate(self.nodes)}
            gamma = gamma_new
        return {node: gamma[i] for i, node in enumerate(self.nodes)}


def recompute_multiplicative_domirank(networks):
    updated_networks = {}
    for year, G in networks.items():
        G_new = copy.deepcopy(G)
        emotion_vectors = {}
        for node in G_new.nodes():
            sentiment_mean = G_new.nodes[node].get('sentiment_mean', 0.0)
            emotion_vectors[node] = {
                'sentiment_polarity': sentiment_mean,
                'sentiment_intensity': abs(sentiment_mean),
                'fear': max(-sentiment_mean, 0),
                'trust': max(sentiment_mean, 0),
                'anger': 0,
                'joy': 0,
                'sadness': 0,
            }

        multiplicative_dr = MultiplicativeGatedDomiRank(
            G_new,
            emotion_vectors,
            alpha=0.01,
            beta=0.5,
            theta=1.0,
        )
        scores = multiplicative_dr.compute(max_iter=5000, tol=1e-6)

        for node, score in scores.items():
            G_new.nodes[node]['multiplicative_domirank'] = float(score)

        for node in G_new.nodes():
            predecessors = list(G_new.predecessors(node))
            if predecessors:
                vals = [G_new.nodes[p].get('multiplicative_domirank', 0.0) for p in predecessors]
                G_new.nodes[node]['neighbor_avg_risk'] = float(np.mean(vals))
            else:
                G_new.nodes[node]['neighbor_avg_risk'] = 0.0

        updated_networks[year] = G_new
    return updated_networks
"""


def _load_runner(seed: int) -> tuple[callable, dict[str, Any]]:
    exp4_path = Path("experiments/exp4_stgnn_temporal.py")
    code = exp4_path.read_text(encoding="utf-8")

    model_path = Path(f"experiments/tmp_multiplicative_seed_{seed}.pt").as_posix()

    code = code.replace(
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
        "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n"
        + HELPER_CODE
        + "\n",
    )

    code = code.replace(
        "with open(networks_file, 'rb') as f:\n        all_networks = pickle.load(f)\n",
        (
            "with open(networks_file, 'rb') as f:\n"
            "        all_networks = pickle.load(f)\n"
            "    all_networks = recompute_multiplicative_domirank(all_networks)\n"
        ),
    )

    code = code.replace("'affective_domirank'", "'multiplicative_domirank'")

    code = code.replace(
        "'experiments/best_stgnn_node_level.pt'",
        repr(model_path),
    )

    code = code.replace(
        "    best_test_r2 = -float('inf')\n",
        "    best_test_r2 = -float('inf')\n    best_model_state = None\n",
    )

    code = code.replace(
        "            torch.save(model.state_dict(), 'experiments/best_stgnn_node_level.pt')\n",
        "            best_model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}\n",
    )

    code = code.replace(
        "    model.load_state_dict(torch.load('experiments/best_stgnn_node_level.pt'))\n",
        (
            "    if best_model_state is None:\n"
            "        raise RuntimeError('best_model_state was not captured during training')\n"
            "    model.load_state_dict(best_model_state)\n"
        ),
    )

    code = code.replace(
        "    import random\n    random.seed(42)\n",
        (
            "    import random\n"
            "    random.seed(EXPERIMENT_SEED)\n"
            "    np.random.seed(EXPERIMENT_SEED)\n"
            "    torch.manual_seed(EXPERIMENT_SEED)\n"
            "    if torch.cuda.is_available():\n"
            "        torch.cuda.manual_seed_all(EXPERIMENT_SEED)\n"
            "    if hasattr(torch.backends, 'cudnn'):\n"
            "        torch.backends.cudnn.deterministic = True\n"
            "        torch.backends.cudnn.benchmark = False\n"
        ),
    )

    code = code.replace(
        "实验4: STGNN节点级时序预测",
        "乘性门控版本: STGNN节点级时序预测",
    )
    code = code.replace(
        "Affective DomiRank作为关键特征",
        "Multiplicative Sentiment Gate作为关键特征",
    )
    code = code.replace(
        "情感DomiRank",
        "乘性门控DomiRank",
    )

    code = code.replace(
        "    print(f\"\\n优势: Transformer注意力机制自适应捕捉时序依赖，比LSTM更强\")\n",
        (
            "    print(f\"\\n优势: Transformer注意力机制自适应捕捉时序依赖，比LSTM更强\")\n"
            "    return {\n"
            "        'variant': 'multiplicative',\n"
            "        'seed': EXPERIMENT_SEED,\n"
            "        'train_metrics': train_metrics,\n"
            "        'test_metrics': test_metrics,\n"
            "        'best_test_r2': best_test_r2,\n"
            "        'num_train_samples': len(train_samples),\n"
            "        'num_test_samples': len(test_samples),\n"
            "    }\n"
        ),
    )

    namespace: dict[str, Any] = {
        "__name__": f"multiplicative_seed_{seed}",
        "__file__": str(exp4_path.resolve()),
        "__package__": None,
        "EXPERIMENT_SEED": seed,
    }
    exec(code, namespace)
    return namespace["main"], namespace


def run_single_seed(seed: int) -> dict[str, Any]:
    main_fn, _ = _load_runner(seed)
    result = main_fn()

    model_path = Path(f"experiments/tmp_multiplicative_seed_{seed}.pt")
    if model_path.exists():
        model_path.unlink()

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ["mae", "rmse", "r2"]:
        values = [r["test_metrics"][metric] for r in results]
        summary[metric] = {
            "values": values,
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multiplicative-gate seeds.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/results/multiplicative_gate_results.json",
        help="Path to save the summary JSON.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Multiplicative Sentiment Gate Test")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")

    runs = []
    for seed in args.seeds:
        print("\n" + "-" * 80)
        print(f"Running Multiplicative Gate | seed={seed}")
        print("-" * 80)
        runs.append(run_single_seed(seed))

    summary = summarize(runs)
    output = {
        "seeds": args.seeds,
        "runs": runs,
        "summary": summary,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for metric in ["mae", "rmse", "r2"]:
        stats = summary[metric]
        print(f"{metric.upper():<6} | mean={stats['mean']:.4f} ± {stats['std']:.4f} | values={stats['values']}")

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
