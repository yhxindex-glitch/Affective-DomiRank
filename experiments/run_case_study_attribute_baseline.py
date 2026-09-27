"""
Case study baseline for Reviewer 2 Comment 9.

Compare:
1. Traditional DomiRank
2. Sentiment-Weighted PageRank
3. Affective DomiRank

Outputs:
- English and Chinese figures
- JSON summary for overlap statistics
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
NETWORK_FILE = ROOT / "data" / "processed" / "networks" / "all_networks_2017_2021.pkl"
FIG_EN = ROOT / "figures" / "topk_nodes_comparison_attribute.png"
FIG_CN = ROOT / "figures_cn" / "topk_nodes_comparison_attribute.png"
JSON_OUT = ROOT / "analysis" / "results" / "case_study_attribute_baseline.json"

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure_export_utils import save_figure_png_and_pdf

COLORS = {
    "traditional": "#3498DB",
    "weighted_pr": "#F39C12",
    "affective": "#E74C3C",
    "background": "#F8F9FA",
    "text": "#2C3E50",
    "grid": "#E9ECEF",
}


def setup_style(lang: str) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    if lang == "cn":
        plt.rcParams.update(
            {
                "font.sans-serif": ["SimHei", "Microsoft YaHei", "STSong", "SimSun", "Arial Unicode MS"],
                "font.family": "sans-serif",
                "axes.unicode_minus": False,
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "DejaVu Serif"],
                "axes.unicode_minus": True,
            }
        )
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_networks() -> dict[int, nx.DiGraph]:
    with open(NETWORK_FILE, "rb") as f:
        return pickle.load(f)


def sentiment_to_risk(sentiment: float, threshold: float = 0.05) -> float:
    if abs(sentiment) < threshold:
        return 0.0
    if sentiment < 0:
        return abs(sentiment) ** 0.8
    return -0.5 * (sentiment**0.8)


def minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    vals = np.array(list(values.values()), dtype=float)
    vmin = float(vals.min())
    vmax = float(vals.max())
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


def compute_sentiment_weighted_pagerank(
    G: nx.DiGraph, alpha: float = 0.85, lambda_weight: float = 0.8
) -> dict[str, float]:
    pagerank_scores = nx.pagerank(G, alpha=alpha, weight="weight")
    weighted_scores: dict[str, float] = {}
    for node, pr in pagerank_scores.items():
        sentiment = float(G.nodes[node].get("sentiment_mean", 0.0))
        risk = sentiment_to_risk(sentiment)
        # Keep the baseline simple and always non-negative.
        weighted_scores[node] = max(0.0, pr * (1.0 + lambda_weight * risk))
    return minmax_normalize(weighted_scores)


def topk_nodes(scores: dict[str, float], k: int) -> list[str]:
    return [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = [node for node, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return {node: idx + 1 for idx, node in enumerate(ordered)}


def build_summary(
    G: nx.DiGraph, k_values: list[int], year: int
) -> dict[str, object]:
    traditional = minmax_normalize({n: float(G.nodes[n].get("domirank", 0.0)) for n in G.nodes()})
    affective = minmax_normalize({n: float(G.nodes[n].get("affective_domirank", 0.0)) for n in G.nodes()})
    weighted_pr = compute_sentiment_weighted_pagerank(G)

    summaries = []
    for k in k_values:
        trad_topk = set(topk_nodes(traditional, k))
        aff_topk = set(topk_nodes(affective, k))
        pr_topk = set(topk_nodes(weighted_pr, k))
        summaries.append(
            {
                "k": k,
                "traditional_affective_overlap": len(trad_topk & aff_topk),
                "weightedpr_affective_overlap": len(pr_topk & aff_topk),
                "traditional_affective_overlap_ratio": len(trad_topk & aff_topk) / k,
                "weightedpr_affective_overlap_ratio": len(pr_topk & aff_topk) / k,
            }
        )

    trad_ranks = rank_map(traditional)
    aff_ranks = rank_map(affective)
    pr_ranks = rank_map(weighted_pr)
    aff_top10 = topk_nodes(affective, 10)

    top10_table = []
    for node in aff_top10:
        top10_table.append(
            {
                "node": node,
                "sentiment_mean": float(G.nodes[node].get("sentiment_mean", 0.0)),
                "traditional_rank": trad_ranks[node],
                "weightedpr_rank": pr_ranks[node],
                "affective_rank": aff_ranks[node],
                "traditional_score": traditional[node],
                "weightedpr_score": weighted_pr[node],
                "affective_score": affective[node],
            }
        )

    return {
        "year": year,
        "k_values": k_values,
        "overlap_summary": summaries,
        "affective_top10": top10_table,
    }


def create_figure(summary: dict[str, object], lang: str, save_path: Path) -> None:
    setup_style(lang)
    is_cn = lang == "cn"
    overlap = summary["overlap_summary"]
    ks = [item["k"] for item in overlap]
    trad_overlap = [item["traditional_affective_overlap_ratio"] for item in overlap]
    pr_overlap = [item["weightedpr_affective_overlap_ratio"] for item in overlap]
    top10 = summary["affective_top10"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")

    ax1, ax2 = axes
    ax1.set_facecolor(COLORS["background"])
    ax2.set_facecolor(COLORS["background"])

    ax1.plot(ks, trad_overlap, marker="o", linewidth=2.5, color=COLORS["traditional"])
    ax1.plot(ks, pr_overlap, marker="s", linewidth=2.5, color=COLORS["weighted_pr"])
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlabel("Top-K" if not is_cn else "Top-K范围")
    ax1.set_ylabel("Overlap Ratio" if not is_cn else "与情感DomiRank的重叠比例")
    ax1.legend(
        [
            "Traditional DomiRank vs Affective"
            if not is_cn
            else "传统DomiRank vs 情感DomiRank",
            "Sentiment-Weighted PageRank vs Affective"
            if not is_cn
            else "情感加权PageRank vs 情感DomiRank",
        ],
        loc="lower right",
    )

    nodes = [row["node"] for row in top10]
    x = np.arange(len(nodes))
    trad_scores = [row["traditional_score"] for row in top10]
    pr_scores = [row["weightedpr_score"] for row in top10]
    aff_scores = [row["affective_score"] for row in top10]

    width = 0.25
    ax2.bar(x - width, trad_scores, width, color=COLORS["traditional"], label="Traditional DomiRank" if not is_cn else "传统DomiRank")
    ax2.bar(x, pr_scores, width, color=COLORS["weighted_pr"], label="Sentiment-Weighted PageRank" if not is_cn else "情感加权PageRank")
    ax2.bar(x + width, aff_scores, width, color=COLORS["affective"], label="Affective DomiRank" if not is_cn else "情感DomiRank")
    ax2.set_xticks(x)
    ax2.set_xticklabels(nodes, rotation=55, ha="right", fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y", linestyle="--")
    ax2.set_ylabel("Normalized Score" if not is_cn else "归一化中心性分数")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_figure_png_and_pdf(save_path, dpi_png=300, dpi_pdf=400, bbox_inches="tight", facecolor="white")
    plt.close()


def main() -> None:
    print("=" * 80)
    print("Case Study Attribute-Aware Baseline Comparison")
    print("=" * 80)

    year = 2020
    k_values = [10, 20, 50, 100]
    networks = load_networks()
    G = networks[year]

    print(f"Loaded year {year}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    summary = build_summary(G, k_values=k_values, year=year)

    create_figure(summary, "en", FIG_EN)
    create_figure(summary, "cn", FIG_CN)

    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved figure: {FIG_EN}")
    print(f"Saved figure: {FIG_CN}")
    print(f"Saved summary: {JSON_OUT}")


if __name__ == "__main__":
    main()
