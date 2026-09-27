"""
Energy + Minerals subnetwork case study for Reviewer 3 Comment 1.

This script generates:
1. A focused subnetwork comparison figure (EN/CN)
2. A JSON summary of overlap and rank changes
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
JSON_OUT = ROOT / "analysis" / "results" / "energy_minerals_case_study.json"
FIG_EN = ROOT / "figures" / "energy_minerals_case_study.png"
FIG_CN = ROOT / "figures_cn" / "energy_minerals_case_study.png"
TARGET_INDUSTRIES = {"Energy", "Minerals"}

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from figure_export_utils import save_figure_png_and_pdf

COLORS = {
    "traditional": "#3498DB",
    "affective": "#E74C3C",
    "background": "#F8F9FA",
    "text": "#2C3E50",
    "grid": "#E9ECEF",
    "highlight": "#F39C12",
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


def minmax_normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.array(list(values.values()), dtype=float)
    vmin = float(arr.min())
    vmax = float(arr.max())
    if abs(vmax - vmin) < 1e-12:
        return {k: 0.0 for k in values}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in values.items()}


def topk_nodes(scores: dict[str, float], k: int) -> list[str]:
    return [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


def rank_map(scores: dict[str, float]) -> dict[str, int]:
    ordered = [n for n, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    return {node: idx + 1 for idx, node in enumerate(ordered)}


def load_subnetwork(year: int = 2020) -> nx.DiGraph:
    with open(NETWORK_FILE, "rb") as f:
        networks = pickle.load(f)
    G = networks[year]
    nodes = [n for n in G.nodes() if G.nodes[n].get("industry") in TARGET_INDUSTRIES]
    return G.subgraph(nodes).copy()


def build_summary(year: int = 2020) -> dict[str, object]:
    G = load_subnetwork(year)
    traditional = minmax_normalize({n: float(G.nodes[n].get("domirank", 0.0)) for n in G.nodes()})
    affective = minmax_normalize({n: float(G.nodes[n].get("affective_domirank", 0.0)) for n in G.nodes()})

    k_values = [10, 20, 30, 50]
    overlap_summary = []
    for k in k_values:
        trad_topk = set(topk_nodes(traditional, k))
        aff_topk = set(topk_nodes(affective, k))
        overlap_summary.append(
            {
                "k": k,
                "overlap_count": len(trad_topk & aff_topk),
                "overlap_ratio": len(trad_topk & aff_topk) / k,
            }
        )

    trad_rank = rank_map(traditional)
    aff_rank = rank_map(affective)
    promoted = []
    for node in G.nodes():
        promoted.append(
            {
                "node": node,
                "country": G.nodes[node].get("country"),
                "industry": G.nodes[node].get("industry"),
                "sentiment_mean": float(G.nodes[node].get("sentiment_mean", 0.0)),
                "traditional_rank": trad_rank[node],
                "affective_rank": aff_rank[node],
                "rank_gain": trad_rank[node] - aff_rank[node],
                "traditional_score": traditional[node],
                "affective_score": affective[node],
            }
        )
    promoted.sort(key=lambda x: x["rank_gain"], reverse=True)

    aff_top10 = topk_nodes(affective, 10)
    top10_table = []
    for node in aff_top10:
        top10_table.append(
            {
                "node": node,
                "country": G.nodes[node].get("country"),
                "industry": G.nodes[node].get("industry"),
                "sentiment_mean": float(G.nodes[node].get("sentiment_mean", 0.0)),
                "traditional_rank": trad_rank[node],
                "affective_rank": aff_rank[node],
                "rank_gain": trad_rank[node] - aff_rank[node],
                "traditional_score": traditional[node],
                "affective_score": affective[node],
            }
        )

    return {
        "year": year,
        "node_count": G.number_of_nodes(),
        "edge_count": G.number_of_edges(),
        "industries": sorted(TARGET_INDUSTRIES),
        "overlap_summary": overlap_summary,
        "affective_top10": top10_table,
        "largest_rank_promotions": promoted[:10],
    }


def create_figure(summary: dict[str, object], lang: str, save_path: Path) -> None:
    setup_style(lang)
    is_cn = lang == "cn"
    overlap = summary["overlap_summary"]
    ks = [item["k"] for item in overlap]
    overlaps = [item["overlap_ratio"] for item in overlap]
    top10 = summary["affective_top10"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    fig.patch.set_facecolor("white")
    ax1, ax2 = axes
    ax1.set_facecolor(COLORS["background"])
    ax2.set_facecolor(COLORS["background"])

    ax1.plot(ks, overlaps, marker="o", linewidth=2.5, color=COLORS["highlight"])
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(ks)
    ax1.grid(True, alpha=0.3, linestyle="--")
    ax1.set_xlabel("Top-K" if not is_cn else "Top-K范围")
    ax1.set_ylabel("Overlap Ratio" if not is_cn else "重叠比例")

    nodes = [row["node"] for row in top10]
    x = np.arange(len(nodes))
    trad_scores = [row["traditional_score"] for row in top10]
    aff_scores = [row["affective_score"] for row in top10]
    width = 0.36

    ax2.bar(x - width / 2, trad_scores, width, color=COLORS["traditional"], label="Traditional DomiRank" if not is_cn else "传统DomiRank")
    ax2.bar(x + width / 2, aff_scores, width, color=COLORS["affective"], label="Affective DomiRank" if not is_cn else "情感DomiRank")
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
    print("Energy + Minerals Case Study")
    print("=" * 80)
    summary = build_summary(year=2020)
    create_figure(summary, "en", FIG_EN)
    create_figure(summary, "cn", FIG_CN)
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    JSON_OUT.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved figure: {FIG_EN}")
    print(f"Saved figure: {FIG_CN}")
    print(f"Saved summary: {JSON_OUT}")


if __name__ == "__main__":
    main()
