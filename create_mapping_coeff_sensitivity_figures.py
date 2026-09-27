"""
映射系数敏感性分析可视化

基于 experiments/mapping_coeff_sensitivity_results.pkl 生成中英文组合图：
- figures/mapping_coeff_sensitivity.png + .pdf（400 dpi PDF）
- figures_cn/mapping_coeff_sensitivity.png + .pdf
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

from figure_export_utils import save_figure_png_and_pdf


PALETTE = {
    "primary": "#4A7298",
    "secondary": "#5B8A72",
    "accent": "#D4A84B",
    "danger": "#B85C4C",
    "light": "#9CA3AF",
    "dark": "#374151",
    "bg_a": "#F2F5FA",
    "bg_b": "#F3F7F4",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 300,
    })


def get_cn_font() -> FontProperties | None:
    for fp in ["C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/msyh.ttc"]:
        if Path(fp).exists():
            return FontProperties(fname=fp)
    return None


def read_results() -> dict:
    path = Path("experiments/mapping_coeff_sensitivity_results.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def panel_style(ax, facecolor: str) -> None:
    ax.set_facecolor(facecolor)
    ax.grid(True, axis="y", alpha=0.18, linestyle="-", linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["light"])


def draw_series(ax, x, y, default_x, best_x, title, xlabel, ylabel, facecolor, marker):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    panel_style(ax, facecolor)
    ax.plot(x, y, color=PALETTE["primary"], linewidth=2.2, marker=marker, markersize=6)
    ax.scatter([default_x], [y[list(x).index(default_x)]], s=120, color=PALETTE["danger"],
               edgecolors="white", linewidths=1.3, zorder=4)
    ax.scatter([best_x], [y[list(x).index(best_x)]], s=150, color=PALETTE["accent"],
               edgecolors="white", linewidths=1.3, zorder=5)

    ax.set_title(title, color=PALETTE["dark"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)

    ymin = float(y.min())
    ymax = float(y.max())
    margin = (ymax - ymin) * 0.30 if ymax > ymin else 0.01
    ax.set_ylim(ymin - margin * 0.4, ymax + margin)

    default_y = y[list(x).index(default_x)]
    best_y = y[list(x).index(best_x)]
    ax.annotate(
        f"Default\nR$^2$={default_y:.3f}",
        xy=(default_x, default_y),
        xytext=(8, -38),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["light"], linewidth=0.6),
    )
    ax.annotate(
        f"Best\nR$^2$={best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(8, 12),
        textcoords="offset points",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=PALETTE["light"], linewidth=0.6),
    )


def create_figure(results: dict, lang: str = "en") -> Path:
    setup_style()
    is_cn = lang == "cn"
    fp = get_cn_font() if is_cn else None
    out_dir = Path("figures_cn" if is_cn else "figures")
    out_dir.mkdir(exist_ok=True)

    exp_data = sorted(results["exponent_sensitivity"], key=lambda item: item["exponent"])
    scale_data = sorted(results["positive_scale_sensitivity"], key=lambda item: item["positive_scale"])

    exp_x = [item["exponent"] for item in exp_data]
    exp_y = [item["test_r2"] for item in exp_data]
    scale_x = [item["positive_scale"] for item in scale_data]
    scale_y = [item["test_r2"] for item in scale_data]

    default_exp = results["metadata"]["default_exponent"]
    default_scale = results["metadata"]["default_positive_scale"]
    best_exp = max(exp_data, key=lambda item: item["test_r2"])["exponent"]
    best_scale = max(scale_data, key=lambda item: item["test_r2"])["positive_scale"]

    if is_cn:
        titles = ("(a) 指数系数 $p$ 敏感性", "(b) 正向衰减系数 $c$ 敏感性")
        xlabel_a = "指数系数 $p$"
        xlabel_b = "正向衰减系数 $c$"
        ylabel = "预测精度 ($R^2$)"
    else:
        titles = ("(a) Exponent $p$ sensitivity", "(b) Positive-scale $c$ sensitivity")
        xlabel_a = "Exponent $p$"
        xlabel_b = "Positive attenuation $c$"
        ylabel = "Prediction Accuracy ($R^2$)"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    draw_series(
        axes[0], exp_x, exp_y, default_exp, best_exp, titles[0], xlabel_a, ylabel, PALETTE["bg_a"], "o"
    )
    draw_series(
        axes[1], scale_x, scale_y, default_scale, best_scale, titles[1], xlabel_b, ylabel, PALETTE["bg_b"], "s"
    )

    if fp is not None:
        for ax, title in zip(axes, titles):
            ax.set_title(title, color=PALETTE["dark"], fontproperties=fp)
            ax.set_xlabel(ax.get_xlabel(), fontproperties=fp)
            ax.set_ylabel(ax.get_ylabel(), fontproperties=fp)
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontproperties(fp)

    plt.tight_layout()
    output = out_dir / "mapping_coeff_sensitivity.png"
    save_figure_png_and_pdf(output, dpi_png=300, dpi_pdf=400, bbox_inches="tight")
    plt.close()
    return output


def main() -> None:
    results = read_results()
    out_en = create_figure(results, "en")
    out_cn = create_figure(results, "cn")
    print(f"[OK] {out_en}")
    print(f"[OK] {out_cn}")


if __name__ == "__main__":
    main()
