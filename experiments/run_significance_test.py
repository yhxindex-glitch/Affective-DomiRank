"""
多随机种子显著性检验脚本

用途：
1. 分别运行 Affective DomiRank 与 Traditional DomiRank 两个版本
2. 使用多个随机种子重复实验
3. 汇总 MAE / RMSE / R² 的均值与标准差
4. 计算 paired t-test 与 Wilcoxon signed-rank test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _load_runner(variant: str, seed: int) -> tuple[callable, dict[str, Any]]:
    """动态加载实验入口，避免修改原始脚本。"""
    exp4_path = Path("experiments/exp4_stgnn_temporal.py")
    code = exp4_path.read_text(encoding="utf-8")

    model_path = Path(f"experiments/tmp_{variant}_seed_{seed}.pt").as_posix()

    if variant == "traditional":
        code = code.replace(
            "affective_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['affective_domirank'], \n                        is_affective=True\n                    )",
            "traditional_vulnerability = self._domirank_to_vulnerability(\n                        cached_features['domirank'], \n                        is_affective=False\n                    )  # 使用传统DomiRank",
        )
        code = code.replace(
            "                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）",
            "                        traditional_vulnerability,  # 传统DomiRank脆弱性",
        )
        code = code.replace(
            "实验4: STGNN节点级时序预测（情感风险预测版）",
            "显著性检验：传统DomiRank版本",
        )
        code = code.replace(
            "Affective DomiRank作为关键特征",
            "Traditional DomiRank作为关键特征",
        )

    code = code.replace(
        "'experiments/best_stgnn_node_level.pt'",
        repr(model_path),
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
        "    print(f\"\\n优势: Transformer注意力机制自适应捕捉时序依赖，比LSTM更强\")\n",
        (
            "    print(f\"\\n优势: Transformer注意力机制自适应捕捉时序依赖，比LSTM更强\")\n"
            "    return {\n"
            "        'variant': EXPERIMENT_VARIANT,\n"
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
        "__name__": f"significance_{variant}_{seed}",
        "__file__": str(exp4_path.resolve()),
        "__package__": None,
        "EXPERIMENT_SEED": seed,
        "EXPERIMENT_VARIANT": variant,
    }
    exec(code, namespace)
    return namespace["main"], namespace


def run_single_variant(variant: str, seed: int) -> dict[str, Any]:
    main_fn, namespace = _load_runner(variant, seed)
    result = main_fn()

    model_path = Path(f"experiments/tmp_{variant}_seed_{seed}.pt")
    if model_path.exists():
        model_path.unlink()

    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["mae", "rmse", "r2"]
    summary: dict[str, Any] = {}

    for metric in metrics:
        values = [r["test_metrics"][metric] for r in results]
        summary[metric] = {
            "values": values,
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
        }

    return summary


def paired_t_test(x: list[float], y: list[float]) -> dict[str, float | None]:
    """无 scipy 时可退化的配对 t 检验。"""
    if len(x) != len(y) or len(x) < 2:
        return {"t_stat": None, "p_value": None}

    try:
        from scipy import stats

        t_stat, p_value = stats.ttest_rel(x, y)
        return {"t_stat": float(t_stat), "p_value": float(p_value)}
    except Exception:
        diffs = [a - b for a, b in zip(x, y)]
        diff_mean = mean(diffs)
        diff_std = stdev(diffs) if len(diffs) > 1 else 0.0
        if diff_std == 0:
            return {"t_stat": None, "p_value": None}
        t_stat = diff_mean / (diff_std / math.sqrt(len(diffs)))
        return {"t_stat": float(t_stat), "p_value": None}


def wilcoxon_test(x: list[float], y: list[float]) -> dict[str, float | None]:
    if len(x) != len(y) or len(x) < 2:
        return {"statistic": None, "p_value": None}

    try:
        from scipy import stats

        stat, p_value = stats.wilcoxon(x, y, zero_method="wilcox")
        return {"statistic": float(stat), "p_value": float(p_value)}
    except Exception:
        return {"statistic": None, "p_value": None}


def compare_variants(
    affective: dict[str, Any], traditional: dict[str, Any]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    metric_directions = {"mae": "lower", "rmse": "lower", "r2": "higher"}

    for metric in ["mae", "rmse", "r2"]:
        a_vals = affective[metric]["values"]
        t_vals = traditional[metric]["values"]
        comparisons[metric] = {
            "direction": metric_directions[metric],
            "affective_mean": affective[metric]["mean"],
            "traditional_mean": traditional[metric]["mean"],
            "affective_std": affective[metric]["std"],
            "traditional_std": traditional[metric]["std"],
            "paired_t_test": paired_t_test(a_vals, t_vals),
            "wilcoxon": wilcoxon_test(a_vals, t_vals),
        }

    return comparisons


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed significance tests.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 52, 62, 72, 82],
        help="Random seeds to evaluate.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="analysis/results/significance_test_results.json",
        help="Path to save the summary JSON.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Affective vs Traditional DomiRank Significance Test")
    print("=" * 80)
    print(f"Seeds: {args.seeds}")

    affective_runs = []
    traditional_runs = []

    for seed in args.seeds:
        print("\n" + "-" * 80)
        print(f"Running Affective DomiRank | seed={seed}")
        print("-" * 80)
        affective_runs.append(run_single_variant("affective", seed))

        print("\n" + "-" * 80)
        print(f"Running Traditional DomiRank | seed={seed}")
        print("-" * 80)
        traditional_runs.append(run_single_variant("traditional", seed))

    affective_summary = summarize(affective_runs)
    traditional_summary = summarize(traditional_runs)
    comparison = compare_variants(affective_summary, traditional_summary)

    output = {
        "seeds": args.seeds,
        "affective_runs": affective_runs,
        "traditional_runs": traditional_runs,
        "affective_summary": affective_summary,
        "traditional_summary": traditional_summary,
        "comparison": comparison,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for metric in ["mae", "rmse", "r2"]:
        cmp = comparison[metric]
        print(
            f"{metric.upper():<6} | "
            f"Affective: {cmp['affective_mean']:.4f} ± {cmp['affective_std']:.4f} | "
            f"Traditional: {cmp['traditional_mean']:.4f} ± {cmp['traditional_std']:.4f}"
        )
        print(
            f"       paired t-test p={cmp['paired_t_test']['p_value']} | "
            f"Wilcoxon p={cmp['wilcoxon']['p_value']}"
        )

    print(f"\nSaved summary to: {output_path}")


if __name__ == "__main__":
    main()
