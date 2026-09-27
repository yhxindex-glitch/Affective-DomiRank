"""
映射系数敏感性分析实验（Reviewer 3 Comment 3）

目的：
1. 固定 alpha / threshold / gamma / window 等其余参数；
2. 单独扫描情感到风险映射函数中的两个启发式系数：
   - exponent p（默认 0.8）
   - positive_scale c（默认 0.5）
3. 采用与正式敏感性分析一致的训练预算，评估不同映射系数下的模型稳健性。

方案A（控制变量法）：
- p ∈ {0.5, 0.6, 0.8, 1.0, 1.2}，固定 c = 0.5
- c ∈ {0.2, 0.3, 0.5, 0.7, 1.0}，固定 p = 0.8
- 默认点 (0.8, 0.5) 只训练一次并在两组结果中复用

用法：
    python experiments/run_mapping_coeff_sensitivity.py
    python experiments/run_mapping_coeff_sensitivity.py --workers 4
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from experiments.run_sensitivity_analysis_full import (  # noqa: E402
    BATCH_SIZE,
    DEFAULT_ALPHA,
    DEFAULT_GAMMA,
    DEFAULT_THRESHOLD,
    DEFAULT_WINDOW,
    DEVICE,
    EPOCHS_PER_EXPERIMENT,
    PATIENCE,
    ROOT,
    RANDOM_SEED,
    run_single_experiment,
)


DEFAULT_EXPONENT = 0.8
DEFAULT_POSITIVE_SCALE = 0.5

EXPONENT_VALUES = [0.5, 0.6, 0.8, 1.0, 1.2]
POSITIVE_SCALE_VALUES = [0.2, 0.3, 0.5, 0.7, 1.0]

DEFAULT_WORKERS = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class MappingParameterizedAffectiveDomiRank:
    """支持自定义 p/c 的 Affective DomiRank。"""

    def __init__(
        self,
        G: nx.DiGraph,
        alpha: float = DEFAULT_ALPHA,
        beta: float = 0.5,
        theta: float = 1.0,
        gamma: float = DEFAULT_GAMMA,
        threshold: float = DEFAULT_THRESHOLD,
        exponent: float = DEFAULT_EXPONENT,
        positive_scale: float = DEFAULT_POSITIVE_SCALE,
    ):
        self.G = G
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.gamma = gamma
        self.threshold = threshold
        self.exponent = exponent
        self.positive_scale = positive_scale

        self.nodes = list(G.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        self.sentiment_risk_scores = self._compute_sentiment_risk()

    def _compute_sentiment_risk(self) -> dict[Any, float]:
        risk_dict: dict[Any, float] = {}

        for node in self.nodes:
            sentiment = float(self.G.nodes[node].get("sentiment_mean", 0.0))

            if abs(sentiment) < self.threshold:
                sentiment_risk = 0.0
            elif sentiment < 0:
                sentiment_risk = abs(sentiment) ** self.exponent
            else:
                sentiment_risk = -(sentiment ** self.exponent) * self.positive_scale

            risk_dict[node] = float(sentiment_risk)

        return risk_dict

    def _dgamma_dt(self, gamma_values: np.ndarray) -> np.ndarray:
        dgamma = np.zeros(self.n, dtype=np.float64)

        for i, node in enumerate(self.nodes):
            neighbor_sum = 0.0
            for neighbor in self.G.predecessors(node):
                j = self.node_to_idx[neighbor]
                neighbor_sum += self.theta - gamma_values[j]

            dgamma[i] = (
                self.alpha * neighbor_sum
                - self.beta * gamma_values[i]
                + self.gamma * self.sentiment_risk_scores[node]
            )

        return dgamma

    def compute(self, max_iter: int = 2000, tol: float = 1e-6, time_step: float = 0.01) -> dict[Any, float]:
        gamma_values = np.ones(self.n, dtype=np.float64)

        for _ in range(max_iter):
            gamma_new = gamma_values + time_step * self._dgamma_dt(gamma_values)
            if np.linalg.norm(gamma_new - gamma_values) < tol:
                gamma_values = gamma_new
                break
            gamma_values = gamma_new

        return {node: float(gamma_values[i]) for i, node in enumerate(self.nodes)}


def recompute_affective_domirank_mapping(
    networks: dict,
    exponent: float,
    positive_scale: float,
    alpha: float = DEFAULT_ALPHA,
    threshold: float = DEFAULT_THRESHOLD,
    gamma: float = DEFAULT_GAMMA,
) -> dict:
    updated_networks = {}

    for year, graph in networks.items():
        graph_new = copy.deepcopy(graph)

        affective_dr = MappingParameterizedAffectiveDomiRank(
            graph_new,
            alpha=alpha,
            beta=0.5,
            theta=1.0,
            gamma=gamma,
            threshold=threshold,
            exponent=exponent,
            positive_scale=positive_scale,
        )
        scores = affective_dr.compute(max_iter=2000, tol=1e-6)

        for node, score in scores.items():
            graph_new.nodes[node]["affective_domirank"] = score

        for node in graph_new.nodes():
            predecessors = list(graph_new.predecessors(node))
            if predecessors:
                domiranks = [graph_new.nodes[p].get("affective_domirank", 0.0) for p in predecessors]
                graph_new.nodes[node]["neighbor_avg_risk"] = float(np.mean(domiranks))
            else:
                graph_new.nodes[node]["neighbor_avg_risk"] = 0.0

        updated_networks[year] = graph_new

    return updated_networks


def _metrics_to_payload(metrics: dict, extra: dict) -> dict:
    payload = {
        "train_r2": metrics["train"]["r2"],
        "train_mae": metrics["train"]["mae"],
        "train_rmse": metrics["train"]["rmse"],
        "test_r2": metrics["test"]["r2"],
        "test_mae": metrics["test"]["mae"],
        "test_rmse": metrics["test"]["rmse"],
        "best_test_r2": metrics["best_test_r2"],
        "best_epoch": metrics["best_epoch"],
    }
    payload.update(extra)
    return payload


def _record(container: list, payload: dict) -> None:
    container.append(payload)
    print(
        f"    ✓ Test R²={payload['test_r2']:.4f}  "
        f"MAE={payload['test_mae']:.4f}  RMSE={payload['test_rmse']:.4f}  "
        f"BestEpoch={payload['best_epoch']}"
    )


def _run_single_job(job: dict) -> dict:
    set_seed(job["seed"])

    with open(job["networks_file"], "rb") as f:
        original_networks = pickle.load(f)

    updated_networks = recompute_affective_domirank_mapping(
        original_networks,
        exponent=job["exponent"],
        positive_scale=job["positive_scale"],
        alpha=job["alpha"],
        threshold=job["threshold"],
        gamma=job["gamma"],
    )
    metrics = run_single_experiment(updated_networks, job["window"], seed=job["seed"])
    return {**job, "metrics": metrics}


def run_mapping_coeff_sensitivity(num_workers: int = DEFAULT_WORKERS) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("=" * 80)
    print("映射系数敏感性分析实验 [方案A：控制变量法]")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"设备: {DEVICE}  |  MaxEpochs: {EPOCHS_PER_EXPERIMENT}  "
        f"|  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}  |  Workers: {num_workers}"
    )
    print("=" * 80)

    networks_file = str(ROOT / "data" / "processed" / "networks" / "all_networks_2017_2021.pkl")

    exponent_jobs = [
        {
            "networks_file": networks_file,
            "alpha": DEFAULT_ALPHA,
            "threshold": DEFAULT_THRESHOLD,
            "gamma": DEFAULT_GAMMA,
            "window": DEFAULT_WINDOW,
            "seed": RANDOM_SEED,
            "exponent": exponent,
            "positive_scale": DEFAULT_POSITIVE_SCALE,
            "job_type": "exponent",
        }
        for exponent in EXPONENT_VALUES
    ]

    positive_scale_jobs = [
        {
            "networks_file": networks_file,
            "alpha": DEFAULT_ALPHA,
            "threshold": DEFAULT_THRESHOLD,
            "gamma": DEFAULT_GAMMA,
            "window": DEFAULT_WINDOW,
            "seed": RANDOM_SEED,
            "exponent": DEFAULT_EXPONENT,
            "positive_scale": positive_scale,
            "job_type": "positive_scale",
        }
        for positive_scale in POSITIVE_SCALE_VALUES
        if positive_scale != DEFAULT_POSITIVE_SCALE
    ]

    all_jobs = exponent_jobs + positive_scale_jobs

    print(
        f"\n实际训练任务: {len(all_jobs)} 组（默认点复用后为 9 组）"
        f"\n固定参数: α={DEFAULT_ALPHA}, τ={DEFAULT_THRESHOLD}, γ={DEFAULT_GAMMA}, w={DEFAULT_WINDOW}"
        f"\n扫描参数: p={EXPONENT_VALUES}, c={POSITIVE_SCALE_VALUES}\n"
    )

    results: dict[str, Any] = {
        "exponent_sensitivity": [],
        "positive_scale_sensitivity": [],
        "metadata": {
            "run_type": "mapping_coeff_sensitivity_plan_a",
            "device": str(DEVICE),
            "seed": RANDOM_SEED,
            "default_alpha": DEFAULT_ALPHA,
            "default_threshold": DEFAULT_THRESHOLD,
            "default_gamma": DEFAULT_GAMMA,
            "default_window": DEFAULT_WINDOW,
            "default_exponent": DEFAULT_EXPONENT,
            "default_positive_scale": DEFAULT_POSITIVE_SCALE,
            "exponent_values": EXPONENT_VALUES,
            "positive_scale_values": POSITIVE_SCALE_VALUES,
            "epochs_per_experiment": EPOCHS_PER_EXPERIMENT,
            "patience": PATIENCE,
            "batch_size": BATCH_SIZE,
            "parallel_workers": num_workers,
            "timestamp": datetime.now().isoformat(),
        },
    }

    result_map: dict[tuple[float, float], dict] = {}

    progress = tqdm(total=len(all_jobs), desc="总体进度", ncols=100)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_job = {executor.submit(_run_single_job, job): job for job in all_jobs}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
                metrics = result["metrics"]
                exponent = result["exponent"]
                positive_scale = result["positive_scale"]
                result_map[(exponent, positive_scale)] = metrics

                if result["job_type"] == "exponent":
                    print(f"\n  [指数] p={exponent}, c={positive_scale}")
                    _record(
                        results["exponent_sensitivity"],
                        _metrics_to_payload(metrics, {"exponent": exponent, "positive_scale": positive_scale}),
                    )
                else:
                    print(f"\n  [正向衰减] p={exponent}, c={positive_scale}")
                    _record(
                        results["positive_scale_sensitivity"],
                        _metrics_to_payload(metrics, {"exponent": exponent, "positive_scale": positive_scale}),
                    )
            except Exception as exc:
                print(
                    f"\n  [错误] {job['job_type']} "
                    f"p={job['exponent']} c={job['positive_scale']}: {exc}"
                )
            progress.update(1)

    progress.close()

    default_metrics = result_map.get((DEFAULT_EXPONENT, DEFAULT_POSITIVE_SCALE))
    if default_metrics is not None:
        existing_scales = {entry["positive_scale"] for entry in results["positive_scale_sensitivity"]}
        if DEFAULT_POSITIVE_SCALE not in existing_scales:
            print(f"\n  [复用默认点] p={DEFAULT_EXPONENT}, c={DEFAULT_POSITIVE_SCALE}")
            _record(
                results["positive_scale_sensitivity"],
                _metrics_to_payload(
                    default_metrics,
                    {"exponent": DEFAULT_EXPONENT, "positive_scale": DEFAULT_POSITIVE_SCALE},
                ),
            )

    results["exponent_sensitivity"].sort(key=lambda x: x["exponent"])
    results["positive_scale_sensitivity"].sort(key=lambda x: x["positive_scale"])

    output_pkl = ROOT / "experiments" / "mapping_coeff_sensitivity_results.pkl"
    output_json = ROOT / "experiments" / "mapping_coeff_sensitivity_results_summary.json"

    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)

    summary = {
        "metadata": results["metadata"],
        "best_exponent": max(results["exponent_sensitivity"], key=lambda x: x["test_r2"]),
        "best_positive_scale": max(results["positive_scale_sensitivity"], key=lambda x: x["test_r2"]),
        "default_point": next(
            item
            for item in results["exponent_sensitivity"]
            if item["exponent"] == DEFAULT_EXPONENT and item["positive_scale"] == DEFAULT_POSITIVE_SCALE
        ),
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("映射系数敏感性分析完成")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果已保存: {output_pkl}")
    print(f"摘要已保存: {output_json}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="映射系数敏感性分析（方案A）")
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行 worker 数（默认 {DEFAULT_WORKERS}）",
    )
    args = parser.parse_args()

    run_mapping_coeff_sensitivity(num_workers=args.workers)
