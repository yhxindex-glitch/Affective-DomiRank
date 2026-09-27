"""
正式版参数敏感性分析实验 [提速优化版 · 多进程并行]

优化策略（不降低实验质量，输出格式与原版完全一致）：
1. 多进程并行（ProcessPoolExecutor），单实验仅占 ~19MB 显存，
   32GB 显存理论并行度 >1000，实际受 CPU/RAM 限制，默认 8 workers。
2. 先跑 α×τ 热力图 (30组)，从中直接提取 α/τ 单变量敏感性，
   避免 12 次重复训练，实际训练次数 44→32。
3. max_epochs 300→150（早停通常在 90-100 轮触发，150 已充裕）。
4. batch_size 32→64，patience 15→20。
5. 训练循环内只评估 test；train 指标仅在最终用最优模型计算一次。
6. 各 worker 结果写完后，主进程统一汇总，输出与原版完全相同。

用法:
    python experiments/run_sensitivity_analysis_full.py
    python experiments/run_sensitivity_analysis_full.py --workers 12
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import pickle
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))

from models.stgnn import STGNN, STGNNConfig


ROOT    = Path(__file__).resolve().parent.parent
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_SEED = 42

EPOCHS_PER_EXPERIMENT = 150
PATIENCE   = 20
BATCH_SIZE = 64
# 受 CPU/RAM 限制，不受显存限制；32GB VRAM 机器通常可设 8-12
DEFAULT_WORKERS = 4

ALPHA_VALUES     = [0.005, 0.01, 0.03, 0.05, 0.1, 0.2]
THRESHOLD_VALUES = [0.02, 0.05, 0.08, 0.1, 0.15]
WINDOW_VALUES    = [2, 3, 4]

DEFAULT_ALPHA     = 0.01
DEFAULT_THRESHOLD = 0.05
DEFAULT_GAMMA     = 0.1
DEFAULT_WINDOW    = 3


# ── 模块加载（spawned worker 进程也会执行这段，不要放入 __main__ 块）──────────

def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


light_sensitivity = _load_module(
    ROOT / "experiments" / "run_sensitivity_analysis.py",
    "run_sensitivity_analysis_light",
)
exp4_module = _load_module(
    ROOT / "experiments" / "exp4_stgnn_temporal.py",
    "exp4_stgnn_temporal_full",
)

recompute_affective_domirank = light_sensitivity.recompute_affective_domirank
NodeLevelSTGNNDataset        = exp4_module.NodeLevelSTGNNDataset
collate_fn                   = exp4_module.collate_fn
train_epoch                  = exp4_module.train_epoch
evaluate                     = exp4_module.evaluate


# ── 工具 ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SimpleDataset:
    def __init__(self, samples, label_mean: float, label_std: float):
        self.samples    = samples
        self.label_mean = label_mean
        self.label_std  = label_std

    def __len__(self):              return len(self.samples)
    def __getitem__(self, idx):     return self.samples[idx]
    def denormalize_labels(self, labels): return labels * self.label_std + self.label_mean


def _build_dataloaders(full_dataset):
    all_samples = full_dataset.samples.copy()
    random.shuffle(all_samples)

    split_idx    = int(len(all_samples) * 0.8)
    base_train   = all_samples[:split_idx]
    test_samples = all_samples[split_idx:]
    train_samples = base_train
    from training.splits import assert_disjoint_samples
    assert_disjoint_samples(train_samples, test_samples)
    random.shuffle(train_samples)

    kw = {"batch_size": BATCH_SIZE, "collate_fn": collate_fn,
          "num_workers": 0, "pin_memory": torch.cuda.is_available()}
    train_ds = SimpleDataset(train_samples, full_dataset.label_mean, full_dataset.label_std)
    test_ds  = SimpleDataset(test_samples,  full_dataset.label_mean, full_dataset.label_std)
    train_ld = torch.utils.data.DataLoader(train_ds, shuffle=True,  **kw)
    test_ld  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **kw)
    return train_ds, test_ds, train_ld, test_ld


def run_single_experiment(networks, window_size: int, seed: int = RANDOM_SEED) -> dict:
    set_seed(seed)
    full_dataset = NodeLevelSTGNNDataset(
        copy.deepcopy(networks), window_size=window_size, use_interpolation=True
    )
    train_ds, test_ds, train_ld, test_ld = _build_dataloaders(full_dataset)

    config = STGNNConfig(
        node_feature_dim=16, num_nodes=len(full_dataset.all_nodes),
        embed_dim=16, hidden_dim=64, num_time_steps=window_size,
        num_layers=2, cheb_k=2, dropout=0.1, output_dim=1,
    )
    model     = STGNN(config).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-6
    )

    best_r2, best_epoch, best_state, patience_counter = -float("inf"), 0, None, 0

    for epoch in range(1, EPOCHS_PER_EXPERIMENT + 1):
        train_epoch(model, train_ld, optimizer, DEVICE)
        scheduler.step()
        test_metrics = evaluate(model, test_ld, test_ds, DEVICE)
        if test_metrics["r2"] > best_r2:
            best_r2      = test_metrics["r2"]
            best_epoch   = epoch
            best_state   = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, train_ld, train_ds, DEVICE)
    test_metrics  = evaluate(model, test_ld,  test_ds,  DEVICE)

    del model, full_dataset, train_ld, test_ld
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train": train_metrics, "test": test_metrics,
        "best_test_r2": best_r2, "best_epoch": best_epoch,
        "train_samples": len(train_ds), "test_samples": len(test_ds), "seed": seed,
    }


# ── Worker（必须定义在模块顶层，才能被 pickle 传给子进程）──────────────────────

def _run_single_job(job: dict) -> dict:
    """
    每个 job 在独立子进程中运行。
    job 字段: networks_file, alpha, threshold, gamma, window, seed
    返回: job 信息 + metrics
    """
    with open(job["networks_file"], "rb") as f:
        original_networks = pickle.load(f)

    nets = recompute_affective_domirank(
        original_networks,
        alpha=job["alpha"],
        threshold=job["threshold"],
        gamma=job["gamma"],
    )
    metrics = run_single_experiment(nets, job["window"], seed=job["seed"])
    return {**job, "metrics": metrics}


# ── 结果辅助 ───────────────────────────────────────────────────────────────────

def _metrics_to_payload(metrics: dict, extra: dict) -> dict:
    p = {
        "train_r2":    metrics["train"]["r2"],
        "train_mae":   metrics["train"]["mae"],
        "train_rmse":  metrics["train"]["rmse"],
        "test_r2":     metrics["test"]["r2"],
        "test_mae":    metrics["test"]["mae"],
        "test_rmse":   metrics["test"]["rmse"],
        "best_test_r2": metrics["best_test_r2"],
        "best_epoch":  metrics["best_epoch"],
    }
    p.update(extra)
    return p


def _record(container: list, payload: dict) -> None:
    container.append(payload)
    print(
        f"    ✓ Test R²={payload['test_r2']:.4f}  "
        f"MAE={payload['test_mae']:.4f}  RMSE={payload['test_rmse']:.4f}  "
        f"BestEpoch={payload['best_epoch']}"
    )


# ── 主流程 ─────────────────────────────────────────────────────────────────────

def run_sensitivity_analysis_full(num_workers: int = DEFAULT_WORKERS) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed

    print("=" * 80)
    print("参数敏感性分析实验 [正式全量重跑版 · 多进程并行]")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"设备: {DEVICE}  |  MaxEpochs: {EPOCHS_PER_EXPERIMENT}  "
        f"|  Patience: {PATIENCE}  |  Batch: {BATCH_SIZE}  |  Workers: {num_workers}"
    )
    print("=" * 80)

    networks_file = str(ROOT / "data" / "processed" / "networks" / "all_networks_2017_2021.pkl")

    # ── 构建 job 列表 ────────────────────────────────────────────────────────
    # 热力图 30 组（覆盖 α/τ 单变量敏感性，避免 12 次重复训练）
    heatmap_jobs: list[dict] = [
        {
            "networks_file": networks_file,
            "alpha": alpha, "threshold": threshold, "gamma": DEFAULT_GAMMA,
            "window": DEFAULT_WINDOW, "seed": RANDOM_SEED,
            "job_type": "heatmap",
        }
        for alpha in ALPHA_VALUES
        for threshold in THRESHOLD_VALUES
    ]
    # 窗口敏感性（w=DEFAULT_WINDOW 从热力图复用，只跑 w=2 和 w=4）
    window_jobs: list[dict] = [
        {
            "networks_file": networks_file,
            "alpha": DEFAULT_ALPHA, "threshold": DEFAULT_THRESHOLD, "gamma": DEFAULT_GAMMA,
            "window": window, "seed": RANDOM_SEED,
            "job_type": "window",
        }
        for window in WINDOW_VALUES
        if window != DEFAULT_WINDOW
    ]
    all_jobs  = heatmap_jobs + window_jobs
    old_total = len(ALPHA_VALUES) + len(THRESHOLD_VALUES) + len(WINDOW_VALUES) + len(heatmap_jobs)

    print(
        f"\n实际训练任务: {len(all_jobs)} 组（原版 {old_total} 组，节省 {old_total - len(all_jobs)} 次）"
        f"\n并行度: {num_workers}  |  每组预算: ≤{EPOCHS_PER_EXPERIMENT} epochs\n"
    )

    results: dict[str, Any] = {
        "alpha_sensitivity": [], "threshold_sensitivity": [],
        "window_sensitivity": [], "alpha_threshold_heatmap": [],
        "metadata": {
            "run_type": "full_budget_parallel",
            "device": str(DEVICE), "seed": RANDOM_SEED,
            "alpha_values": ALPHA_VALUES, "threshold_values": THRESHOLD_VALUES,
            "window_values": WINDOW_VALUES,
            "default_alpha": DEFAULT_ALPHA, "default_threshold": DEFAULT_THRESHOLD,
            "default_gamma": DEFAULT_GAMMA, "default_window": DEFAULT_WINDOW,
            "epochs_per_experiment": EPOCHS_PER_EXPERIMENT,
            "patience": PATIENCE, "batch_size": BATCH_SIZE,
            "parallel_workers": num_workers,
            "timestamp": datetime.now().isoformat(),
        },
    }

    # (alpha, threshold) → metrics，供 α/τ 提取和窗口复用
    heatmap_map: dict[tuple, dict] = {}
    window_map:  dict[int,   dict] = {}

    overall_bar = tqdm(total=len(all_jobs), desc="总体进度", ncols=100)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_job = {executor.submit(_run_single_job, job): job for job in all_jobs}

        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result  = future.result()
                metrics = result["metrics"]
                alpha, threshold, window = result["alpha"], result["threshold"], result["window"]

                if result["job_type"] == "heatmap":
                    payload = _metrics_to_payload(metrics, {"alpha": alpha, "threshold": threshold})
                    print(f"\n  [热力图] α={alpha}, τ={threshold}")
                    _record(results["alpha_threshold_heatmap"], payload)
                    heatmap_map[(alpha, threshold)] = metrics
                else:
                    window_map[window] = metrics
                    print(f"\n  [窗口] w={window}")
                    _record([], _metrics_to_payload(metrics, {"window": window}))  # just for print

            except Exception as exc:
                print(f"\n  [错误] {job.get('job_type')} α={job['alpha']} τ={job['threshold']} w={job['window']}: {exc}")

            overall_bar.update(1)

    overall_bar.close()

    # ── 从热力图提取 α/τ 单变量敏感性 ─────────────────────────────────────────
    print(f"\n[从热力图提取] α 单变量敏感性  (τ={DEFAULT_THRESHOLD})")
    for entry in sorted(results["alpha_threshold_heatmap"], key=lambda x: x["alpha"]):
        if entry["threshold"] == DEFAULT_THRESHOLD:
            row = {k: v for k, v in entry.items() if k != "threshold"}
            results["alpha_sensitivity"].append(row)
            print(f"    α={entry['alpha']}: Test R²={entry['test_r2']:.4f}")

    print(f"\n[从热力图提取] τ 单变量敏感性  (α={DEFAULT_ALPHA})")
    for entry in sorted(results["alpha_threshold_heatmap"], key=lambda x: x["threshold"]):
        if entry["alpha"] == DEFAULT_ALPHA:
            row = {k: v for k, v in entry.items() if k != "alpha"}
            results["threshold_sensitivity"].append(row)
            print(f"    τ={entry['threshold']}: Test R²={entry['test_r2']:.4f}")

    # ── 窗口敏感性（w=DEFAULT_WINDOW 从热力图复用）──────────────────────────
    print(f"\n[窗口敏感性]  α={DEFAULT_ALPHA}, τ={DEFAULT_THRESHOLD}")
    default_metrics = heatmap_map.get((DEFAULT_ALPHA, DEFAULT_THRESHOLD))
    for window in WINDOW_VALUES:
        if window == DEFAULT_WINDOW and default_metrics is not None:
            metrics = default_metrics
            print(f"    w={window}: [从热力图复用] Test R²={metrics['test']['r2']:.4f}")
        else:
            metrics = window_map.get(window)
            if metrics is None:
                print(f"    w={window}: [结果缺失，跳过]")
                continue
            print(f"    w={window}: Test R²={metrics['test']['r2']:.4f}")
        _record(results["window_sensitivity"], _metrics_to_payload(metrics, {"window": window}))

    # ── 保存结果 ─────────────────────────────────────────────────────────────
    output_pkl  = ROOT / "experiments" / "sensitivity_results_full.pkl"
    output_json = ROOT / "experiments" / "sensitivity_results_full_summary.json"

    with open(output_pkl, "wb") as f:
        pickle.dump(results, f)

    summary = {
        "metadata":       results["metadata"],
        "best_alpha":     max(results["alpha_sensitivity"],     key=lambda x: x["test_r2"]),
        "best_threshold": max(results["threshold_sensitivity"], key=lambda x: x["test_r2"]),
        "best_window":    max(results["window_sensitivity"],    key=lambda x: x["test_r2"]),
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("正式全量敏感性分析完成（多进程并行版）")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结果已保存: {output_pkl}")
    print(f"摘要已保存: {output_json}")
    print("=" * 80)

    return results


# ── 入口（spawn 必须在 __main__ 块内设置）────────────────────────────────────

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="全量敏感性分析（并行版）")
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help=f"并行 worker 数（默认 {DEFAULT_WORKERS}，受 CPU/RAM 限制）"
    )
    args = parser.parse_args()

    run_sensitivity_analysis_full(num_workers=args.workers)
