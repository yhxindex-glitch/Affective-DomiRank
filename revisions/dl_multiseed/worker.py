"""One recorded run; original baselines or validation-selected GAT tuning/refit."""
from __future__ import annotations

import argparse
import csv
import ctypes
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from engine import CachedData, DataLoader, WORK, OUT, evaluate, make_model, np, optim, source_namespace, torch, train_epoch


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save_json(path, obj):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def save_predictions(path, data, ids, pred, truth):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'node', 'input_years', 'target_year', 'observed', 'predicted'])
        for i, p, y in zip(ids, pred, truth):
            node = data.raw['nodes'][int(data.raw['node_indices'][i])]
            years = data.raw['window_years'][int(data.windows[i])]
            writer.writerow([i, node, '|'.join(map(str, years)), data.raw['target_years'][i], float(y), float(p)])


def run(job):
    destination = Path(job['directory'])
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'metrics.json').exists() or (destination / 'training_history.csv').exists():
        raise RuntimeError('This run already has results/history. Refusing to overwrite or repeat it.')
    name, config, seed, mode = job['model'], job['config'], job['seed'], job['mode']
    save_json(destination / 'status.json', {'status': 'STARTING', 'pid': os.getpid(), 'updated_at': stamp(), 'job_id': job['id']})
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Bound cached CUDA blocks so three workers can share the 32 GB GPU.
    # This affects future workers only; active runs are not restarted.
    torch.cuda.set_per_process_memory_fraction(0.22, 0)
    required_gib = {'mlp': 0.6, 'lstm': 0.7, 'gcn': 1.2, 'gcn_lstm': 1.3,
                    'gat_lstm': 3.4, 'dcrnn': 3.4, 'mtgnn': 1.2, 'gat': 3.4}[name]
    if name == 'gat' and config.get('heads', 2) == 4:
        required_gib = 6.2
    while True:
        available, _ = torch.cuda.mem_get_info()
        if available / 2 ** 30 >= required_gib:
            break
        save_json(destination / 'status.json', {'status': 'WAITING_FOR_GPU', 'job_id': job['id'],
                  'pid': os.getpid(), 'updated_at': stamp(), 'free_gpu_gib': available / 2 ** 30,
                  'required_free_gpu_gib': required_gib})
        print(f'Waiting for GPU memory: free={available / 2 ** 30:.2f} GiB, required={required_gib:.2f} GiB', flush=True)
        time.sleep(5)
    manifest = json.loads((WORK / 'source_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(Path(manifest['source']).read_bytes()).hexdigest() == manifest['candidate_sha256']
    data_manifest = json.loads((OUT / 'dataset_manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((OUT / 'dataset_cache.pt').read_bytes()).hexdigest() == data_manifest['cache_sha256']
    data = CachedData()
    # All five repetitions keep the baseline's split fixed; seeds affect initialization/training.
    train_ids, evaluation_ids = data.split(seed=42)
    held_out_ids = evaluation_ids.copy()
    if mode == 'gat_tuning':
        random.Random(20260922).shuffle(train_ids)
        validation_count = len(train_ids) // 5
        validation_ids, train_ids = train_ids[:validation_count], train_ids[validation_count:]
        evaluation_ids = validation_ids
        metric_prefix = 'validation'
        selection_rule = 'Highest internal-validation R2; original 1440 evaluation samples are not accessed.'
    elif mode == 'gat_refit':
        metric_prefix = 'evaluation'
        selection_rule = 'Fixed configuration and epoch count chosen using internal validation; evaluate once after refit.'
    else:
        metric_prefix = 'evaluation'
        selection_rule = 'Preserved baseline runner: highest evaluation R2, patience 15.'
    if job.get('smoke'):
        train_ids, evaluation_ids = train_ids[:64], evaluation_ids[:64]
    assert not set(train_ids).intersection(evaluation_ids)
    if mode == 'gat_tuning':
        assert not (set(train_ids) | set(evaluation_ids)).intersection(held_out_ids)
    train_loader = DataLoader(train_ids, batch_size=config['batch_size'], shuffle=True, num_workers=0)
    train_eval_loader = DataLoader(train_ids, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    evaluation_loader = DataLoader(evaluation_ids, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    model = make_model(name, config, data.num_nodes, source_namespace()).to('cuda')
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)
    metadata = {**job, 'started_at': stamp(), 'pid': os.getpid(), 'split_seed': 42,
                'selection_rule': selection_rule, 'precision': 'FP32', 'gpu': torch.cuda.get_device_name(0),
                'torch': torch.__version__, 'source_sha256': manifest['candidate_sha256'],
                'engine_sha256': hashlib.sha256((WORK / 'engine.py').read_bytes()).hexdigest(),
                'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'dataset_sha256': data_manifest['cache_sha256'],
                'train_sample_count': len(train_ids), 'evaluation_sample_count': len(evaluation_ids),
                'metric_partition': metric_prefix, 'training_seeds_change_data_split': False,
                'allocator_memory_fraction': 0.22, 'startup_required_free_gpu_gib': required_gib,
                'graph_alignment_fix_models': ['gcn', 'gat', 'gcn_lstm', 'gat_lstm'],
                'worker_exit_strategy': 'After CUDA synchronization, closed result files and flushed streams, TerminateProcess on this completed Windows child only; avoids observed LSTM DLL teardown failure 0xc0000409. Other platforms use os._exit.',
                'scheduler': {'name': 'CosineAnnealingWarmRestarts', 'T_0': 20, 'T_mult': 2}}
    save_json(destination / 'run_metadata.json', metadata)
    save_json(destination / 'sample_ids.json', {'training': train_ids, metric_prefix: evaluation_ids})
    print(f"{job['id']} | train={len(train_ids)} {metric_prefix}={len(evaluation_ids)} | {selection_rule}", flush=True)
    started = time.perf_counter()
    best_score, best_epoch, stale, best_metrics = -math.inf, None, 0, None
    epochs = job.get('fixed_epochs', config['max_epochs'])
    if job.get('smoke'):
        epochs = 1
    history_path = destination / 'training_history.csv'
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[0]['lr']
        loss = train_epoch(model, name, train_loader, data, optimizer, microbatch=job['microbatch'])
        scheduler.step()
        training, _, _, _ = evaluate(model, name, train_eval_loader, data)
        # GAT final repetitions never inspect final evaluation scores during fitting.
        observed = pred = truth = sample_ids = None
        if mode != 'gat_refit' or epoch == epochs:
            observed, pred, truth, sample_ids = evaluate(model, name, evaluation_loader, data)
        assert math.isfinite(loss) and all(math.isfinite(v) for v in training.values()), 'Non-finite training values'
        if observed is not None:
            assert all(math.isfinite(v) for v in observed.values()), 'Non-finite evaluation values'
        improved = observed is not None and (mode == 'gat_refit' or observed['R2'] > best_score)
        if improved:
            best_score, best_epoch, stale = observed['R2'], epoch, 0
            best_metrics = {'train': training, metric_prefix: observed}
            torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, destination / 'best_model.pt')
            save_predictions(destination / 'best_predictions.csv', data, sample_ids, pred, truth)
        elif mode != 'gat_refit':
            stale += 1
        record = {'epoch': epoch, 'learning_rate': learning_rate, 'train_loss_normalized': loss,
                  **{'train_' + k: v for k, v in training.items()},
                  **{metric_prefix + '_' + k: observed[k] if observed is not None else '' for k in ('MAE', 'RMSE', 'R2')},
                  'best_epoch': best_epoch, 'best_score': best_score if math.isfinite(best_score) else '',
                  'seconds': time.perf_counter() - epoch_start}
        with history_path.open('a', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if epoch == 1:
                writer.writeheader()
            writer.writerow(record)
        # Return unused blocks between epochs so other queued workers can start.
        torch.cuda.empty_cache()
        progress = {'status': 'RUNNING', 'job_id': job['id'], 'pid': os.getpid(), 'updated_at': stamp(),
                    'completed_epochs': epoch, 'max_epochs': epochs, 'train_R2': training['R2'],
                    'metric_partition': metric_prefix, 'best_epoch': best_epoch,
                    'best_R2': best_score if math.isfinite(best_score) else None,
                    'seconds': time.perf_counter() - started,
                    'allocated_gpu_gib': torch.cuda.memory_allocated() / 2 ** 30,
                    'reserved_gpu_gib': torch.cuda.memory_reserved() / 2 ** 30}
        save_json(destination / 'status.json', progress)
        score_text = f"{metric_prefix} R2={observed['R2']:.6f} best={best_score:.6f}" if observed is not None else 'fixed-epoch refit'
        print(f"epoch {epoch}/{epochs} train R2={training['R2']:.6f} {score_text} time={record['seconds']:.2f}s", flush=True)
        if mode != 'gat_refit' and stale >= config['patience']:
            break
    assert best_metrics is not None
    # Recompute saved metrics from the stored predictions, independently in float64.
    rows = list(csv.DictReader((destination / 'best_predictions.csv').open(encoding='utf-8')))
    y = np.array([float(r['observed']) for r in rows], dtype=np.float64)
    p = np.array([float(r['predicted']) for r in rows], dtype=np.float64)
    checked = {'MAE': float(np.abs(y - p).mean()), 'RMSE': float(np.sqrt(np.square(y - p).mean())),
               'R2': float(1 - np.square(y - p).sum() / np.square(y - y.mean()).sum())}
    difference = max(abs(checked[k] - best_metrics[metric_prefix][k]) for k in checked)
    assert difference < 1e-5, difference
    torch.cuda.synchronize()
    result = {'status': 'COMPLETE', 'job_id': job['id'], 'model': name, 'seed': seed, 'mode': mode,
              'config': config, 'best_epoch': best_epoch, 'stopped_epoch': epoch,
              'selection_rule': selection_rule, 'duration_seconds': time.perf_counter() - started,
              'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2 ** 30,
              'prediction_file_metric_check_max_error': difference, **best_metrics}
    save_json(destination / 'metrics.json', result)
    save_json(destination / 'status.json', {**progress, 'status': 'COMPLETE', 'updated_at': stamp(), **result})
    print(json.dumps(result, ensure_ascii=False), flush=True)
    # Bypass the local Windows DLL-detach failure after a completed LSTM run.
    # All writes use closed context managers and CUDA has synchronized above.
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == 'nt':
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        kernel.TerminateProcess.restype = ctypes.c_int
        if not kernel.TerminateProcess(kernel.GetCurrentProcess(), 0):
            raise ctypes.WinError(ctypes.get_last_error())
    os._exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = json.loads(Path(args.job).read_text(encoding='utf-8'))
    try:
        run(job)
    except Exception as exc:
        error = traceback.format_exc()
        print(error, flush=True)
        destination = Path(job['directory'])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / 'error.txt').write_text(error, encoding='utf-8')
        save_json(destination / 'status.json', {'status': 'FAILED', 'job_id': job['id'], 'pid': os.getpid(),
                                              'updated_at': stamp(), 'error': str(exc)})
        raise
