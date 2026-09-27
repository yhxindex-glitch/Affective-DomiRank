"""Persistent local queue. Preserve every trial; GAT selection reads validation only."""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

WORK = Path(__file__).resolve().parent
OUT = WORK.parents[1] / 'results' / 'dl_multiseed'


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, PermissionError):
        return default


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def collect(jobs):
    records = []
    for job in jobs:
        directory = Path(job['directory'])
        status = read_json(directory / 'status.json', {'status': 'PENDING'})
        records.append({'id': job['id'], 'model': job['model'], 'seed': job['seed'], 'mode': job['mode'],
                        'directory': str(directory), **status})
    return records


def summary(jobs, plan):
    completed = []
    for job in jobs:
        if job['mode'] == 'gat_tuning':
            continue
        result = read_json(Path(job['directory']) / 'metrics.json')
        if result is not None:
            completed.append(result)
    with (OUT / 'individual_results.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=['model', 'seed', 'MAE', 'RMSE', 'R2', 'best_epoch', 'stopped_epoch', 'mode'])
        writer.writeheader()
        for result in completed:
            writer.writerow({**{k: result[k] for k in ['model', 'seed', 'best_epoch', 'stopped_epoch', 'mode']}, **result['evaluation']})
    rows = []
    for model in plan['models']:
        subset = [r for r in completed if r['model'] == model]
        row = {'model': model, 'completed_seeds': len(subset), 'planned_seeds': len(plan['seeds']),
               'five_seed_complete': len(subset) == len(plan['seeds'])}
        for metric in ['MAE', 'RMSE', 'R2']:
            values = [r['evaluation'][metric] for r in subset]
            row[metric + '_mean'] = statistics.mean(values) if values else None
            row[metric + '_sample_std'] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    save_json(OUT / 'summary.json', {'updated_at': now(), 'rows': rows, 'std_rule': 'Sample standard deviation, ddof=1.',
                                    'protocols': {'gat': plan['gat_refit_rule'], 'other_models': plan['other_models_rule']}})
    return rows


def enqueue_gat(jobs, plan):
    existing = read_json(OUT / 'gat_selected_config.json')
    if existing is not None:
        selected_jobs = existing['refit_jobs']
        known_ids = {j['id'] for j in jobs}
        jobs.extend(j for j in selected_jobs if j['id'] not in known_ids)
        return True
    candidates = [j for j in jobs if j['mode'] == 'gat_tuning']
    states = [read_json(Path(j['directory']) / 'status.json', {}) for j in candidates]
    if not all(s.get('status') in {'COMPLETE', 'FAILED'} for s in states):
        return False
    finished = [read_json(Path(j['directory']) / 'metrics.json') for j in candidates]
    finished = [r for r in finished if r is not None]
    if not finished:
        save_json(OUT / 'gat_selection_failed.json', {'status': 'FAILED', 'reason': 'No completed tuning candidates.'})
        return True
    winner = max(finished, key=lambda r: r['validation']['R2'])
    refits = []
    for seed in plan['seeds']:
        job_id = f'gat_seed_{seed}'
        directory = OUT / 'runs' / job_id
        directory.mkdir(parents=True, exist_ok=False)
        job = {'id': job_id, 'model': 'gat', 'seed': seed, 'mode': 'gat_refit', 'config': winner['config'],
               'fixed_epochs': winner['best_epoch'], 'microbatch': plan['microbatch'], 'directory': str(directory),
               'selected_from_trial': winner['job_id']}
        save_json(directory / 'job.json', job)
        refits.append(job)
    selection = {'selected_at': now(), 'selection_metric': 'internal-validation R2',
                 'selected_trial': winner['job_id'], 'selected_config': winner['config'],
                 'fixed_refit_epochs': winner['best_epoch'], 'validation': winner['validation'],
                 'completed_candidates': len(finished), 'planned_candidates': len(candidates),
                 'all_candidate_scores': [{'job_id': r['job_id'], 'validation': r['validation'],
                                          'best_epoch': r['best_epoch']} for r in finished],
                 'final_evaluation_used_for_selection': False, 'refit_jobs': refits}
    save_json(OUT / 'gat_selected_config.json', selection)
    # Start the selected GAT repetitions at the next free worker slots.
    first_pending = next((i for i, j in enumerate(jobs) if not (Path(j['directory']) / 'status.json').exists()), len(jobs))
    jobs[first_pending:first_pending] = refits
    print(f"Selected GAT {winner['job_id']} validation R2={winner['validation']['R2']:.6f}; queued five fixed-epoch refits", flush=True)
    return True


def main():
    plan = read_json(OUT / 'plan.json')
    lock = OUT / 'supervisor.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(os.getpid()))
    jobs = list(plan['jobs'])
    active = {}
    try:
        gat_resolved = False
        while True:
            for job_id, (process, stdout, stderr, job) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                stdout.close()
                stderr.close()
                status_path = Path(job['directory']) / 'status.json'
                status = read_json(status_path, {})
                if code != 0 or status.get('status') != 'COMPLETE':
                    save_json(status_path, {**status, 'status': 'FAILED', 'exit_code': code, 'updated_at': now()})
                print(f"Finished {job_id}: exit={code} status={read_json(status_path)['status']}", flush=True)
                del active[job_id]
            if not gat_resolved:
                gat_resolved = enqueue_gat(jobs, plan)
            for job in jobs:
                if len(active) >= plan['parallel_workers']:
                    break
                status_path = Path(job['directory']) / 'status.json'
                if job['id'] in active or status_path.exists():
                    continue
                directory = Path(job['directory'])
                stdout = (directory / 'training.log').open('a', encoding='utf-8')
                stderr = (directory / 'stderr.log').open('a', encoding='utf-8')
                save_json(status_path, {'status': 'STARTING', 'job_id': job['id'], 'updated_at': now()})
                process = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', str(WORK / 'worker.py'),
                                            '--job', str(directory / 'job.json')], cwd=str(WORK.parents[1]),
                                           stdout=stdout, stderr=stderr, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
                                           env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'PYTHONUNBUFFERED': '1'})
                active[job['id']] = (process, stdout, stderr, job)
                print(f"Started {job['id']} pid={process.pid}", flush=True)
            records = collect(jobs)
            rows = summary(jobs, plan)
            counts = {key: sum(r['status'] == key for r in records) for key in ['COMPLETE', 'FAILED', 'PENDING', 'STARTING', 'RUNNING']}
            done = not active and counts['PENDING'] == 0 and gat_resolved
            state = {'status': ('COMPLETE_WITH_FAILURES' if counts['FAILED'] or (OUT / 'gat_selection_failed.json').exists() else 'COMPLETE') if done else 'RUNNING',
                     'supervisor_pid': os.getpid(), 'updated_at': now(), 'parallel_workers': plan['parallel_workers'],
                     'total_planned_jobs': plan['formal_run_count'] + plan['tuning_run_count'],
                     'gat_refits_waiting_for_selection': 0 if gat_resolved else len(plan['seeds']),
                     'counts': counts, 'active_job_ids': list(active), 'jobs': records}
            save_json(OUT / 'queue_status.json', state)
            lines = ['# 深度学习多种子训练进度', '', f"状态：{state['status']}。并行进程：{plan['parallel_workers']}。",
                     f"已完成：{counts['COMPLETE']}；运行中：{len(active)}；失败：{counts['FAILED']}。计划共40次正式训练及4次GAT调参。", '',
                     '| 模型 | 已完成种子 | 平均 R² | R² 标准差 |', '|---|---:|---:|---:|']
            for row in rows:
                avg = f"{row['R2_mean']:.6f}" if row['R2_mean'] is not None else '待完成'
                std = f"{row['R2_sample_std']:.6f}" if row['R2_sample_std'] is not None else '—'
                lines.append(f"| {row['model']} | {row['completed_seeds']}/5 | {avg} | {std} |")
            lines.extend(['', '五个种子全部完成后再使用最终均值与标准差。GAT配置按内部验证集选择。', '',
                          '正在运行：' + '、'.join(active), '', f"更新时间（UTC）：{state['updated_at']}"])
            (OUT / 'progress.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
            if done:
                print(json.dumps({k: v for k, v in state.items() if k != 'jobs'}, ensure_ascii=False), flush=True)
                return
            time.sleep(3)
    finally:
        # Only remove this supervisor's small lock file. Completed run files stay intact.
        if lock.exists() and lock.read_text(encoding='utf-8') == str(os.getpid()):
            lock.unlink()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        (OUT / 'supervisor_error.txt').write_text(error, encoding='utf-8')
        print(error, flush=True)
        raise
