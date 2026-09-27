import json
from pathlib import Path

from engine import ALL_MODELS, OUT, SEEDS, WORK, default_config

if (OUT / 'plan.json').exists():
    raise RuntimeError('Plan already exists; will not recreate or overwrite it.')

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=3)
parser.add_argument('--microbatch', type=int, default=16)
args = parser.parse_args()
if args.workers < 1 or not 1 <= args.microbatch <= 32:
    parser.error('workers must be positive and microbatch must be between 1 and 32')
parallel = {'recommended_workers': args.workers, 'microbatch': args.microbatch}
jobs = []


def add_job(name, seed, mode='baseline', overrides=None, tag=None):
    config = default_config(name)
    config.update(overrides or {})
    job_id = tag or f'{name}_seed_{seed}'
    group = 'gat_tuning' if mode == 'gat_tuning' else 'runs'
    directory = OUT / group / job_id
    directory.mkdir(parents=True, exist_ok=False)
    job = {'id': job_id, 'model': name, 'seed': seed, 'mode': mode,
           'config': config, 'microbatch': parallel['microbatch'], 'directory': str(directory)}
    (directory / 'job.json').write_text(json.dumps(job, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    jobs.append(job)


# Interleave GAT tuning with the first seed of the other baselines.
add_job('gat', 42, 'gat_tuning', {'dropout': 0.1, 'patience': 35}, 'gat_d01_lr001_h2')
add_job('mlp', 42)
add_job('lstm', 42)
add_job('gat', 42, 'gat_tuning', {'dropout': 0.0, 'patience': 35}, 'gat_d00_lr001_h2')
add_job('gcn', 42)
add_job('gcn_lstm', 42)
add_job('gat', 42, 'gat_tuning', {'dropout': 0.1, 'learning_rate': 0.0005, 'heads': 4, 'patience': 35}, 'gat_d01_lr0005_h4')
add_job('gat_lstm', 42)
add_job('mtgnn', 42)
add_job('gat', 42, 'gat_tuning', {'dropout': 0.0, 'learning_rate': 0.0005, 'heads': 4, 'patience': 35}, 'gat_d00_lr0005_h4')
add_job('dcrnn', 42)
for seed in SEEDS[1:]:
    for name in ALL_MODELS:
        if name != 'gat':
            add_job(name, seed)

plan = {'parallel_workers': parallel['recommended_workers'], 'microbatch': parallel['microbatch'], 'seeds': SEEDS, 'models': ALL_MODELS,
        'formal_run_count': 40, 'tuning_run_count': 4, 'jobs': jobs,
        'fixed_split_seed': 42, 'effective_batch_size': 32, 'precision': 'FP32',
        'gat_selection_rule': 'Highest internal-validation R2 across four configurations; no final-evaluation scores used for configuration selection.',
        'gat_refit_rule': 'Refit selected configuration on original full training partition for selected validation epoch count, for all five predefined seeds; evaluate once per seed.',
        'other_models_rule': 'Preserve original model hyperparameters and evaluation-based early-stopping rule; only approved graph alignment and checked execution optimizations applied.',
        'gat_five_seed_runs': 'Automatically enqueue after all four tuning candidates finish.',
        'main_model_runner': 'experiments/run_significance_test.py', 'excluded_models': ['Random Forest', 'XGBoost', 'LightGBM']}
(OUT / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'jobs_ready': len(jobs), 'gat_refits_pending_selection': 5,
                  'parallel_workers': plan['parallel_workers'], 'plan': str(OUT / 'plan.json')}, ensure_ascii=False))
