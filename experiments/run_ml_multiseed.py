"""Run all requested machine-learning seeds and record newly computed metrics."""
import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 52, 62, 72, 82])
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/ml_multiseed')
    args = parser.parse_args()
    import numpy as np
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    from training.splits import assert_disjoint_indices
    spec = importlib.util.spec_from_file_location('ml_baseline_functions', ROOT / 'experiments/run_ml_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with (ROOT / 'data/processed/networks/all_networks_2017_2021.pkl').open('rb') as stream:
        networks = pickle.load(stream)
    dataset = module.NodeLevelSTGNNDataset(networks, window_size=3, use_interpolation=True)
    X, y, _ = module.prepare_ml_data(dataset)
    order = np.random.RandomState(42).permutation(len(X))
    cut = int(0.8 * len(order))
    train_ids, test_ids = order[:cut], order[cut:]
    assert_disjoint_indices(train_ids, test_ids, len(X))
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'sample_ids.json').write_text(json.dumps({'training': train_ids.tolist(), 'evaluation': test_ids.tolist()}), encoding='utf-8')
    factories = {'random_forest': module.RandomForestRegressor,
                 'xgboost': module.xgb.XGBRegressor, 'lightgbm': module.lgb.LGBMRegressor}
    rows = []
    truth = y[test_ids] * dataset.label_std + dataset.label_mean
    for seed in args.seeds:
        for name, factory in factories.items():
            def estimator(**kwargs):
                kwargs.update(random_state=seed, n_jobs=args.threads)
                return factory(**kwargs)
            if name == 'random_forest':
                module.RandomForestRegressor = estimator
            elif name == 'xgboost':
                module.xgb = SimpleNamespace(XGBRegressor=estimator)
            else:
                module.lgb = SimpleNamespace(LGBMRegressor=estimator)
            np.random.seed(seed)
            result = getattr(module, 'train_' + name)(X[train_ids], y[train_ids], X[test_ids], y[test_ids], dataset)
            predicted = np.asarray(result['test_pred']).reshape(-1)
            difference = predicted.astype(float) - truth.astype(float)
            metrics = {'model': name, 'seed': seed, 'MAE': float(np.abs(difference).mean()),
                       'RMSE': float(np.sqrt(np.square(difference).mean())),
                       'R2': float(1 - np.square(difference).sum() / np.square(truth - truth.mean()).sum())}
            destination = output / f'{name}_seed_{seed}'
            destination.mkdir()
            (destination / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            with (destination / 'predictions.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.writer(stream)
                writer.writerow(['sample_id', 'observed', 'predicted'])
                writer.writerows(zip(test_ids.tolist(), truth.tolist(), predicted.tolist()))
            rows.append(metrics)
    with (output / 'individual_results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['model', 'seed', 'MAE', 'RMSE', 'R2'])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    main()
