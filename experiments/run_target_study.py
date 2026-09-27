"""Run target-weight and sentiment-input comparisons with the packaged STGNN."""
from __future__ import annotations
import argparse
import ast
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = {'original': [0.40, 0.25, 0.20, 0.15],
           'equal': [0.25, 0.25, 0.25, 0.25],
           'structure80': [0.50, 0.30, 0.12, 0.08],
           'structural': [8 / 13, 5 / 13, 0, 0]}
COMPONENTS = ['supplier_loss_rate', 'degree_decline_rate',
              'negative_sentiment_risk', 'sentiment_deterioration']
REMOVED_FEATURE_LINES = [
    "                        cached_features['sentiment_mean'],\n",
    "                        cached_features['news_count'],\n",
    "                        affective_vulnerability,  # 转换后的脆弱性指数（高=脆弱）\n",
    "                        cached_features['neighbor_avg_sentiment'],\n",
    "                        cached_features['neighbor_avg_risk'],\n",
    "                        sentiment_change_rate,\n",
    "                        sentiment_risk_score,\n",
]


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Experiment template changed: ' + old[:90])
    return text.replace(old, new, 1)


def build_runner(source, labels_path, checkpoint, input_variant):
    code = replace_once(source, "'data/processed/networks/risk_labels.pkl'", repr(labels_path.as_posix()))
    old = "'experiments/best_stgnn_node_level.pt'"
    if code.count(old) != 2:
        raise ValueError('Expected one checkpoint save and one checkpoint load.')
    code = code.replace(old, repr(checkpoint.as_posix()))
    if input_variant == 'structural':
        for line in REMOVED_FEATURE_LINES:
            code = replace_once(code, line, '')
        code = replace_once(code, '        node_feature_dim=16,', '        node_feature_dim=9,')
    tree = ast.parse(code)
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    main.body.extend(ast.parse("return {'train': train_metrics, 'evaluation': test_metrics, 'train_samples': len(train_samples), 'evaluation_samples': len(test_samples)}").body)
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', choices=WEIGHTS, nargs='+', default=['structural'])
    parser.add_argument('--inputs', choices=['all', 'structural'], nargs='+', default=['all', 'structural'])
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--output', type=Path, default=ROOT / 'results/target_study')
    args = parser.parse_args()
    import numpy as np
    import pandas as pd
    import torch
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    labels = pd.read_pickle(ROOT / 'data/processed/networks/risk_labels.pkl')
    source_path = ROOT / 'experiments/exp4_stgnn_temporal.py'
    source = source_path.read_text(encoding='utf-8')
    for scheme in args.weights:
        folder = args.output.resolve() / scheme
        folder.mkdir(parents=True, exist_ok=True)
        target = labels.copy(deep=True)
        target['comprehensive_risk_score'] = sum(w * labels[c] for w, c in zip(WEIGHTS[scheme], COMPONENTS))
        labels_path = folder / 'target_labels.pkl'
        target.to_pickle(labels_path)
        for seed in args.seeds:
            for variant in args.inputs:
                destination = folder / f'{variant}_seed_{seed}'
                destination.mkdir(parents=True, exist_ok=False)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                tree = build_runner(source, labels_path, destination / 'model.pt', variant)
                namespace = {'__name__': 'target_study_run', '__file__': str(source_path)}
                exec(compile(tree, str(source_path), 'exec'), namespace)
                metrics = namespace['main']()
                result = {'seed': seed, 'weight_scheme': scheme, 'weights': WEIGHTS[scheme],
                          'input_variant': variant, **metrics}
                (destination / 'metrics.json').write_text(json.dumps(result, indent=2, default=float), encoding='utf-8')


if __name__ == '__main__':
    main()
