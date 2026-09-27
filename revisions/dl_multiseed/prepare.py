"""Build the tensor cache consumed by the multiseed GPU workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OUT = ROOT / 'results/dl_multiseed'
INPUTS = [ROOT / 'data/processed/networks/all_networks_2017_2021.pkl',
          ROOT / 'data/processed/networks/risk_labels.pkl']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    missing = [str(p.relative_to(ROOT)) for p in INPUTS if not p.is_file()]
    if missing:
        parser.error('Required local inputs: ' + ', '.join(missing))
    if args.check_inputs:
        print('All required local inputs are present.')
        return
    sys.path.insert(0, str(ROOT / 'revisions/runtime'))
    from runtime_support import attach_full_features, build_dataset, torch
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = OUT / 'dataset_cache.pt'
    if cache_path.exists():
        raise FileExistsError('Tensor cache already exists; select a fresh working copy for another dataset.')
    torch.set_num_threads(4)
    dataset = build_dataset(OUT / 'dataset_build.log')
    full = attach_full_features(dataset)
    windows = {sample['window_idx']: sample['input_years'] for sample in dataset.samples}
    years = list(dataset.years)
    year_index = {year: index for index, year in enumerate(years)}
    node_index = {node: index for index, node in enumerate(dataset.all_nodes)}
    cache = {
        'node_features': torch.stack([s['node_features_seq'] for s in dataset.samples]),
        'labels': torch.stack([s['label'] for s in dataset.samples]).reshape(-1),
        'node_indices': torch.tensor([node_index[s['node']] for s in dataset.samples]),
        'window_indices': torch.tensor([s['window_idx'] for s in dataset.samples]),
        'target_years': [s['target_year'] for s in dataset.samples],
        'nodes': list(dataset.all_nodes),
        'window_years': [windows[i] for i in range(len(windows))],
        'window_graph_indices': [[year_index[y] for y in windows[i]] for i in range(len(windows))],
        'full_features': torch.stack([full[i] for i in range(len(full))]),
        'label_mean': float(dataset.label_mean), 'label_std': float(dataset.label_std),
        'graphs': [{'year': y, 'edge_index': dataset.pyg_graphs[y].edge_index,
                    'edge_weight': dataset.pyg_graphs[y].edge_weight} for y in years],
    }
    torch.save(cache, cache_path)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    evidence = {'samples': len(dataset.samples), 'nodes': len(dataset.all_nodes),
                'features_per_step': 16, 'window_steps': 3, 'cache_sha256': sha(cache_path),
                'input_files_sha256': {str(p): sha(p) for p in INPUTS}}
    (OUT / 'dataset_manifest.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    source = ROOT / 'experiments/run_baselines.py'
    manifest = {'source': str(source), 'candidate_sha256': sha(source)}
    (WORK / 'source_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Tensor cache prepared: {cache_path}')


if __name__ == '__main__':
    main()
