"""List, check inputs for, and launch the packaged experiment entry points."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
NETWORK = 'data/processed/networks/all_networks_2017_2021.pkl'
LABELS = 'data/processed/networks/risk_labels.pkl'
TASKS = {
    'trade': ('data_processing/01_process_baci_trade.py', ['data/raw/archive/country_codes.csv', 'data/raw/archive/product_codes.csv', *[f'data/raw/archive/Year_{y}/Year_{y}.csv' for y in range(2017, 2022)]]),
    'news': ('data_processing/02_process_business_news.py', ['data/raw/business_news/business_news_sentiment.csv']),
    'networks': ('data_processing/03_build_yearly_networks.py', ['data/processed/news_sentiment/news_sentiment_complete.csv', 'data/processed/baci/baci_trade_2017_2021.csv']),
    'labels': ('build_risk_labels.py', [NETWORK]),
    'main': ('experiments/exp4_stgnn_temporal.py', [NETWORK, LABELS]),
    'main-seeds': ('experiments/run_significance_test.py', [NETWORK, LABELS]),
    'baselines': ('experiments/run_baselines.py', [NETWORK, LABELS]),
    'ml': ('experiments/run_ml_baselines.py', [NETWORK, LABELS]),
    'ml-seeds': ('experiments/run_ml_multiseed.py', [NETWORK, LABELS]),
    'traditional': ('experiments/run_ablation_v2_domirank.py', [NETWORK, LABELS]),
    'no-domirank': ('experiments/run_ablation_v3_no_domirank.py', [NETWORK, LABELS]),
    'no-sentiment': ('experiments/run_ablation_v4_no_sentiment.py', [NETWORK, LABELS]),
    'target-study': ('experiments/run_target_study.py', [NETWORK, LABELS]),
    'sensitivity': ('experiments/run_sensitivity_analysis_full.py', [NETWORK, LABELS]),
    'mapping': ('experiments/run_mapping_coeff_sensitivity.py', [NETWORK, LABELS]),
    'case-energy': ('experiments/run_case_study_energy_subnetwork.py', [NETWORK]),
    'case-attributes': ('experiments/run_case_study_attribute_baseline.py', [NETWORK]),
    'stability': ('revisions/network_analysis/stability.py', [NETWORK, *[f'data/processed/networks/network_{y}.graphml' for y in range(2017, 2022)]]),
    'decomposition': ('revisions/network_analysis/decomposition.py', ['data/processed/networks/network_2020.graphml', 'analysis/results/case_study_attribute_baseline.json', 'results/network_analysis/stability_check.json']),
    'node-removal': ('revisions/network_analysis/removal.py', ['data/processed/networks/network_2020.graphml', 'analysis/results/energy_minerals_case_study.json']),
    'node-backbone': ('revisions/network_analysis/backbone.py', ['results/node_removal/results.json', 'results/node_removal/removal_orders.csv', 'results/network_analysis/decomposition.json', 'results/network_analysis/all_node_decomposition.json']),
    'dl-cache': ('revisions/dl_multiseed/prepare.py', [NETWORK, LABELS]),
    'dl-plan': ('revisions/dl_multiseed/create_plan.py', ['results/dl_multiseed/dataset_cache.pt']),
    'dl-seeds': ('revisions/dl_multiseed/supervisor.py', ['results/dl_multiseed/plan.json']),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', choices=TASKS)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--check', action='store_true', help='Check local inputs without importing training libraries.')
    args, extra = parser.parse_known_args()
    if args.list or args.task is None:
        for name, (path, _) in TASKS.items():
            print(f'{name:18} {path}')
        return
    path, needed = TASKS[args.task]
    missing = [name for name in needed if not (ROOT / name).is_file()]
    if missing:
        print('Required local files are missing:\n' + '\n'.join('  ' + name for name in missing))
        raise SystemExit(2)
    if args.check:
        print(f'Inputs are ready for {args.task}.')
        return
    os.chdir(ROOT)
    for name in ['analysis/results', 'figures', 'figures_cn', 'results']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str((ROOT / path).parent))
    sys.argv = [str(ROOT / path)] + (extra[1:] if extra[:1] == ['--'] else extra)
    runpy.run_path(str(ROOT / path), run_name='__main__')


if __name__ == '__main__':
    main()
