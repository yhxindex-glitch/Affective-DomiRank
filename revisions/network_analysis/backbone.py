"""Exploratory disparity-backbone dismantling check with all fixed levels."""
from pathlib import Path
from xml.etree import ElementTree as ET
import csv
import json
import sys
import time
import numpy as np

BASE = Path(__file__).resolve().parents[2]
WORK = Path(__file__).resolve().parent
OUT = BASE / 'results/node_removal/backbone_check'
OLD = BASE / 'results/node_removal'
sys.path.insert(0, str(WORK))
import removal


def filter_backbone(W, level):
    positive = W > 0
    out_degree = positive.sum(axis=1)
    in_degree = positive.sum(axis=0)
    out_strength = W.sum(axis=1)
    in_strength = W.sum(axis=0)
    out_share = np.divide(W, out_strength[:, None], out=np.zeros_like(W), where=out_strength[:, None] > 0)
    in_share = np.divide(W, in_strength[None, :], out=np.zeros_like(W), where=in_strength[None, :] > 0)
    out_tail = np.ones_like(W)
    in_tail = np.ones_like(W)
    mask = out_degree > 1
    out_tail[mask] = (1 - np.clip(out_share[mask], 0, 1)) ** (out_degree[mask, None] - 1)
    mask = in_degree > 1
    in_tail[:, mask] = (1 - np.clip(in_share[:, mask], 0, 1)) ** (in_degree[None, mask] - 1)
    both_single = (out_degree[:, None] == 1) & (in_degree[None, :] == 1)
    return positive & ((out_tail < level) | (in_tail < level) | both_single)


def main():
    started = time.perf_counter()
    protocol_path = WORK / 'backbone_protocol.json'
    protocol = json.loads(protocol_path.read_text('utf-8'))
    previous = json.loads((OLD / 'results.json').read_text('utf-8'))
    protected = {Path(p): h for p, h in previous['source_hashes'].items()}
    protected.update({p: removal.sha(p) for p in OLD.rglob('*') if p.is_file() and OUT not in p.parents})
    protected[protocol_path] = removal.sha(protocol_path)
    assert all(removal.sha(p) == h for p, h in protected.items())
    names, attrs, original_B = removal.read_graph()
    ix = {name: i for i, name in enumerate(names)}
    root = ET.parse(removal.SOURCE).getroot()
    keys = {x.get('id'): x.get('attr.name') for x in root.findall('g:key', removal.NS)}
    W = np.zeros(original_B.shape)
    for edge in root.find('g:graph', removal.NS).findall('g:edge', removal.NS):
        a, b = edge.get('source'), edge.get('target')
        if a in ix and b in ix:
            data = {keys[d.get('key')]: d.text for d in edge.findall('g:data', removal.NS)}
            W[ix[a], ix[b]] = float(data['weight'])
    assert np.array_equal(W > 0, original_B)
    with (OLD / 'removal_orders.csv').open(encoding='utf-8-sig', newline='') as f:
        old_orders = list(csv.DictReader(f))
    decomp_dir = BASE / 'results/network_analysis'
    decomp_meta = json.loads((decomp_dir / 'decomposition.json').read_text('utf-8'))
    assert decomp_meta['source_hashes'][str(removal.SOURCE)] == removal.sha(removal.SOURCE)
    decomp = {r['node']: r for r in json.loads((decomp_dir / 'all_node_decomposition.json').read_text('utf-8'))}
    forcing = np.array([decomp[name]['sentiment_mapping_at_centrality_stage'] for name in names])
    for name in names:
        assert abs(decomp[name]['stored_total'] - float(attrs[name]['affective_domirank'])) < 1e-12
    levels = sorted([protocol['main_filter_level']] + protocol['sensitivity_filter_levels'])
    scopes = ['Combined', 'Energy', 'Minerals']
    families = ['Existing case rankings', 'Recomputed backbone rankings']
    seed_children = np.random.SeedSequence(protocol['random_reference']['seed']).spawn(3)
    tie_children = np.random.SeedSequence(protocol['tie_sensitivity']['seed']).spawn(18)
    result = {'protocol': protocol, 'protocol_sha256': removal.sha(protocol_path),
              'source_hashes': {str(p): h for p, h in protected.items()}, 'conditions': []}
    curves, summaries, topology, ranks, ties = [], [], [], [], []
    checked_curves = 0
    OUT.mkdir(parents=True, exist_ok=True)
    for si, scope in enumerate(scopes):
        chosen = [i for i, name in enumerate(names) if scope == 'Combined' or attrs[name]['industry'] == scope]
        snames = [names[i] for i in chosen]
        subW = W[np.ix_(chosen, chosen)]
        n = len(snames)
        sub_forcing = forcing[chosen]
        base_scores, base_orders = {}, {}
        for method in removal.METHODS:
            rows = [r for r in old_orders if r['scope'] == scope and r['method'] == method]
            base_orders[method] = [snames.index(r['node']) for r in rows]
            score_map = {r['node']: float(r['score']) for r in rows}
            base_scores[method] = np.array([score_map[name] for name in snames])
        rng = np.random.default_rng(seed_children[si])
        random_orders = [list(map(int, rng.permutation(n))) for _ in range(protocol['random_reference']['permutations'])]
        prior = np.zeros(subW.shape, dtype=bool)
        for li, level in enumerate(levels):
            B = filter_backbone(subW, level)
            assert np.all(~prior | B)
            prior = B
            assert np.array_equal(B, filter_backbone(subW * 37.5, level))
            assert np.array_equal(B.T, filter_backbone(subW.T, level))
            pre = (removal.masks(B | B.T), removal.masks(B), removal.masks(B.T))
            initial = removal.closure_components(B)
            assert initial[0] > 0 and initial[1] > 0
            degrees = B.sum(axis=0) + B.sum(axis=1)
            info = {'scope': scope, 'filter_level': level, 'original_nodes': n,
                    'original_edges': int((subW > 0).sum()), 'retained_edges': int(B.sum()),
                    'retained_edge_fraction': float(B.sum() / (subW > 0).sum()),
                    'retained_trade_fraction': float(subW[B].sum() / subW.sum()),
                    'nodes_with_retained_links': int(np.sum(degrees > 0)), 'isolates_retained': int(np.sum(degrees == 0)),
                    'initial_largest_weak': int(initial[0]), 'initial_largest_strong': int(initial[1]),
                    'original_degree_one_out_endpoints': int(np.sum((subW > 0).sum(axis=1) == 1)),
                    'original_degree_one_in_endpoints': int(np.sum((subW > 0).sum(axis=0) == 1))}
            topology.append(info)
            random_values = np.stack([removal.curve(order, B, pre) for order in random_orders])
            q = np.arange(n + 1) / n
            random_auc = np.trapezoid(random_values / random_values[:, 0:1, :], q, axis=1)
            random_summary = {metric: removal.dist_summary(random_auc[:, j]) for j, metric in enumerate(['weak', 'strong'])}
            np.savez_compressed(OUT / f'{scope}_{level}_random_curves.npz', counts=random_values)
            for stat, values in [('mean', random_values.mean(axis=0)), ('p025', np.quantile(random_values, .025, axis=0)), ('p975', np.quantile(random_values, .975, axis=0))]:
                for k, value in enumerate(values):
                    curves.append({'scope': scope, 'filter_level': level, 'ranking_family': 'Shared random reference',
                                   'method': 'Random ' + stat, 'removed': k, 'removed_fraction': k / n,
                                   'weak_count': float(value[0]), 'strong_count': float(value[1]),
                                   'weak_initial_fraction': float(value[0] / initial[0]),
                                   'strong_initial_fraction': float(value[1] / initial[1])})
            M = .5 * np.eye(n) + .01 * B.T
            rhs = .01 * B.T @ np.ones(n)
            recomputed = {'Traditional DomiRank': np.linalg.solve(M, rhs),
                          'Affective DomiRank': np.linalg.solve(M, rhs + .1 * sub_forcing),
                          'Total degree': degrees}
            recomputed['PageRank'], _ = removal.pagerank(B)
            assert max(np.linalg.eigvals(-M).real) < 0
            assert np.max(abs(M @ recomputed['Traditional DomiRank'] - rhs)) < 1e-12
            assert np.max(abs(M @ recomputed['Affective DomiRank'] - rhs - .1 * sub_forcing)) < 1e-12
            for fi, family in enumerate(families):
                scores = base_scores if fi == 0 else recomputed
                condition = {**info, 'ranking_family': family, 'methods': {}, 'random_reference': random_summary, 'tie_sensitivity': {}}
                for method in removal.METHODS:
                    order = base_orders[method] if fi == 0 else sorted(range(n), key=lambda i: float(scores[method][i]), reverse=True)
                    values = removal.curve(order, B, pre)
                    removal.verify_curve(order, B, values)
                    checked_curves += 1
                    sm = removal.summarize(values)
                    condition['methods'][method] = sm
                    for metric, row in sm.items():
                        summaries.append({'scope': scope, 'filter_level': level, 'ranking_family': family,
                                          'method': method, 'metric': metric, **row})
                    for rank, i in enumerate(order, 1):
                        ranks.append({'scope': scope, 'filter_level': level, 'ranking_family': family,
                                      'method': method, 'rank': rank, 'node': snames[i], 'score': float(scores[method][i])})
                    for k, value in enumerate(values):
                        curves.append({'scope': scope, 'filter_level': level, 'ranking_family': family,
                                       'method': method, 'removed': k, 'removed_fraction': k / n,
                                       'weak_count': int(value[0]), 'strong_count': int(value[1]),
                                       'weak_initial_fraction': float(value[0] / initial[0]),
                                       'strong_initial_fraction': float(value[1] / initial[1])})
                rng = np.random.default_rng(tie_children[si * 6 + li * 2 + fi])
                for method in removal.FIELDS:
                    tv = np.stack([removal.curve(removal.permute_ties(scores[method], rng), B, pre)
                                   for _ in range(protocol['tie_sensitivity']['permutations'])])
                    ta = np.trapezoid(tv / tv[:, 0:1, :], q, axis=1)
                    condition['tie_sensitivity'][method] = {metric: removal.dist_summary(ta[:, j]) for j, metric in enumerate(['weak', 'strong'])}
                    for metric, row in condition['tie_sensitivity'][method].items():
                        ties.append({'scope': scope, 'filter_level': level, 'ranking_family': family,
                                     'method': method, 'metric': metric, **row})
                result['conditions'].append(condition)
                print(json.dumps({**info, 'ranking_family': family,
                      'affective': condition['methods']['Affective DomiRank'],
                      'traditional': condition['methods']['Traditional DomiRank']}, ensure_ascii=False), flush=True)
    assert all(removal.sha(p) == h for p, h in protected.items())
    result['verification'] = {'input_files_and_prior_results_unchanged': True,
                              'deterministic_curves_independently_verified': checked_curves,
                              'all_predeclared_filter_levels_and_scopes_reported': True,
                              'scale_invariance_and_transpose_symmetry_verified': True,
                              'nested_backbones_verified': True, 'model_parameters_unchanged': True,
                              'training_runs': 0, 'elapsed_seconds': time.perf_counter() - started}
    removal.dump(OUT / 'results.json', result)
    removal.save_csv(OUT / 'backbone_curves.csv', curves)
    removal.save_csv(OUT / 'summary_metrics.csv', summaries)
    removal.save_csv(OUT / 'topology.csv', topology)
    removal.save_csv(OUT / 'removal_orders.csv', ranks)
    removal.save_csv(OUT / 'tie_sensitivity.csv', ties)
    print('BACKBONE_ANALYSIS_COMPLETE', json.dumps(result['verification']), flush=True)


if __name__ == '__main__':
    main()
