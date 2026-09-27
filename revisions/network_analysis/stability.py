"""Read-only check of saved annual graph matrices and stored DomiRank scores.

The input graph and score files are read without modification. The small
industry blocks are used only after verifying there are no cross-industry edges.
"""
from pathlib import Path
import hashlib
import json
import pickle
from collections import defaultdict
from xml.etree import ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT
WORK = ROOT / 'results/network_analysis'
WORK.mkdir(parents=True, exist_ok=True)
NS = {'g': 'http://graphml.graphdrawing.org/xmlns'}
ALPHA, BETA, GAMMA, THETA, DT = 0.01, 0.5, 0.1, 1.0, 0.01
records = []
global_aff, global_base = [], []

class Snapshot:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

class SnapshotReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('networkx.'):
            return Snapshot
        return super().find_class(module, name)

primary_path = SOURCE / 'data/processed/networks/all_networks_2017_2021.pkl'
with primary_path.open('rb') as handle:
    primary = SnapshotReader(handle).load()

for year in range(2017, 2022):
    path = SOURCE / f'data/processed/networks/network_{year}.graphml'
    raw = path.read_bytes()
    root = ET.fromstring(raw)
    keys = {k.attrib['id']: k.attrib['attr.name'] for k in root.findall('g:key', NS)}
    graph = root.find('g:graph', NS)
    nodes = {n.attrib['id']: {keys[d.attrib['key']]: d.text for d in n.findall('g:data', NS)} for n in graph.findall('g:node', NS)}
    assert set(nodes) == set(primary[year]._node)
    for name, attrs in nodes.items():
        for field in ('domirank', 'affective_domirank', 'sentiment_mean'):
            assert float(attrs[field]) == float(primary[year]._node[name][field])
    stored_edges = {(u, v) for u, links in primary[year]._succ.items() for v in links}
    assert stored_edges == {(e.attrib['source'], e.attrib['target']) for e in graph.findall('g:edge', NS)}
    groups = defaultdict(list)
    for name in nodes:
        groups[name.rsplit('_', 1)[1]].append(name)
    indices, mats = {}, {}
    for industry, names in groups.items():
        for i, name in enumerate(names):
            indices[name] = (industry, i)
        mats[industry] = np.zeros((len(names), len(names)), dtype=float)
    weights, stored_norms = [], []
    edge_count = 0
    for edge in graph.findall('g:edge', NS):
        a, b = edge.attrib['source'], edge.attrib['target']
        industry, i = indices[a]
        industry_b, j = indices[b]
        assert industry == industry_b
        assert mats[industry][i, j] == 0
        mats[industry][i, j] = 1.0
        values = {keys[d.attrib['key']]: d.text for d in edge.findall('g:data', NS)}
        weights.append(float(values['weight']))
        stored_norms.append(float(values['normalized_weight']))
        edge_count += 1
    weights = np.array(weights)
    normalized = (weights - weights.min()) / (weights.max() - weights.min())
    norm_error = float(np.max(np.abs(normalized - np.array(stored_norms))))
    eigs, residuals_base, residuals_aff = [], [], []
    diffs_base, diffs_aff, solve_residuals = [], [], []
    temporal_updates_after_centrality = 0
    post_smoothing_residuals = []
    for industry, names in groups.items():
        A = mats[industry]
        eigs.extend(np.linalg.eigvals(A))
        K = BETA * np.eye(len(names)) + ALPHA * A.T
        polarity = np.array([float(nodes[n]['sentiment_mean']) for n in names])
        risk = np.where(np.abs(polarity) < 0.05, 0.0,
                        np.where(polarity < 0, np.abs(polarity) ** 0.8,
                                 -0.5 * np.abs(polarity) ** 0.8))
        # The builder calls _temporal_sentiment_smoothing only after both centralities
        # have been computed. It changes only abs(sentiment_mean)<0.01, whose risk
        # at the centrality stage is zero under the documented 0.05 threshold.
        temporal_mask = np.array([nodes[n].get('fill_strategy', '') in
            ('temporal_interp', 'temporal_decay', 'temporal_forward') for n in names])
        temporal_updates_after_centrality += int(np.sum(temporal_mask))
        centrality_stage_risk = np.where(temporal_mask, 0.0, risk)
        rhs_base = ALPHA * THETA * A.T @ np.ones(len(names))
        rhs_aff = rhs_base + GAMMA * centrality_stage_risk
        base = np.array([float(nodes[n]['domirank']) for n in names])
        aff = np.array([float(nodes[n]['affective_domirank']) for n in names])
        residuals_base.extend(K @ base - rhs_base)
        residuals_aff.extend(K @ aff - rhs_aff)
        post_smoothing_residuals.extend(K @ aff - (rhs_base + GAMMA * risk))
        exact_base = np.linalg.solve(K, rhs_base)
        exact_aff = np.linalg.solve(K, rhs_aff)
        diffs_base.extend(base - exact_base)
        diffs_aff.extend(aff - exact_aff)
        solve_residuals.extend(K @ exact_aff - rhs_aff)
        global_base.extend(base)
        global_aff.extend(aff)
    eigs = np.array(eigs)
    assert eigs.size == len(nodes)
    jac_eigs = -BETA - ALPHA * eigs
    item = {
        'year': year, 'source': str(path), 'sha256': hashlib.sha256(raw).hexdigest(),
        'nodes': len(nodes), 'edges': edge_count,
        'independent_industry_blocks': len(groups),
        'max_block_size': max(len(x) for x in groups.values()),
        'adjacency_min_real_eigenvalue': float(np.min(eigs.real)),
        'adjacency_max_real_eigenvalue': float(np.max(eigs.real)),
        'jacobian_max_real_eigenvalue': float(np.max(jac_eigs.real)),
        'asymptotically_stable': bool(np.max(jac_eigs.real) < 0),
        'euler_dt_0_01_spectral_radius': float(np.max(np.abs(1 + DT * jac_eigs))),
        'stored_base_equilibrium_residual_linf': float(np.max(np.abs(residuals_base))),
        'stored_affective_equilibrium_residual_linf': float(np.max(np.abs(residuals_aff))),
        'stored_base_vs_direct_equilibrium_linf': float(np.max(np.abs(diffs_base))),
        'stored_affective_vs_direct_equilibrium_linf': float(np.max(np.abs(diffs_aff))),
        'direct_linear_solve_residual_linf': float(np.max(np.abs(solve_residuals))),
        'stored_edge_minmax_normalization_error_linf': norm_error,
        'primary_pickle_topology_and_score_fields_match_graphml': True,
        'post_centrality_temporal_updates': temporal_updates_after_centrality,
        'diagnostic_residual_if_later_smoothed_sentiment_is_used_instead': float(np.max(np.abs(post_smoothing_residuals))),
    }
    records.append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
result = {
    'scope': 'Read-only spectral and saved-score verification; no prediction model retrained, no source graph overwritten.',
    'parameters': {'alpha': ALPHA, 'beta': BETA, 'gamma': GAMMA, 'theta': THETA},
    'matrix_definition': 'A[i,j]=1 for a stored directed edge i->j; incoming dynamics use A.T.',
    'ode': 'dGamma/dt = alpha*theta*A.T*1 - (beta*I+alpha*A.T)*Gamma + gamma*s',
    'condition': 'max real eig(-(beta*I+alpha*A.T)) < 0',
    'snapshot_inputs_fixed': 'The sentiment forcing s is fixed within each annual centrality calculation, so it does not enter the Jacobian.',
    'forcing_stage': 'Annual centrality is computed before the later temporal sentiment fill. For nodes marked temporal_interp, temporal_decay or temporal_forward, pre-fill abs(sentiment)<0.01 implies zero centrality-stage risk under the 0.05 threshold. This reconstruction uses documented source control flow, not forcing values inferred from the saved scores.',
    'primary_pickle': {'path': str(primary_path), 'sha256': hashlib.sha256(primary_path.read_bytes()).hexdigest()},
    'years': records,
    'stored_score_ranges_for_feature_normalization': {
        'affective': [float(min(global_aff)), float(max(global_aff))],
        'traditional': [float(min(global_base)), float(max(global_base))],
    },
    'source_code_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE/'algorithms/domirank.py', SOURCE/'algorithms/affective_domirank.py', SOURCE/'data_processing/03_build_yearly_networks.py', SOURCE/'experiments/exp4_stgnn_temporal.py']},
}
(WORK / 'stability_check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print('ALL_YEARS_STABLE', all(r['asymptotically_stable'] for r in records))
