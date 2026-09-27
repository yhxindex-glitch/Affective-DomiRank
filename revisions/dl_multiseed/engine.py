"""FP32 execution optimizations; model equations and effective batch size retained."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
OUT = ROOT / 'results' / 'dl_multiseed'
sys.path.insert(0, str(ROOT / 'revisions' / 'runtime'))
from runtime_support import SOURCE, Data, DataLoader, F, nn, np, optim, torch, load_definitions

GRAPH_MODELS = {'gcn', 'gat', 'gcn_lstm', 'gat_lstm'}
ADVANCED_MODELS = {'dcrnn', 'mtgnn'}
ALL_MODELS = ['mlp', 'lstm', 'gcn', 'gcn_lstm', 'gat_lstm', 'dcrnn', 'mtgnn', 'gat']
SEEDS = [42, 52, 62, 72, 82]


def source_namespace(candidate=False):
    path = SOURCE / 'experiments/run_baselines.py'
    return load_definitions(path)


def advanced_module():
    spec = importlib.util.spec_from_file_location('dl_advanced_baselines', SOURCE / 'models/advanced_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def default_config(name):
    result = {'input_dim': 16, 'hidden_dim': 64, 'dropout': 0.5, 'learning_rate': 0.001,
              'weight_decay': 0.0005, 'max_epochs': 200, 'patience': 15, 'batch_size': 32}
    if name in {'gat', 'gat_lstm'}:
        result['heads'] = 2
    if name in {'lstm', 'gcn_lstm', 'gat_lstm'}:
        result['lstm_layers'] = 2
    if name in ADVANCED_MODELS:
        result.update(hidden_dim=32, dropout=0.3)
    if name == 'dcrnn':
        result.update(K=1, num_layers=1)
    if name == 'mtgnn':
        result['num_layers'] = 2
    return result


def make_model(name, config, num_nodes, namespace):
    common = {'input_dim': 16, 'hidden_dim': config['hidden_dim'], 'dropout': config['dropout']}
    if name == 'mlp':
        return namespace['MLP'](**common)
    if name == 'lstm':
        return namespace['LSTMModel'](**common, num_layers=config['lstm_layers'])
    if name == 'gcn':
        return namespace['GCNModel'](**common, num_nodes=num_nodes)
    if name == 'gat':
        return namespace['GATModel'](**common, num_nodes=num_nodes, heads=config['heads'])
    if name in {'gcn_lstm', 'gat_lstm'}:
        kwargs = {**common, 'num_nodes': num_nodes, 'lstm_layers': config['lstm_layers']}
        if name == 'gat_lstm':
            kwargs['heads'] = config['heads']
        return namespace['GCN_LSTM_Model' if name == 'gcn_lstm' else 'GAT_LSTM_Model'](**kwargs)
    kwargs = {**common, 'num_nodes': num_nodes, 'num_layers': config['num_layers']}
    if name == 'dcrnn':
        kwargs['K'] = config['K']
    return advanced_module().AdvancedBaselineWrapper(model_type=name, **kwargs)


class CachedData:
    def __init__(self, device='cuda'):
        raw = torch.load(OUT / 'dataset_cache.pt', map_location='cpu', mmap=True, weights_only=True)
        self.raw = raw
        self.device = device
        self.x = raw['node_features'].to(device)
        self.y = raw['labels'].to(device)
        self.target = raw['node_indices'].to(device)
        self.windows = raw['window_indices']
        self.full = raw['full_features'].to(device)
        self.num_nodes = len(raw['nodes'])
        self.graphs = [Data(edge_index=g['edge_index'].to(device), edge_weight=g['edge_weight'].to(device),
                            edge_attr=g['edge_weight'].to(device), nodes=raw['nodes']) for g in raw['graphs']]
        self.last_graphs = [self.graphs[indices[-1]] for indices in raw['window_graph_indices']]
        self.first_graphs = [self.graphs[indices[0]] for indices in raw['window_graph_indices']]

    def split(self, seed=42):
        indices = list(range(len(self.y)))
        random.Random(seed).shuffle(indices)
        cut = int(0.8 * len(indices))
        return indices[:cut], indices[cut:]

    def batch(self, ids):
        ids = torch.as_tensor(ids, dtype=torch.long, device='cpu')
        windows = self.windows[ids].tolist()
        gpu_ids = ids.to(self.device)
        return {'ids': ids, 'windows': windows, 'x': self.x[gpu_ids], 'y': self.y[gpu_ids],
                'target': self.target[gpu_ids], 'full': self.full[windows],
                'last_graphs': [self.last_graphs[w] for w in windows],
                'first_graph': self.first_graphs[windows[0]]}

    def reference_batch(self, ids, namespace):
        samples = []
        for idx in ids:
            idx = int(idx)
            w = int(self.windows[idx])
            samples.append({'node': self.raw['nodes'][int(self.raw['node_indices'][idx])],
                            'graphs': [self.graphs[j] for j in self.raw['window_graph_indices'][w]],
                            'input_years': self.raw['window_years'][w],
                            'node_features_seq': self.x[idx], 'label': self.y[idx].reshape(1),
                            'full_graph_features': self.full[w]})
        return namespace['graph_collate_fn'](samples)


def block_edges(graphs, num_nodes):
    return (torch.cat([g.edge_index + i * num_nodes for i, g in enumerate(graphs)], dim=1),
            torch.cat([g.edge_weight for g in graphs]))


def fast_graph(model, name, x_seq, full_features, graphs, targets):
    batch_size, num_nodes, feature_dim = full_features.shape
    temporal = None
    if name.endswith('_lstm'):
        _, (hidden, _) = model.lstm(x_seq)
        temporal = hidden[-1]
    edges, weights = block_edges(graphs, num_nodes)
    x = full_features.reshape(batch_size * num_nodes, feature_dim)
    is_attention = name in {'gat', 'gat_lstm'}
    if is_attention:
        x = F.elu(model.conv1(x, edges, edge_attr=weights.reshape(-1, 1)))
    else:
        x = F.relu(model.conv1(x, edges, weights))
    x = F.dropout(x, p=model.dropout, training=model.training)
    if is_attention:
        x = F.elu(model.conv2(x, edges, edge_attr=weights.reshape(-1, 1)))
    else:
        x = F.relu(model.conv2(x, edges, weights))
    x = F.dropout(x, p=model.dropout, training=model.training)
    selected = x[targets + torch.arange(batch_size, device=x.device) * num_nodes]
    if temporal is not None:
        selected = torch.cat([selected, temporal], dim=-1)
    return model.fc(selected).reshape(-1)


def fast_dcrnn(model, x, edges, weights):
    base = model.model
    batch_size, time_steps, num_nodes, _ = x.shape
    block_index = torch.cat([edges + i * num_nodes for i in range(batch_size)], dim=1)
    block_weight = weights.repeat(batch_size)
    hidden = [x.new_zeros(batch_size * num_nodes, base.hidden_dim) for _ in range(base.num_layers)]
    for t in range(time_steps):
        for layer in range(base.num_layers):
            values = x[:, t].reshape(batch_size * num_nodes, -1) if layer == 0 else hidden[layer - 1]
            hidden[layer] = base.encoder_cells[layer](values, hidden[layer], block_index, block_weight)
            if layer < base.num_layers - 1:
                hidden[layer] = F.dropout(hidden[layer], p=base.dropout, training=base.training)
    return base.output_proj(hidden[-1]).reshape(batch_size, num_nodes)


def forward(model, name, batch, optimized=True, x_override=None, edge_override=None):
    x = batch['x'] if x_override is None else x_override
    if name in GRAPH_MODELS:
        if optimized:
            return fast_graph(model, name, x, batch['full'], batch['last_graphs'], batch['target'])
        return model(x, [[g] for g in batch['last_graphs']], batch['target'], batch['full'])
    if name in ADVANCED_MODELS:
        b, t, f = x.shape
        n = batch['full'].shape[1]
        features = x.new_zeros(b, t, n, f)
        features[torch.arange(b, device=x.device), :, batch['target'], :] = x
        graph = batch['first_graph']
        edges, weights = (graph.edge_index, graph.edge_weight) if edge_override is None else edge_override
        all_predictions = fast_dcrnn(model, features, edges, weights) if name == 'dcrnn' and optimized else model(features, edges, weights)
        return all_predictions[torch.arange(b, device=x.device), batch['target']]
    return model(x, [])


def metrics(predictions, labels, data):
    pred = predictions.detach().float().cpu().numpy() * data.raw['label_std'] + data.raw['label_mean']
    truth = labels.detach().float().cpu().numpy() * data.raw['label_std'] + data.raw['label_mean']
    residual = np.square(pred - truth)
    result = {'MAE': float(np.abs(pred - truth).mean()), 'RMSE': float(np.sqrt(residual.mean())),
              'R2': float(1 - residual.sum() / np.square(truth - truth.mean()).sum())}
    return result, pred, truth


def evaluate(model, name, loader, data):
    model.eval()
    predictions, labels, sample_ids = [], [], []
    spatial_cache = {}
    with torch.no_grad():
        for ids in loader:
            batch = data.batch(ids)
            if name in GRAPH_MODELS:
                for w in set(batch['windows']):
                    if w not in spatial_cache:
                        graph = data.last_graphs[w]
                        x = data.full[w]
                        if name in {'gat', 'gat_lstm'}:
                            x = F.elu(model.conv1(x, graph.edge_index, edge_attr=graph.edge_weight.reshape(-1, 1)))
                            x = F.elu(model.conv2(x, graph.edge_index, edge_attr=graph.edge_weight.reshape(-1, 1)))
                        else:
                            x = F.relu(model.conv1(x, graph.edge_index, graph.edge_weight))
                            x = F.relu(model.conv2(x, graph.edge_index, graph.edge_weight))
                        spatial_cache[w] = x
                embeddings = torch.stack([spatial_cache[w][batch['target'][i]] for i, w in enumerate(batch['windows'])])
                if name.endswith('_lstm'):
                    _, (hidden, _) = model.lstm(batch['x'])
                    embeddings = torch.cat([embeddings, hidden[-1]], dim=-1)
                pred = model.fc(embeddings).reshape(-1)
            else:
                pred = forward(model, name, batch)
            predictions.append(pred)
            labels.append(batch['y'])
            sample_ids.extend(ids.tolist())
    result, pred, truth = metrics(torch.cat(predictions), torch.cat(labels), data)
    return result, pred, truth, sample_ids


def train_epoch(model, name, loader, data, optimizer, microbatch=8):
    model.train()
    total_loss, samples = 0.0, 0
    for ids in loader:
        batch = data.batch(ids)
        x = batch['x'] + torch.randn_like(batch['x']) * 0.05
        mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > 0.2
        x = x * mask
        graph = batch['first_graph']
        edge_mask = torch.rand(graph.edge_index.shape[1], device=x.device) > 0.1
        edges, weights = graph.edge_index[:, edge_mask], graph.edge_weight[edge_mask]
        optimizer.zero_grad(set_to_none=True)
        effective_size = len(ids)
        chunk_size = microbatch if name in GRAPH_MODELS or name == 'dcrnn' else effective_size
        batch_loss = 0.0
        for start in range(0, effective_size, chunk_size):
            stop = min(start + chunk_size, effective_size)
            part = {k: (v[start:stop] if k in {'ids', 'windows', 'x', 'y', 'target', 'full', 'last_graphs'} else v) for k, v in batch.items()}
            pred = forward(model, name, part, x_override=x[start:stop], edge_override=(edges, weights))
            loss = F.mse_loss(pred, part['y'], reduction='sum') / effective_size
            loss.backward()
            batch_loss += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += batch_loss * effective_size
        samples += effective_size
    return total_loss / samples
