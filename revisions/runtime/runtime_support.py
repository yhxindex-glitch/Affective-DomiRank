"""Load existing experiment definitions without running the source main()."""
import ast
import contextlib
import io
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm


def load_definitions(path):
    namespace = dict(globals())
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    definitions = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def build_dataset(log_path):
    namespace = load_definitions(SOURCE / 'experiments' / 'exp4_stgnn_temporal.py')
    with (SOURCE / 'data' / 'processed' / 'networks' / 'all_networks_2017_2021.pkl').open('rb') as handle:
        networks = pickle.load(handle)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        dataset = namespace['NodeLevelSTGNNDataset'](
            networks=networks, window_size=3, use_interpolation=True,
            risk_labels_path=str(SOURCE / 'data' / 'processed' / 'networks' / 'risk_labels.pkl'),
        )
    Path(log_path).write_text(log.getvalue(), encoding='utf-8')
    return dataset


def attach_full_features(dataset):
    """Reuse verified normalized last-step vectors; keep all 16 features."""
    windows = {}
    for sample in dataset.samples:
        windows.setdefault(sample['window_idx'], {})[sample['node']] = sample
    full = {w: torch.stack([items[node]['node_features_seq'][-1] for node in dataset.all_nodes])
            for w, items in windows.items()}
    for sample in dataset.samples:
        sample['full_graph_features'] = full[sample['window_idx']]
    return full


class SampleDataset:
    def __init__(self, samples, dataset):
        self.samples = samples
        self.label_mean = dataset.label_mean
        self.label_std = dataset.label_std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

    def denormalize_labels(self, labels):
        return labels * self.label_std + self.label_mean
