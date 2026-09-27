"""Exercise the actual allocation blocks and dynamically generated variants."""
import ast
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.splits import assert_disjoint_indices, assert_disjoint_samples


@contextmanager
def project_directory():
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield
    finally:
        os.chdir(previous)


def allocation(code, function='main', seed=42):
    tree = ast.parse(code)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_samples' for t in n.targets))
    selected = []
    for n in fn.body[start:]:
        if isinstance(n, (ast.ClassDef, ast.For)):
            break
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {'kw', 'train_windows', 'stats'} for t in n.targets):
            break
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name) and n.value.func.id == 'print':
            break
        selected.append(n)
    samples = [{'node': f'N{i // 6}', 'input_years': (i % 6, i % 6 + 0.5, i % 6 + 1), 'target_year': i % 6 + 1.5} for i in range(7200)]
    state = {'full_dataset': SimpleNamespace(samples=samples), 'dataset': SimpleNamespace(samples=samples), 'random': random}
    random.seed(seed)
    exec(compile(ast.Module(body=selected, type_ignores=[]), '<allocation>', 'exec'), state)
    train, test = state['train_samples'], state['test_samples']
    assert_disjoint_samples(train, test)
    return len(train), len(test)


class PartitionTests(unittest.TestCase):
    def test_all_direct_entry_points(self):
        for script, function in [('exp4_stgnn_temporal.py', 'main'),
                                 ('run_baselines.py', 'main'),
                                 ('run_sensitivity_analysis_full.py', '_build_dataloaders'),
                                 ('run_sensitivity_analysis.py', 'train_and_evaluate')]:
            code = (ROOT / 'experiments' / script).read_text(encoding='utf-8-sig')
            if script == 'run_sensitivity_analysis.py':
                tree = ast.parse(code)
                function = next(n.name for n in tree.body if isinstance(n, ast.FunctionDef) and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'all_samples' for t in x.targets) for x in n.body))
            for seed in [42, 43, 44, 52, 62, 72, 82]:
                with self.subTest(script=script, seed=seed):
                    self.assertEqual(allocation(code, function, seed), (5760, 1440))

    def test_ablation_wrappers_use_current_source(self):
        for script in ['run_ablation_v2_domirank.py', 'run_ablation_v3_no_domirank.py', 'run_ablation_v4_no_sentiment.py']:
            captured = []
            namespace = {'__name__': 'capture_variant', 'exec': lambda code: captured.append(code)}
            with project_directory():
                exec(compile((ROOT / 'experiments' / script).read_text(encoding='utf-8-sig'), script, 'exec'), namespace)
            self.assertEqual(len(captured), 1)
            self.assertEqual(allocation(captured[0]), (5760, 1440))

    def test_multiseed_generated_main(self):
        for script, variants in [('run_significance_test.py', ['affective', 'traditional']), ('run_multiplicative_gate_test.py', [None])]:
            tree = ast.parse((ROOT / 'experiments' / script).read_text(encoding='utf-8-sig'))
            definition = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_load_runner')
            def capture(code, namespace):
                namespace['main'] = lambda: None
                namespace['captured'] = code
            namespace = {'Path': Path, 'Any': object, 'exec': capture}
            if script == 'run_multiplicative_gate_test.py':
                helper = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'HELPER_CODE' for t in n.targets))
                namespace['HELPER_CODE'] = ast.literal_eval(helper.value)
            exec(compile(ast.Module(body=[definition], type_ignores=[]), script, 'exec'), namespace)
            for variant in variants:
                for seed in [42, 52, 62, 72, 82]:
                    with project_directory():
                        _, generated = namespace['_load_runner'](variant, seed) if variant else namespace['_load_runner'](seed)
                    self.assertEqual(allocation(generated['captured'], seed=seed), (5760, 1440))

    def test_target_study_both_input_dimensions(self):
        path = ROOT / 'experiments/run_target_study.py'
        spec = importlib.util.spec_from_file_location('target_builder', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        source = (ROOT / 'experiments/exp4_stgnn_temporal.py').read_text(encoding='utf-8-sig')
        for variant in ['all', 'structural']:
            tree = module.build_runner(source, Path('labels.pkl'), Path('model.pt'), variant)
            self.assertEqual(allocation(ast.unparse(tree)), (5760, 1440))
            dataset = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NodeLevelSTGNNDataset')
            features = next(n for n in ast.walk(dataset) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'features' for t in n.targets) and isinstance(n.value, ast.List))
            self.assertEqual(len(features.value.elts), 16 if variant == 'all' else 9)

    def test_gpu_runner_partition(self):
        tree = ast.parse((ROOT / 'revisions/dl_multiseed/engine.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CachedData')
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'split')
        state = {'random': random}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), '<gpu split>', 'exec'), state)
        for seed in [42, 52, 62, 72, 82]:
            train, test = state['split'](SimpleNamespace(y=list(range(7200))), seed)
            assert_disjoint_indices(train, test, 7200)
            self.assertEqual((len(train), len(test)), (5760, 1440))

    def test_machine_learning_partitions(self):
        import numpy as np
        for filename, marker, train_name, test_name in [
            ('run_ml_baselines.py', 'indices', 'train_indices', 'test_indices'),
            ('run_ml_multiseed.py', 'order', 'train_ids', 'test_ids'),
        ]:
            tree = ast.parse((ROOT / 'experiments' / filename).read_text('utf-8-sig'))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == marker for t in n.targets))
            stop = next(i + 1 for i, n in enumerate(fn.body) if i >= start
                        and isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                        and isinstance(n.value.func, ast.Name) and n.value.func.id == 'assert_disjoint_indices')
            state = {'np': np, 'X': np.empty((7200, 1)), 'assert_disjoint_indices': assert_disjoint_indices}
            np.random.seed(42)
            exec(compile(ast.Module(body=fn.body[start:stop], type_ignores=[]), filename, 'exec'), state)
            self.assertEqual((len(state[train_name]), len(state[test_name])), (5760, 1440))
            assert_disjoint_indices(state[train_name], state[test_name], 7200)

    def test_semantic_duplicate_rejected(self):
        sample = {'node': 'N1', 'input_years': (2017, 2017.5, 2018), 'target_year': 2018.5}
        with self.assertRaises(ValueError):
            assert_disjoint_samples([sample], [dict(sample)])
        with self.assertRaises(ValueError):
            assert_disjoint_indices([0, 1], [1, 2], 3)
        with self.assertRaises(ValueError):
            assert_disjoint_indices([0], [2], 3)


if __name__ == '__main__':
    unittest.main()
