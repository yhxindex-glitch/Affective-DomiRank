"""Training utilities, with neural dependencies imported when requested."""
from importlib import import_module

__all__ = [
    'Trainer',
    'Evaluator',
    'SupplyChainDataset',
]


def __getattr__(name):
    modules = {'Trainer': '.trainer', 'Evaluator': '.evaluator', 'SupplyChainDataset': '.dataset'}
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

