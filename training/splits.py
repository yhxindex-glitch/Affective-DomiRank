"""Sample-identity checks shared by experiment entry points."""


def sample_key(sample):
    """Identify a node-window example independently of Python object identity."""
    return (str(sample['node']), tuple(sample['input_years']), sample['target_year'])


def assert_disjoint_samples(training, evaluation):
    train_keys = [sample_key(sample) for sample in training]
    test_keys = [sample_key(sample) for sample in evaluation]
    if len(set(train_keys)) != len(train_keys) or len(set(test_keys)) != len(test_keys):
        raise ValueError('Each sample must occur once within its partition.')
    if set(train_keys).intersection(test_keys):
        raise ValueError('Training and evaluation partitions share a node-window sample.')
    if not train_keys or not test_keys:
        raise ValueError('Both partitions must contain samples.')


def assert_disjoint_indices(training, evaluation, total):
    train_ids = [int(x) for x in training]
    test_ids = [int(x) for x in evaluation]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError('Partition indices must be unique.')
    if set(train_ids).intersection(test_ids):
        raise ValueError('Training and evaluation indices must be disjoint.')
    if set(train_ids).union(test_ids) != set(range(total)):
        raise ValueError('Partitions must cover the dataset exactly once.')
