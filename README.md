# Affective DomiRank

Python code for sentiment-aware centrality, country–industry trade-network processing and supply-chain vulnerability prediction.

## Contents

| Directory | Contents |
|---|---|
| `algorithms/` | DomiRank, Affective DomiRank and reference centrality methods |
| `models/` | STGNN and neural-network baseline architectures |
| `configs/` | Country and industry definitions |
| `data_processing/` | Trade processing, news matching and annual graph construction |
| `training/` | Dataset, training, evaluation and sample-identity utilities |
| `experiments/` | Main model, baseline models, ablations, sensitivity analyses and case studies |
| `revisions/dl_multiseed/` | Cached GPU execution and parallel multiseed jobs |
| `revisions/network_analysis/` | Spectral stability, score decomposition and node-removal analyses |
| `visualization/` | Network and result visualization utilities |
| `tests/` | Lightweight checks for experiment entry points |

## Installation

Use Python 3.10 or later. Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
```

Activate the virtual environment before installation. For GPU training, install a PyTorch build compatible with your CUDA environment. The parallel GPU runner requires CUDA; its concurrency and microbatch size are configurable.

## Local data

Data, model weights, caches and generated results are stored locally and are not bundled in this repository. The experiment entry points use these prepared files:

```text
data/processed/networks/all_networks_2017_2021.pkl
data/processed/networks/risk_labels.pkl
```

To construct them from the original trade and news inputs, arrange the input files as follows:

```text
data/raw/archive/country_codes.csv
data/raw/archive/product_codes.csv
data/raw/archive/Year_2017/Year_2017.csv
data/raw/archive/Year_2018/Year_2018.csv
data/raw/archive/Year_2019/Year_2019.csv
data/raw/archive/Year_2020/Year_2020.csv
data/raw/archive/Year_2021/Year_2021.csv
data/raw/business_news/business_news_sentiment.csv
```

The trade source is BACI; the news file contains the provider's headline and sentiment records. Column mappings are defined in the corresponding preprocessing scripts.

```bash
python run.py trade
python run.py news
python run.py networks
python run.py labels
```

## Running experiments

Run commands from the repository root. List available tasks or check local inputs without starting training:

```bash
python run.py --list
python run.py main --check
```

Main model and comparisons:

```bash
python run.py main
python run.py main-seeds -- --seeds 42 52 62 72 82
python run.py baselines
python run.py ml
python run.py ml-seeds -- --seeds 42 52 62 72 82
python run.py traditional
python run.py no-domirank
python run.py no-sentiment
```

Structural-target input comparison and label-weight sensitivity:

```bash
python run.py target-study -- --weights structural --inputs all structural --seeds 42 43 44
python run.py target-study -- --weights original equal structure80 structural --inputs all --seeds 42
```

The four weight configurations are ordered as supplier loss, degree decline, negative sentiment and sentiment deterioration:

| Configuration | Weights |
|---|---|
| `original` | 0.40, 0.25, 0.20, 0.15 |
| `equal` | 0.25, 0.25, 0.25, 0.25 |
| `structure80` | 0.50, 0.30, 0.12, 0.08 |
| `structural` | 8/13, 5/13, 0, 0 |

Additional analyses:

```bash
python run.py sensitivity
python run.py mapping
python run.py case-energy
python run.py case-attributes
python run.py stability
python run.py decomposition
python run.py node-removal
python run.py node-backbone
```

Run the case studies before the corresponding decomposition and node-removal tasks. These analyses also read the annual GraphML files created by the network builder. Node-removal settings are stored in `revisions/network_analysis/`; outputs go to `results/node_removal/` and `results/network_analysis/`.

## Parallel neural baselines

The runner covers MLP, LSTM, GCN, GAT, GCN–LSTM, GAT–LSTM, DCRNN and MTGNN, using five seeds: 42, 52, 62, 72 and 82. It records each job's settings, sample identifiers, predictions and metrics.

```bash
python run.py dl-cache
python run.py dl-plan -- --workers 3 --microbatch 16
python run.py dl-seeds
```

Choose fewer workers or a smaller microbatch for GPUs with less available memory. GAT configuration selection runs before its five final fits. Completed jobs and their outputs remain in `results/dl_multiseed/`.

## Results and figures

Experiment scripts write the metrics produced by the current run. Target studies write to `results/target_study/`; other scripts specify their output paths in the entry point. Root-level `create_*.py` files contain figure routines and layout templates. Supply the corresponding experiment results before exporting figures.

## Checks

These checks run without training or loading research data:

```bash
python -m unittest discover -s tests -v
```
