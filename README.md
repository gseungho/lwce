# LWCE: log-weighted cross-entropy for class-imbalanced classification

Code and results for the paper *Log-Weighted Cross-Entropy Framework for Imbalanced
Classification* (submitted to Applied Intelligence).

Inverse-frequency weighting (WCE) gives a rare class a weight proportional to `N / n_c`,
so the ratio between the rarest and the most frequent class grows with the imbalance.
LWCE replaces the weight by `1 / log(1 + n_c)`, which compresses that ratio
logarithmically and adds no hyperparameter. The repository contains the four proposed
losses (LWCE, PLWCE, ES-LWCE, BR-LWCE), the baselines, the training code for the four
domains of the paper, the saved results of every run, and the scripts that rebuild the
paper's tables and figures from those results.

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/gseungho/lwce.git
cd lwce
pip install -e .                 # losses, training code, tables
pip install -e ".[figures]"      # + matplotlib for the figures
pip install -e ".[text]"         # + Hugging Face datasets (GoEmotions)
pip install -e ".[network]"      # + kagglehub and pyarrow (network-intrusion data)
pip install -e ".[tabular]"      # + xlrd (reads one .xls file)
```

`requirements.txt` lists the versions the code was tested with. pandas must be below 3.

## Using the losses

```python
from lwce import get_clf_loss

class_counts = [5000, 2000, 300, 40]            # training samples per class
criterion = get_clf_loss('lwce', class_counts)  # an nn.Module
loss = criterion(logits, targets)               # logits: (B, C), targets: (B,)
```

| Name | Weight `w_c` (normalised to mean 1) | Parameter |
|---|---|---|
| `lwce` | `1 / log(1 + n_c)` | none |
| `plwce` | `1 / log(1 + n_c) ** alpha` | `alpha` |
| `eslwce` | `1 / (log(1 + n_c) + eps)` | `eps` |
| `brlwce` | `1 / (1 + log(1 + C p_c))`, `p_c = n_c / N` | none |
| `ce`, `wce`, `pwce`, `sqce`, `cb` | 1, `N / n_c`, `(N / n_c) ** alpha`, `sqrt(N / n_c)`, class-balanced | `alpha` (pwce), `beta` (cb) |
| `focal`, `logitadj`, `ldam` | focal loss, logit adjustment, LDAM | `gamma`, `tau`, none |

`combined` (`(log(1 + n_c) + eps) ** -alpha`) is used only in the ablation.

## Reproducing the paper's numbers from the saved results

No GPU and no data download are needed.

```bash
python scripts/paper_tables.py                  # main table, ranks, statistics, ablation, ...
python scripts/figures/make_weight_compression.py
python scripts/figures/make_gradient_mechanism.py
python scripts/figures/make_severity.py
python scripts/figures/make_sensitivity.py
```

`paper_tables.py` recomputes every cross-domain number of the paper (per-domain macro-F1,
mean ranks, Friedman and Holm-corrected Wilcoxon tests, TOST, ablation, selection
analysis) from `results/`, compares each with the printed value and reports the
mismatches. The figure scripts write PDFs to `figures/`.

## Re-running the experiments

Every script resumes: a run whose result is already in the output folder is skipped, and
the aggregated files always cover every run in the folder, including those of earlier
calls with other `--losses`, `--seeds` or `--datasets`. **The default output folders under
`results/` already hold the paper's runs**, so point `--out-dir` to a new folder to train
from scratch. Each output folder records its training settings (`--epochs`,
`--proxy-epochs`) in `run_config.json`; a call with different settings on the same folder
stops instead of mixing runs made with different budgets.

```bash
# Image: CIFAR-10-LT / CIFAR-100-LT, ResNet-32 (CIFAR is downloaded by torchvision)
python scripts/run_image.py --dataset cifar100 --data-dir data/cifar --out-dir runs/cifar100_lt

# Tabular: 11 datasets, MLP (raw files under data/tabular, see below)
python scripts/run_tabular.py --data-dir data/tabular --out-dir runs/tabular
python scripts/analyze_mlp.py --domain tabular --results-dir runs/tabular

# Network intrusion: 8 datasets, MLP (downloaded with kagglehub, which may ask you to log in)
python scripts/run_network.py --out-dir runs/network
python scripts/analyze_mlp.py --domain network --results-dir runs/network

# Text: GoEmotions, Transformer encoder (downloaded from the Hugging Face Hub)
python scripts/run_text.py --out-dir runs/text
```

Each script first selects the parameter of every tuned loss on a proxy subset of the
training split (30 grid points, 60 for `combined`), then trains each of the 13 losses with
seeds 42-46 and evaluates on the test split. `--losses`, `--seeds` and `--epochs` run a
subset; `--help` lists all options. The defaults are the paper's settings. The full image
sweep is by far the most expensive part.

### Data

- **CIFAR-10/100**: downloaded automatically. The long-tailed subsets and the 80/20
  train/validation split are built in `lwce/data/cifar_lt.py`.
- **Tabular**: place the raw files directly in `--data-dir`. File names are listed in
  `lwce/data/tabular.py`. Sources: UCI Machine Learning Repository (APS Failure at Scania
  Trucks, Bank Marketing, Statlog German Credit, SECOM, Default of Credit Card Clients,
  Glass Identification, Steel Plates Faults, Yeast, Page Blocks Classification) and Kaggle
  (`mlg-ulb/creditcardfraud`, `blastchar/telco-customer-churn`).
- **Network intrusion**: Kaggle copies of NSL-KDD, UNSW-NB15, CICIDS2017, CICIDS2018,
  CIC-DDoS2019, Bot-IoT, TON_IoT and RT-IoT2022. The handles and pinned versions are in
  `lwce/data/network.py`. CICIDS2017, CICIDS2018, CIC-DDoS2019 and Bot-IoT are reduced to
  a stratified subset of about 300,000 flows with the benign class capped at 100,000.
- **GoEmotions**: `google-research-datasets/go_emotions` (simplified), single-label
  examples only.

## Reproducibility notes

- The code was checked against the notebooks that produced the paper: on small CPU inputs
  every training loop, data pipeline and result file is bit-identical to the original.
- Re-runs on a GPU are not bit-identical to the saved results, because the training does
  not force deterministic cuDNN kernels. Expect differences of the size of the seed-to-seed
  variation.
- Some details are kept exactly as they were run, even where another choice would be
  cleaner: the image losses use the class counts before the 80/20 split, the learning
  rate milestones are fixed at epochs 160 and 180 (also in the 20-epoch proxy, where they
  never fire), LDAM uses a 5-epoch learning-rate warm-up on the image backbone, a few
  tabular encodings are fit before the train/test split, and Bot-IoT takes the first
  8,000 rows of each file.

## Repository layout

| Path | Content |
|---|---|
| `lwce/losses.py` | all losses and `get_clf_loss` |
| `lwce/metrics.py` | macro-F1 helpers, G-mean, worst-class recall, gradient logger |
| `lwce/search.py` | grids and grid search |
| `lwce/models/`, `lwce/data/`, `lwce/train/` | backbones, data loaders, training loops |
| `scripts/` | experiment runners, `paper_tables.py`, `figures/` |
| `results/cifar10_lt`, `results/cifar100_lt`, `results/tabular`, `results/network`, `results/text` | saved results of every run in the paper |
| `tests/` | unit tests |

## Citation

The citation will be added when the paper is published.

## License

MIT License. See [LICENSE](LICENSE).
