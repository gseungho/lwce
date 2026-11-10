"""GoEmotions ('simplified' config) as 28-class single-label classification.

Only examples with exactly one label are kept, in the official train / validation / test
splits. Texts are tokenised at word level (lowercase, regex [a-z0-9']+), the vocabulary is
built from the training split only, and every text is truncated or right-padded to MAX_LEN.
"""

import re
from collections import Counter

import torch
from torch.utils.data import DataLoader, TensorDataset

NUM_CLASSES = 28           # 27 emotions + neutral
MAX_LEN = 64
MIN_FREQ = 2               # minimum token frequency for the vocabulary
BATCH_SIZE = 64

TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts, min_freq=2):
    """Word-level vocabulary from the training texts (0 = <pad>, 1 = <unk>)."""
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    itos = ['<pad>', '<unk>']
    for tok, c in counter.most_common():
        if c >= min_freq:
            itos.append(tok)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, itos


def encode(text, stoi, max_len=MAX_LEN):
    """Token ids (unknown = 1), truncated to max_len and right-padded with 0."""
    ids = [stoi.get(tok, 1) for tok in tokenize(text)][:max_len]
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))
    return ids


def to_single(split):
    """Texts and labels of the examples that have exactly one label."""
    texts, labels = [], []
    for ex in split:
        if len(ex['labels']) == 1:
            texts.append(ex['text'])
            labels.append(ex['labels'][0])
    return texts, labels


def load_goemotions(batch_size=BATCH_SIZE, cache_dir=None):
    """Download GoEmotions from Hugging Face and build the three loaders.

    Returns:
        (train_loader, val_loader, test_loader, class_counts, stoi, itos, emotion_names);
        class_counts are the training-split label counts.
    """
    from datasets import load_dataset  # optional dependency (pip install .[text])

    ds = load_dataset('google-research-datasets/go_emotions', 'simplified', cache_dir=cache_dir)
    emotion_names = ds['train'].features['labels'].feature.names   # 28 class names

    tr_texts, tr_labels = to_single(ds['train'])
    va_texts, va_labels = to_single(ds['validation'])
    te_texts, te_labels = to_single(ds['test'])

    stoi, itos = build_vocab(tr_texts, MIN_FREQ)

    def make_loader(texts, labels, shuffle):
        X = torch.tensor([encode(t, stoi) for t in texts], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.long)
        return DataLoader(TensorDataset(X, y), batch_size=batch_size,
                          shuffle=shuffle, num_workers=0)

    train_loader = make_loader(tr_texts, tr_labels, True)
    val_loader   = make_loader(va_texts, va_labels, False)
    test_loader  = make_loader(te_texts, te_labels, False)

    class_counts = [0] * NUM_CLASSES
    for lab in tr_labels:
        class_counts[lab] += 1

    return train_loader, val_loader, test_loader, class_counts, stoi, itos, emotion_names
