"""CIFAR-10-LT and CIFAR-100-LT: long-tailed training set, train/val split, balanced test set.

Class c keeps n_c = int(n_max * IR ** (-c / (K - 1))) training images, drawn with the
global NumPy seed 42, so the long-tailed set is the same for every run seed. Each class
is then split 80/20 into train and validation. The returned class counts are the
counts before the split; the losses and the gradient logger use them.
"""

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

SPECS = {
    'cifar10': {'num_classes': 10, 'n_max': 5000,
                'mean': [0.4914, 0.4822, 0.4465], 'std': [0.2023, 0.1994, 0.2010]},
    'cifar100': {'num_classes': 100, 'n_max': 500,
                 'mean': [0.5071, 0.4867, 0.4408], 'std': [0.2675, 0.2565, 0.2761]},
}


def lt_class_counts(dataset, ir):
    """Per-class training counts n_c = int(n_max * IR ** (-c / (K - 1)))."""
    K, n_max = SPECS[dataset]['num_classes'], SPECS[dataset]['n_max']
    rho = ir ** (-1 / (K - 1))
    return [int(n_max * (rho ** i)) for i in range(K)]


def make_lt_indices(targets, dataset, ir, seed=42):
    """Indices of the long-tailed training set within the full training set.

    Returns:
        (indices, class_counts): shuffled list of indices and the per-class counts.
    """
    K = SPECS[dataset]['num_classes']
    targets = np.array(targets)
    class_indices = [np.where(targets == c)[0] for c in range(K)]
    class_counts = lt_class_counts(dataset, ir)

    np.random.seed(seed)
    lt_indices = []
    for c, n_samples in enumerate(class_counts):
        n_samples = max(1, n_samples)   # at least one image per class
        selected = np.random.choice(class_indices[c], size=n_samples, replace=False)
        lt_indices.extend(selected)

    lt_indices = np.array(lt_indices)
    np.random.shuffle(lt_indices)
    return lt_indices.tolist(), class_counts


def split_train_val(lt_indices, targets, num_classes, seed=42):
    """Per-class 80/20 split; every class gets max(1, n_c // 5) validation images.

    The NumPy seed is reset before shuffling each class.
    """
    lt_indices = np.array(lt_indices)
    lt_targets = np.array(targets)[lt_indices]
    train_indices, val_indices = [], []
    for c in range(num_classes):
        c_idx = np.where(lt_targets == c)[0]
        # From the CIFAR-100 cell. Never triggers (every class keeps >= 1 image), and for
        # CIFAR-10 an empty class would add nothing anyway, so one guard serves both.
        if len(c_idx) == 0:
            continue
        np.random.seed(seed)
        np.random.shuffle(c_idx)
        n_c_val = max(1, len(c_idx) // 5)
        val_indices.extend(lt_indices[c_idx[:n_c_val]])
        train_indices.extend(lt_indices[c_idx[n_c_val:]])
    return train_indices, val_indices


def load_cifar_lt(dataset, ir, data_dir, batch_size=128, num_workers=0, seed=42):
    """Train, validation and test loaders and the long-tailed class counts.

    Train images are augmented (random crop with 4-pixel padding, horizontal flip);
    validation and test images are only normalised. Train and validation use separate
    dataset objects so that their transforms stay separate.

    Returns:
        (train_loader, val_loader, test_loader, class_counts)
    """
    spec = SPECS[dataset]
    cifar = datasets.CIFAR10 if dataset == 'cifar10' else datasets.CIFAR100

    full_dataset = cifar(root=data_dir, train=True, download=True, transform=None)
    lt_indices, class_counts = make_lt_indices(full_dataset.targets, dataset, ir, seed=seed)
    train_indices, val_indices = split_train_val(lt_indices, full_dataset.targets,
                                                 spec['num_classes'], seed=seed)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=spec['mean'], std=spec['std']),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=spec['mean'], std=spec['std']),
    ])

    train_base = cifar(root=data_dir, train=True, download=True, transform=train_tf)
    val_base = cifar(root=data_dir, train=True, download=True, transform=test_tf)
    train_ds = Subset(train_base, train_indices)
    val_ds = Subset(val_base, val_indices)
    test_ds = cifar(root=data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    return train_loader, val_loader, test_loader, class_counts
