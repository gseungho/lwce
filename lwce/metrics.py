"""Evaluation metrics and the gradient logger shared by all four domains."""

import numpy as np
import torch


def group_masks(class_counts, num_classes=None):
    """Many/few class masks from the training counts (tertiles).

    Many = the n // 3 most frequent classes, few = the n - 2 * (n // 3) rarest. For
    fewer than three classes the most frequent class is "many" and the rest "few".

    Returns:
        (is_many, is_few), boolean tensors of shape (num_classes,).
    """
    counts = np.asarray(class_counts)
    n = len(counts)
    if num_classes is None:
        num_classes = n
    order = np.argsort(counts)[::-1].copy()   # most to least frequent
    if n < 3:
        many_idx, few_idx = order[:1], order[1:]
    else:
        many_idx, few_idx = order[:n // 3], order[2 * (n // 3):]
    is_many = torch.zeros(num_classes, dtype=torch.bool)
    is_few = torch.zeros(num_classes, dtype=torch.bool)
    is_many[torch.as_tensor(many_idx, dtype=torch.long)] = True
    is_few[torch.as_tensor(few_idx, dtype=torch.long)] = True
    return is_many, is_few


def total_grad_norm(model):
    """Total L2 norm of the parameter gradients (does not modify them)."""
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += float(p.grad.detach().norm().item()) ** 2
    return sq ** 0.5


class GradLogger:
    """Per-epoch gradient statistics.

    Usage inside any training loop::

        glog = GradLogger(class_counts, num_classes, device)
        for epoch in range(epochs):
            glog.reset()
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = model(xb)
                logits.retain_grad()            # required
                loss = criterion(logits, yb)
                loss.backward()
                glog.update(logits, yb, model)  # before optimizer.step()
                optimizer.step()
            stats = glog.epoch_end()

    Recorded per epoch:
        grad_norm   mean over batches of the total parameter-gradient norm
        grad_many   mean per-sample norm of the logit gradient over many-class samples
        grad_few    the same over few-class samples
        grad_ratio  grad_few / grad_many (the minority-to-majority gradient ratio R_G)
    """

    def __init__(self, class_counts, num_classes, device):
        is_many, is_few = group_masks(class_counts, num_classes)
        self.is_many = is_many.to(device)
        self.is_few = is_few.to(device)
        self.reset()

    def reset(self):
        self._gn_sum = 0.0
        self._n_batch = 0
        self._many_sum = 0.0
        self._many_n = 0
        self._few_sum = 0.0
        self._few_n = 0

    def update(self, logits, yb, model=None, grad_norm=None):
        """Call after loss.backward() and before optimizer.step().

        If `grad_norm` is given (e.g. the return value of clip_grad_norm_), it is used
        instead of recomputing the parameter-gradient norm.
        """
        if grad_norm is None:
            if model is None:
                raise ValueError('Either model or grad_norm is required.')
            grad_norm = total_grad_norm(model)
        self._gn_sum += float(grad_norm)
        self._n_batch += 1

        if logits.grad is None:
            raise RuntimeError('logits.grad is None: call logits.retain_grad() before '
                               'loss.backward().')
        with torch.no_grad():
            gs = logits.grad.norm(dim=1)
            m_mask, f_mask = self.is_many[yb], self.is_few[yb]
            if m_mask.any():
                self._many_sum += float(gs[m_mask].sum())
                self._many_n += int(m_mask.sum())
            if f_mask.any():
                self._few_sum += float(gs[f_mask].sum())
                self._few_n += int(f_mask.sum())

    def epoch_end(self):
        many = self._many_sum / max(1, self._many_n)
        few = self._few_sum / max(1, self._few_n)
        return {
            'grad_norm': self._gn_sum / max(1, self._n_batch),
            'grad_many': many,
            'grad_few': few,
            'grad_ratio': (few / many) if many > 0 else float('nan'),
        }


def extended_metrics(y_true, y_pred, num_classes):
    """Per-class recall and the metrics derived from it.

    Returns:
        dict with Per_Class_Acc (recall per class), G_Mean (geometric mean of the
        recalls), Worst_Acc (smallest recall) and Balanced_Acc (mean recall).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rec = np.zeros(num_classes, dtype=float)
    for c in range(num_classes):
        m = (y_true == c)
        rec[c] = float((y_pred[m] == c).mean()) if m.sum() > 0 else 0.0
    return {
        'Per_Class_Acc': rec.tolist(),
        'G_Mean': float(np.exp(np.mean(np.log(rec + 1e-12)))),
        'Worst_Acc': float(rec.min()),
        'Balanced_Acc': float(rec.mean()),
    }
