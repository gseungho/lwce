"""Classification losses compared in the paper.

All losses are built through :func:`get_clf_loss`. Re-weighting losses use a
weighted-mean cross-entropy with per-class weights normalised to mean 1 over the
classes; because the objective is a weighted mean, this rescaling does not change
training and only fixes the reporting scale.

Loss names
    ce        cross-entropy, no weights
    wce       inverse frequency            w_c = N / n_c
    pwce      inverse power                w_c = (N / n_c) ** alpha
    sqce      inverse square root          w_c = sqrt(N / n_c)
    lwce      log-weighted (proposed)      w_c = 1 / log(1 + n_c)
    plwce     power-log (proposed)         w_c = 1 / log(1 + n_c) ** alpha
    eslwce    eps-smoothed (proposed)      w_c = 1 / (log(1 + n_c) + eps)
    brlwce    bounded relative (proposed)  w_c = 1 / (1 + log(1 + C * p_c)),  p_c = n_c / N
    combined  ablation only                w_c = 1 / (log(1 + n_c) + eps) ** alpha
    cb        class-balanced               w_c = (1 - beta) / (1 - beta ** n_c)
    focal     focal loss (unweighted)      (1 - p_t) ** gamma * CE
    logitadj  logit adjustment             CE on logits + tau * log(prior)
    ldam      LDAM margin loss             margin m_c ~ n_c ** (-1/4), max margin 0.5, scale 30
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PAPER_LOSSES = ['ce', 'wce', 'pwce', 'sqce', 'cb', 'focal', 'logitadj', 'ldam',
                'lwce', 'plwce', 'eslwce', 'brlwce']


def calculate_weights(class_counts, mode='ce', alpha=1.0, beta=0.9999, eps=0.1):
    """Per-class weights for a re-weighting mode, normalised to mean 1 over the classes.

    Args:
        class_counts: number of training samples per class.
        mode: 'ce', 'wce', 'pwce', 'sqce', 'lwce', 'plwce', 'eslwce', 'brlwce',
            'combined' or 'cb'.
        alpha: exponent of 'pwce', 'plwce' and 'combined'.
        beta: 'cb' parameter.
        eps: smoothing constant of 'eslwce' and 'combined'.

    Returns:
        torch.Tensor of shape (num_classes,), float32.
    """
    counts = np.array(class_counts, dtype=np.float32)
    total = np.sum(counts)
    safe_counts = counts + 1e-6

    if mode == 'ce':
        weights = np.ones_like(counts)
    elif mode == 'wce':
        weights = total / safe_counts
    elif mode == 'pwce':
        weights = np.power(total / safe_counts, alpha)
    elif mode == 'sqce':
        weights = np.sqrt(total / safe_counts)
    elif mode == 'lwce':
        weights = 1.0 / np.log1p(safe_counts)
    elif mode == 'plwce':
        weights = 1.0 / np.power(np.log1p(safe_counts), alpha)
    elif mode == 'eslwce':
        # The denominator is at least eps, so the weight stays finite even at n_c = 0.
        weights = 1.0 / (np.log1p(safe_counts) + eps)
    elif mode == 'brlwce':
        # Depends on proportions only (invariant to dataset replication); the added 1
        # keeps rare-class weights bounded as p_c approaches zero.
        num_classes = len(counts)
        p = safe_counts / total
        weights = 1.0 / (1.0 + np.log1p(num_classes * p))
    elif mode == 'combined':
        # alpha = 1 gives ES-LWCE, eps = 0 gives PLWCE, both give LWCE.
        weights = 1.0 / np.power(np.log1p(safe_counts) + eps, alpha)
    elif mode == 'cb':
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts) + 1e-6)
    else:
        raise ValueError(f'Unknown weight mode: {mode}')

    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class FocalLoss(nn.Module):
    """Focal loss: cross-entropy scaled by (1 - p_t) ** gamma, averaged over the batch."""

    def __init__(self, gamma=2.0, weights=None):
        super().__init__()
        self.gamma = gamma
        self.weights = weights

    def forward(self, logits, targets):
        if self.weights is not None:
            if self.weights.device != logits.device:
                self.weights = self.weights.to(logits.device)
            ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weights, reduction='none')
        else:
            ce_loss = F.cross_entropy(logits, targets.long(), reduction='none')
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        return ((1 - pt).pow(self.gamma) * ce_loss).mean()


class LogitAdjustedLoss(nn.Module):
    """Logit adjustment (Menon et al., ICLR 2021): CE on logits + tau * log(prior)."""

    def __init__(self, class_counts, tau=1.0):
        super().__init__()
        counts = np.array(class_counts, dtype=np.float32)
        prior = counts / np.sum(counts)
        self.adj = torch.tensor(tau * np.log(prior + 1e-12), dtype=torch.float32)

    def forward(self, logits, targets):
        if self.adj.device != logits.device:
            self.adj = self.adj.to(logits.device)
        return F.cross_entropy(logits + self.adj, targets.long())


class LDAMLoss(nn.Module):
    """LDAM (Cao et al., NeurIPS 2019) without the deferred re-weighting schedule.

    The target-class logit is reduced by a margin m_c proportional to n_c ** (-1/4),
    scaled so that the largest margin is max_m, and the logits are multiplied by s.
    """

    def __init__(self, class_counts, max_m=0.5, s=30.0):
        super().__init__()
        counts = np.array(class_counts, dtype=np.float32)
        m_list = 1.0 / np.sqrt(np.sqrt(counts))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32)
        self.s = s

    def forward(self, logits, targets):
        if self.m_list.device != logits.device:
            self.m_list = self.m_list.to(logits.device)
        targets = targets.long()
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.unsqueeze(1), True)
        batch_m = self.m_list[targets].unsqueeze(1)
        logits_m = torch.where(index, logits - batch_m, logits)
        return F.cross_entropy(self.s * logits_m, targets)


class ClassificationLoss(nn.Module):
    """Loss selected by name; see the module docstring for the list of names."""

    def __init__(self, loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0,
                 eps=0.1, tau=1.0):
        super().__init__()
        self.loss_name = loss_name

        # Substring matching: names that contain 'lwce' must be tested before 'lwce'.
        if 'brlwce' in loss_name:
            weight_mode = 'brlwce'
        elif 'combined' in loss_name:
            weight_mode = 'combined'
        elif 'plwce' in loss_name:
            weight_mode = 'plwce'
        elif 'eslwce' in loss_name:
            weight_mode = 'eslwce'
        elif 'lwce' in loss_name:
            weight_mode = 'lwce'
        elif 'pwce' in loss_name:
            weight_mode = 'pwce'
        elif 'sqce' in loss_name:
            weight_mode = 'sqce'
        elif 'wce' in loss_name:
            weight_mode = 'wce'
        elif 'cb' in loss_name:
            weight_mode = 'cb'
        else:
            weight_mode = 'ce'

        self.weights = None
        if weight_mode != 'ce':
            if class_counts is None:
                raise ValueError(f"Loss '{loss_name}' requires class_counts.")
            self.weights = calculate_weights(class_counts, mode=weight_mode, alpha=alpha,
                                             beta=beta, eps=eps)

        if 'logitadj' in loss_name:
            self.criterion = LogitAdjustedLoss(class_counts, tau=tau)
        elif 'ldam' in loss_name:
            self.criterion = LDAMLoss(class_counts)
        elif 'focal' in loss_name:
            self.criterion = FocalLoss(gamma=gamma, weights=self.weights)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=self.weights)

    def get_weights(self):
        """Normalised class weights actually used, or None for unweighted losses."""
        if self.weights is None:
            return None
        return self.weights.detach().cpu().numpy()

    def forward(self, logits, targets):
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            if self.criterion.weight is not None and self.criterion.weight.device != logits.device:
                self.criterion.weight = self.criterion.weight.to(logits.device)
        return self.criterion(logits, targets.long())


def get_clf_loss(loss_name, class_counts, alpha=1.0, beta=0.9999, gamma=2.0, eps=0.1, tau=1.0):
    """Build the loss called `loss_name` for the given training class counts."""
    return ClassificationLoss(loss_name, class_counts, alpha, beta, gamma, eps, tau)
