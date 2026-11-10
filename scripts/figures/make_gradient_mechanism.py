# -*- coding: utf-8 -*-
r"""
§4.7 Optimization stability / mechanism figure.

Per-epoch curves on CIFAR-100-LT (rho=100, 100 classes):
  (a) training loss, (b) parameter gradient L2-norm,
  (c) minority-to-majority per-sample logit-gradient ratio
for CE, WCE, CB, LWCE and PLWCE (mean over seeds).

Shows the early gradient amplification of WCE/CB and how the log family avoids it.
Output: figures/gradient_mechanism.pdf
"""

import io
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["pdf.fonttype"] = 42
rcParams["axes.linewidth"] = 0.6

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
CKPT = os.path.join(ROOT, "results", "cifar100_lt", "results_checkpoint.json")
d = json.load(io.open(CKPT, encoding="utf-8"))
SEEDS = [42, 43, 44, 45, 46]
IR = "IR100"

# Five representative losses: WCE/CB unbounded, LWCE/PLWCE proposed, CE baseline
SHOW = [
    ("ce", "CE", "#7f7f7f", "-"),
    ("wce", "WCE", "#c0392b", "-"),
    ("cb", "CB", "#e67e22", (0, (4, 2))),
    ("lwce", "LWCE", "#1a5276", "-"),
    ("plwce", "PLWCE", "#2980b9", (0, (3, 1.5))),
]


def mean_curve(loss, key):
    arr = np.array([d[f"{IR}_{loss}_s{s}"]["history"][key] for s in SEEDS], dtype=float)
    return np.nanmean(arr, 0)


fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9))
for loss, lab, col, ls in SHOW:
    ep = np.arange(len(mean_curve(loss, "train_loss")))
    axes[0].plot(ep, mean_curve(loss, "train_loss"), color=col, ls=ls, lw=1.4, label=lab)
    axes[1].plot(ep, mean_curve(loss, "grad_norm"), color=col, ls=ls, lw=1.4, label=lab)
    axes[2].plot(ep, mean_curve(loss, "grad_ratio"), color=col, ls=ls, lw=1.4, label=lab)

axes[0].set_ylabel("Training loss")
axes[0].set_title("(a) Training loss", fontsize=9)
axes[1].set_yscale("log")
axes[1].set_title("(b) Gradient norm", fontsize=9)
axes[1].set_ylabel(r"$\|\nabla_\theta \mathcal{L}\|_2$")
axes[2].set_yscale("log")
axes[2].set_title("(c) Minority/majority grad. ratio", fontsize=9)
axes[2].set_ylabel(r"$R_G$")
axes[2].axhline(1.0, color="k", ls=":", lw=0.8, alpha=0.6)
for ax in axes:
    ax.set_xlabel("Epoch")
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=8)
axes[0].legend(fontsize=7.5, loc="upper right", framealpha=0.95)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(ROOT, "figures", "gradient_mechanism.pdf"), bbox_inches="tight")
print("saved gradient_mechanism.pdf")

# --- numbers quoted in the paper ---
print("\nCIFAR-100 IR=100 - mechanism numbers (mean over seeds):")
print(
    f"{'loss':8s} | {'gnorm_final':>11} | {'ratio_peak':>10} | {'ratio_ep0':>9} | "
    f"{'ratio_final':>11}"
)
for loss, lab, *_ in SHOW + [("pwce", "PWCE"), ("eslwce", "ES-LWCE")]:
    gn = mean_curve(loss, "grad_norm")
    gr = mean_curve(loss, "grad_ratio")
    print(f"{lab:8s} | {gn[-1]:11.3f} | {np.nanmax(gr):10.2f} | {gr[0]:9.2f} | {gr[-1]:11.2f}")

# --- dynamic range of the normalised weights (Proposition 3) ---
print("\nDynamic range of the normalised weights (CIFAR-100 IR=100):")
print(f"{'loss':10s} | {'max_w':>7} {'min_w':>7} {'max/min':>8} {'CV':>6}")
for loss in ["wce", "pwce", "sqce", "lwce", "plwce", "eslwce", "brlwce", "cb"]:
    w = d[f"{IR}_{loss}_s42"]["metrics"].get("class_weights")
    if w is None:
        continue
    w = np.array(w)
    print(
        f"{loss:10s} | {w.max():7.3f} {w.min():7.3f} {w.max() / w.min():8.1f} "
        f"{w.std() / w.mean():6.2f}"
    )
