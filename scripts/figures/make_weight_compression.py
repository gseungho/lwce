# -*- coding: utf-8 -*-
r"""
Visualisation of Proposition 3 (dynamic-range compression).

y-axis: rare-to-frequent raw-weight ratio w(n_c) / w(n_max) for a fixed n_max.
The ratio is invariant to the mean normalisation, so raw weights are used.
The formulas match lwce.losses.calculate_weights (log1p = log(1 + n)).

Output: figures/weight_compression.pdf (vector)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["axes.linewidth"] = 0.6
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
rcParams["pdf.fonttype"] = 42  # embed TrueType fonts

N_MAX = 10_000  # majority-class count (reference)
N_HI = 300  # x-axis limit (curves barely differ beyond n = 200)
n = np.arange(1, N_HI + 1)  # minority-class count n_c


def ratio(wfun):
    """w(n_c) / w(n_max) - the rare-to-frequent ratio of Proposition 3."""
    return wfun(n) / wfun(N_MAX)


# raw weights as in lwce.losses.calculate_weights
def w_wce(x):  # WCE: 1/n
    return 1.0 / x


def w_sqce(x):  # inverse square root: 1/sqrt(n)
    return 1.0 / np.sqrt(x)


def w_lwce(x):  # LWCE: 1/log(1+n)
    return 1.0 / np.log1p(x)


def w_plwce(x, a):  # PLWCE: 1/log(1+n)^alpha
    return 1.0 / np.log1p(x) ** a


def w_eslw(x, e):  # ES-LWCE: 1/(log(1+n)+eps)
    return 1.0 / (np.log1p(x) + e)


fig, ax = plt.subplots(figsize=(5.4, 3.2))

# --- frequency-based baselines ---
ax.plot(n, ratio(w_wce), color="#c0392b", lw=2.0, ls="-", label=r"WCE  $1/n_c$")
ax.plot(n, ratio(w_sqce), color="#e67e22", lw=1.6, ls="-", label=r"Inverse-sqrt  $1/\sqrt{n_c}$")

# --- proposed family ---
ax.plot(
    n,
    ratio(lambda x: w_plwce(x, 3.0)),
    color="#8e44ad",
    lw=1.4,
    ls=(0, (5, 2)),
    label=r"PLWCE  $\alpha=3$",
)
ax.plot(
    n,
    ratio(lambda x: w_plwce(x, 2.0)),
    color="#2980b9",
    lw=1.4,
    ls=(0, (3, 1.5)),
    label=r"PLWCE  $\alpha=2$",
)
ax.plot(n, ratio(w_lwce), color="#1a5276", lw=2.4, ls="-", label=r"LWCE  $1/\log(1+n_c)$")
ax.plot(
    n,
    ratio(lambda x: w_eslw(x, 1.0)),
    color="#16a085",
    lw=1.6,
    ls=(0, (1, 1)),
    label=r"ES-LWCE  $\varepsilon=1$",
)

ax.axhline(1.0, color="0.4", lw=0.8, ls=(0, (4, 3)))
ax.text(N_HI - 3, 1.05, "majority baseline", ha="right", va="bottom", fontsize=7.5, color="0.35")

ax.set_yscale("log")
ax.set_xlim(0, N_HI)
ax.set_ylim(0.8, 2e4)
ax.set_xlabel(
    r"Minority-class sample count $n_c$  (majority fixed at $n_{\max}=10^4$)", fontsize=9.5
)
ax.set_ylabel(r"Weight ratio $w_{\min}/w_{\max}$", fontsize=9.5)
ax.tick_params(labelsize=8.5)
ax.grid(True, which="major", color="0.9", lw=0.5)
ax.grid(True, which="minor", color="0.96", lw=0.4)
ax.set_axisbelow(True)

# annotate the ratio at n_c = 1 (imbalance rho = 10^4)
ax.annotate(
    r"WCE $\to \rho=10^4$",
    xy=(1, w_wce(1) / w_wce(N_MAX)),
    xytext=(70, 6000),
    fontsize=8,
    color="#c0392b",
    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7),
)
ax.annotate(
    r"LWCE $\approx 13$ (max)",
    xy=(1, w_lwce(1) / w_lwce(N_MAX)),
    xytext=(120, 30),
    fontsize=8,
    color="#1a5276",
    arrowprops=dict(arrowstyle="->", color="#1a5276", lw=0.7),
)

ax.legend(fontsize=7.6, loc="upper right", framealpha=0.95, ncol=1, handlelength=2.4)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(ROOT, "figures", "weight_compression.pdf"), bbox_inches="tight")
print("saved weight_compression.pdf")

# print the numbers quoted in the paper
print(
    f"ratio at n_c=1 - WCE={w_wce(1) / w_wce(N_MAX):.0f}, "
    f"sqrt={w_sqce(1) / w_sqce(N_MAX):.1f}, "
    f"LWCE={w_lwce(1) / w_lwce(N_MAX):.2f}, "
    f"PLWCE(a=2)={w_plwce(1, 2) / w_plwce(N_MAX, 2):.1f}, "
    f"ES-LWCE(e=1)={w_eslw(1, 1) / w_eslw(N_MAX, 1):.2f}"
)
