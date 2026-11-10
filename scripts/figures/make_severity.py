# -*- coding: utf-8 -*-
r"""
Severity analysis figure (Section 4.6).

For every dataset-and-imbalance setting we plot the macro-F1 gain of the
logarithmic family (mean of LWCE/PLWCE/ES-LWCE) against imbalance severity on a
log10(IR) axis, where IR = n_max / n_min.

Two panels make the honest point:
  (a) gain over WCE grows significantly with imbalance (the dynamic-range
      mechanism: WCE's weight ratio explodes and it collapses, the log family
      stays bounded, so the gap widens);
  (b) gain over plain CE is imbalance-independent (the log family is a safe
      default at every severity, not a high-IR-only remedy).

Output: severity.pdf
"""

import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from scipy import stats

rcParams["font.family"] = "serif"
rcParams["mathtext.fontset"] = "cm"
rcParams["pdf.fonttype"] = 42
rcParams["axes.linewidth"] = 0.6

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
LOG = ["lwce", "plwce", "eslwce"]

# ---- collect per-setting F1 and IR ---------------------------------------
settings = {}  # name -> {loss: F1}, domain


def add(name, domain, lossF1, ir):
    settings[name] = dict(F1=lossF1, dom=domain, ir=ir)


# image: exact exponential counts
def cc(ds, ir):
    K = 10 if ds == "cifar10" else 100
    nmax = 5000 if ds == "cifar10" else 500
    rho = ir ** (-1 / (K - 1))
    return [int(nmax * rho**i) for i in range(K)]


for ds, tag in [("cifar10", "CIFAR10_LT"), ("cifar100", "CIFAR100_LT")]:
    for ir in [10, 50, 100]:
        d = json.load(
            open(f"{ROOT}/results/{tag.lower()}/IR{ir}/results_agg.json", encoding="utf-8")
        )
        c = cc(ds, ir)
        add(
            f"{tag}_IR{ir}",
            "Image",
            {L: d[L]["F1_Macro_mean"] for L in LOG + ["ce", "wce"]},
            max(c) / min(c),
        )

# tabular / network: per-dataset mean F1 + IR from analysis weight_ratios (wce ratio == IR)
for dom, pre in [("tabular", "Tabular"), ("network", "Network")]:
    ws = json.load(open(f"{ROOT}/results/{dom}/analysis/analysis_summary.json", encoding="utf-8"))[
        "weight_ratios"
    ]
    ir_of = {r["Dataset"]: r["IR"] for r in ws if r["Loss"] == "wce"}
    agg = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(f"{ROOT}/results/{dom}/*_results.json"):
        if "all_results" in p:
            continue
        for r in json.load(open(p, encoding="utf-8")):
            agg[r["Dataset"]][r["Loss"]].append(r["F1_Macro"])
    for dsname, lf in agg.items():
        if dsname not in ir_of:
            continue
        add(
            f"{dom}:{dsname}",
            pre,
            {L: float(np.mean(lf[L])) for L in LOG + ["ce", "wce"]},
            ir_of[dsname],
        )

# ---- assemble arrays ------------------------------------------------------
COL = {"Image": "#2980b9", "Tabular": "#27ae60", "Network": "#c0392b"}
MRK = {"Image": "o", "Tabular": "s", "Network": "^"}


def logfam(s):
    return np.mean([s["F1"][L] for L in LOG])


logIR = np.array([np.log10(s["ir"]) for s in settings.values()])
gain_wce = np.array([logfam(s) - s["F1"]["wce"] for s in settings.values()])
gain_ce = np.array([logfam(s) - s["F1"]["ce"] for s in settings.values()])
doms = [s["dom"] for s in settings.values()]


def fit_line(x, y):
    b, a = np.polyfit(x, y, 1)  # slope, intercept
    r = np.corrcoef(x, y)[0, 1]
    return b, a, r


fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
for ax, gain, title in [
    (axes[0], gain_wce, "(a) vs. WCE"),
    (axes[1], gain_ce, "(b) vs. CE"),
]:
    for dom in ["Image", "Tabular", "Network"]:
        m = [d == dom for d in doms]
        ax.scatter(
            logIR[m],
            gain[m],
            s=34,
            c=COL[dom],
            marker=MRK[dom],
            edgecolors="k",
            linewidths=0.4,
            alpha=0.85,
            label=dom,
            zorder=3,
        )
    b, a, r = fit_line(logIR, gain)
    xs = np.linspace(logIR.min(), logIR.max(), 50)
    ax.plot(xs, a + b * xs, "k--", lw=1.0, alpha=0.7, zorder=2)
    ax.axhline(0, color="gray", lw=0.6, ls=":", zorder=1)
    ax.set_xlabel(r"Imbalance severity $\log_{10}(\mathrm{IR})$")
    ax.set_ylabel(r"Macro-$F_1$ gain of log family")
    ax.set_title(f"{title}  ($r={r:.2f},\\ R^2={r**2:.2f}$)", fontsize=9)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=8)
axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(ROOT, "figures", "severity.pdf"), bbox_inches="tight")
print("saved severity.pdf")

# ---- console numbers ------------------------------------------------------
print(f"\nn = {len(settings)} settings")
for lab, g in [("LogFam-WCE", gain_wce), ("LogFam-CE", gain_ce)]:
    r = stats.pearsonr(logIR, g)
    print(
        f"  {lab:11s} vs log10(IR): r={r.statistic:+.3f} R2={r.statistic**2:.3f} p={r.pvalue:.3f}"
    )
for dom in ["Tabular", "Network", "Image"]:
    m = np.array([d == dom for d in doms])
    r = stats.pearsonr(logIR[m], gain_wce[m])
    print(
        f"  [{dom}] LogFam-WCE: r={r.statistic:+.3f} R2={r.statistic**2:.3f} "
        f"p={r.pvalue:.3f} n={m.sum()}"
    )
