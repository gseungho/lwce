# -*- coding: utf-8 -*-
r"""
Sensitivity figure (Section 4.7, RQ3).

Reuses the recorded proxy-validation macro-F1 of every Optuna trial---no extra
training---to show three things:

  (a) PLWCE macro-F1 varies with the exponent alpha and peaks at an interior
      optimum, i.e. alpha is a genuine, active accuracy lever (RQ3);
  (b) the selected alpha* differs systematically across the 26 settings and by
      domain, i.e. its optimal value is dataset-dependent;
  (c) ES-LWCE macro-F1 is nearly flat in the smoothing constant epsilon, i.e.
      epsilon is a safeguard rather than an accuracy knob.

Output: sensitivity.pdf
"""

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


# ---- load all proxy trials, keyed by "Domain:name" --------------------------
def load_trials():
    T = {}
    for ds, tag in [("CIFAR10_LT", "CIFAR10"), ("CIFAR100_LT", "CIFAR100")]:
        d = json.load(open(f"{ROOT}/results/{ds.lower()}/optuna_trials.json", encoding="utf-8"))
        for ir, v in d.items():
            T[f"Image:{tag}-IR{ir}"] = v
    for path, dom in [
        (f"{ROOT}/results/tabular/_optuna_trials.json", "Tabular"),
        (f"{ROOT}/results/network/_optuna_trials.json", "Network"),
    ]:
        for ds, v in json.load(open(path, encoding="utf-8")).items():
            T[f"{dom}:{ds}"] = v
    d = json.load(open(f"{ROOT}/results/text/optuna_trials.json", encoding="utf-8"))
    T["Text:GoEmotions"] = d if "plwce" in d else list(d.values())[0]
    return T


T = load_trials()


# ---- selected parameters, derived from the same best-params checkpoints ------
def load_selected():
    out = []
    for ds, tag in [("CIFAR10_LT", "CIFAR10"), ("CIFAR100_LT", "CIFAR100")]:
        d = json.load(open(f"{ROOT}/results/{ds.lower()}/optuna_checkpoint.json", encoding="utf-8"))
        for ir, v in d.items():
            out.append(
                (
                    "Image",
                    f"{tag}-IR{ir}",
                    v.get("plwce", {}).get("alpha"),
                    v.get("eslwce", {}).get("eps"),
                )
            )
    for path, dom in [
        (f"{ROOT}/results/tabular/_best_params.json", "Tabular"),
        (f"{ROOT}/results/network/_best_params.json", "Network"),
    ]:
        for ds, v in json.load(open(path, encoding="utf-8")).items():
            out.append((dom, ds, v.get("plwce", {}).get("alpha"), v.get("eslwce", {}).get("eps")))
    d = json.load(open(f"{ROOT}/results/text/optuna_checkpoint.json", encoding="utf-8"))
    out.append(
        ("Text", "GoEmotions", d.get("plwce", {}).get("alpha"), d.get("eslwce", {}).get("eps"))
    )
    return out


def curve(trials, param):
    pts = [
        (t["params"][param], t["value"])
        for t in trials
        if t.get("value") is not None and param in t.get("params", {})
    ]
    pts.sort()
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


# representative datasets, one per domain
REP = [
    ("Image:CIFAR100-IR100", "CIFAR-100-LT ($\\rho{=}100$)", "#2980b9"),
    ("Tabular:yeast", "yeast (tabular, 10-cls)", "#27ae60"),
    ("Network:NSL-KDD", "NSL-KDD (network, 5-cls)", "#c0392b"),
]

COL = {"Image": "#2980b9", "Tabular": "#27ae60", "Network": "#c0392b", "Text": "#8e44ad"}
MRK = {"Image": "o", "Tabular": "s", "Network": "^", "Text": "D"}

fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))

# (a) alpha -> proxy F1
axa = axes[0]
for key, lab, col in REP:
    x, y = curve(T[key]["plwce"], "alpha")
    axa.plot(x, y, marker="o", ms=2.5, lw=1.2, color=col, label=lab)
    axa.axvline(x[np.argmax(y)], color=col, ls=":", lw=0.8, alpha=0.6)
axa.axvline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
axa.text(
    1.02,
    axa.get_ylim()[0],
    r"$\alpha{=}1$ (LWCE)",
    fontsize=6.5,
    color="gray",
    rotation=90,
    va="bottom",
    ha="left",
)
axa.set_xlabel(r"PLWCE exponent $\alpha$")
axa.set_ylabel("Proxy-val. macro-$F_1$")
axa.set_title(r"(a) $\alpha$ is an active lever", fontsize=9)
axa.legend(fontsize=6.0, loc="lower center")
axa.grid(alpha=0.25)
axa.tick_params(labelsize=8)

# (b) selected alpha* distribution by domain
axb = axes[1]
sel = load_selected()  # [(domain,name,alpha,eps),...]
doms = ["Image", "Tabular", "Network", "Text"]
rng = np.random.RandomState(0)
for i, dom in enumerate(doms):
    ys = [s[2] for s in sel if s[0] == dom and s[2] is not None]
    xs = np.full(len(ys), i) + (np.linspace(-0.18, 0.18, len(ys)) if len(ys) > 1 else [0])
    axb.scatter(
        xs, ys, s=26, c=COL[dom], marker=MRK[dom], edgecolors="k", linewidths=0.4, alpha=0.85
    )
    axb.plot([i - 0.28, i + 0.28], [np.mean(ys)] * 2, color="k", lw=1.4, zorder=3)
axb.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7)
axb.set_xticks(range(len(doms)))
axb.set_xticklabels(doms, fontsize=7.5, rotation=15)
axb.set_ylabel(r"Selected $\alpha^\ast$")
axb.set_title(r"(b) $\alpha^\ast$ is dataset-dependent", fontsize=9)
axb.grid(axis="y", alpha=0.25)
axb.tick_params(labelsize=8)

# (c) epsilon -> proxy F1 (flat)
axc = axes[2]
for key, lab, col in REP:
    x, y = curve(T[key]["eslwce"], "eps")
    axc.plot(x, y, marker="o", ms=2.5, lw=1.2, color=col, label=lab)
axc.set_xscale("log")
axc.set_xlabel(r"ES-LWCE smoothing $\varepsilon$")
axc.set_ylabel("Proxy-val. macro-$F_1$")
axc.set_title(r"(c) $\varepsilon$ is nearly inert", fontsize=9)
axc.grid(alpha=0.25, which="both")
axc.tick_params(labelsize=8)

fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(ROOT, "figures", "sensitivity.pdf"), bbox_inches="tight")
print("saved sensitivity.pdf")


# ---- console numbers --------------------------------------------------------
def swing(trials, p):
    _, y = curve(trials, p)
    return (y.max() - y.min()) if len(y) else 0.0


a = np.array([swing(v["plwce"], "alpha") for v in T.values() if "plwce" in v])
e = np.array([swing(v["eslwce"], "eps") for v in T.values() if "eslwce" in v])
al = np.array([s[2] for s in sel if s[2] is not None])
print(f"alpha* : range [{al.min():.2f},{al.max():.2f}], median {np.median(al):.2f}")
print(
    f"alpha F1 swing median {np.median(a):.3f} ; eps F1 swing median {np.median(e):.3f} "
    f"({np.median(a) / np.median(e):.1f}x)"
)
