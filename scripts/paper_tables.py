"""Recompute the cross-domain numbers of the manuscript from the saved per-seed results.

Covers
  - the main-results table: per-domain macro-F1 and the mean rank over the 26 settings,
  - the per-domain rank spread of Section 4.3,
  - the statistics table: Friedman, Wilcoxon signed-rank of LWCE vs each loss with Holm
    correction, win/tie/loss, and the paired TOST equivalence analysis,
  - the ablation table, averaged over the 26 settings,
  - the inclusion-versus-selection analysis of Section 4.5 and the selected-alpha
    statistics of the sensitivity paragraph,
  - the CIFAR-LT gap numbers of Section 4.4,
and compares every value with the number printed in paper/main.tex, rounded to the
printed precision. It reads the files under results/ and writes
results/paper_tables_summary.json.

    python scripts/paper_tables.py
"""

import json
import os
import sys

import numpy as np
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(REPO, "results", "paper_tables_summary.json")

MAIN = [
    "ce",
    "wce",
    "pwce",
    "sqce",
    "cb",
    "focal",
    "logitadj",
    "ldam",
    "lwce",
    "plwce",
    "eslwce",
    "brlwce",
]  # the twelve losses of the main-results table
ABLATION = ["wce", "lwce", "plwce", "eslwce", "combined"]
SEEDS = ["42", "43", "44", "45", "46"]
TABULAR = [
    "credit_card_fraud",
    "aps_failure",
    "bank_marketing",
    "telco_churn",
    "german_credit",
    "secom",
    "credit_card_default",
    "glass",
    "steel_faults",
    "yeast",
    "page_blocks",
]
NETWORK = [
    "NSL-KDD",
    "UNSW-NB15",
    "CICIDS2017",
    "CICIDS2018",
    "CIC-DDoS2019",
    "Bot-IoT",
    "TON_IoT",
    "RT-IoT2022",
]
DOMAINS = ["image", "tabular", "network", "text"]


def _load(*parts):
    with open(os.path.join(REPO, *parts), encoding="utf-8") as f:
        return json.load(f)


def _seed_vector(per_seed):
    """{seed: metrics} -> macro-F1 of seeds 42..46 in order."""
    assert sorted(per_seed) == SEEDS, sorted(per_seed)
    return np.array([per_seed[s]["F1_Macro"] for s in SEEDS], dtype=float)


def _rows_to_vectors(rows):
    """Tabular/network row list -> {loss: macro-F1 of seeds 42..46}."""
    out = {}
    for loss in sorted({r["Loss"] for r in rows}):
        by_seed = {str(r["Seed"]): r for r in rows if r["Loss"] == loss}
        out[loss] = _seed_vector(by_seed)
    return out


def load_settings():
    """26 settings: domain, per-seed macro-F1 per loss, grid-search trials and selected params."""
    settings = {}
    for folder, tag in [("cifar10_lt", "CIFAR-10-LT"), ("cifar100_lt", "CIFAR-100-LT")]:
        trials = _load("results", folder, "optuna_trials.json")
        best = _load("results", folder, "optuna_checkpoint.json")
        for ir in (10, 50, 100):
            per_seed = _load("results", folder, f"IR{ir}", "results_per_seed.json")
            settings[f"{tag} rho={ir}"] = dict(
                domain="image",
                f1={k: _seed_vector(v) for k, v in per_seed.items()},
                trials=trials[str(ir)],
                best=best[str(ir)],
            )
    for domain, folder, names in [
        ("tabular", "tabular", TABULAR),
        ("network", "network", NETWORK),
    ]:
        trials = _load("results", folder, "_optuna_trials.json")
        best = _load("results", folder, "_best_params.json")
        for ds in names:
            rows = _load("results", folder, f"{ds}_results.json")
            settings[ds] = dict(
                domain=domain, f1=_rows_to_vectors(rows), trials=trials[ds], best=best[ds]
            )
    per_seed = _load("results", "text", "results_per_seed.json")
    settings["GoEmotions"] = dict(
        domain="text",
        f1={k: _seed_vector(v) for k, v in per_seed.items()},
        trials=_load("results", "text", "optuna_trials.json"),
        best=_load("results", "text", "optuna_checkpoint.json"),
    )
    assert len(settings) == 26
    return settings


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[idx]))
        adj[idx] = running
    return adj


def tost(d, margin):
    """Paired TOST on differences d (points): 90% CI, p of the two one-sided t-tests."""
    n = len(d)
    mean, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.95, n - 1)
    p_lower = 1 - stats.t.cdf((mean + margin) / se, n - 1)  # H0: mean <= -margin
    p_upper = stats.t.cdf((mean - margin) / se, n - 1)  # H0: mean >= +margin
    return dict(
        mean=mean, sd=sd, ci90=[mean - tcrit * se, mean + tcrit * se], p=max(p_lower, p_upper)
    )


def trial_value(trials, **params):
    """Proxy score of the grid point with the given parameter values."""
    hits = [
        t["value"]
        for t in trials
        if all(np.isclose(t["params"][k], v, atol=1e-9) for k, v in params.items())
    ]
    assert len(hits) == 1, (params, len(hits))
    return hits[0]


def compute(settings):
    names = list(settings)
    dom = np.array([settings[s]["domain"] for s in names])
    M = {
        loss: np.array([settings[s]["f1"][loss].mean() for s in names])
        for loss in MAIN + ["combined"]
    }
    c = {}

    # Table 2: per-domain means and mean ranks over the 26 settings (rank 1 = best macro-F1).
    scores = np.column_stack([M[loss] for loss in MAIN])
    ranks = np.vstack([stats.rankdata(-row, method="average") for row in scores])
    for j, loss in enumerate(MAIN):
        for d in DOMAINS:
            c[f"main.{loss}.{d}"] = M[loss][dom == d].mean()
            c[f"rank_by_domain.{loss}.{d}"] = ranks[dom == d, j].mean()
        c[f"main.{loss}.rank"] = ranks[:, j].mean()

    # Table 5: Friedman, Wilcoxon (LWCE vs each) with Holm over eleven comparisons, W/T/L.
    chi2, p_f = stats.friedmanchisquare(*[M[loss] for loss in MAIN])
    c["friedman.chi2"], c["friedman.p"] = chi2, p_f
    others = [loss for loss in MAIN if loss != "lwce"]
    raw = [stats.wilcoxon(M["lwce"], M[o]).pvalue for o in others]
    for o, p_raw, p_adj in zip(others, raw, holm(raw), strict=True):
        c[f"stats.{o}.p_raw"], c[f"stats.{o}.p_holm"] = p_raw, p_adj
        c[f"stats.{o}.wtl"] = "{}/{}/{}".format(
            int((M["lwce"] > M[o]).sum()),
            int((M["lwce"] == M[o]).sum()),
            int((M["lwce"] < M[o]).sum()),
        )

    # TOST of each loss against LWCE, differences (other - LWCE) in macro-F1 points.
    for o in ["brlwce", "eslwce", "pwce", "logitadj"]:
        r = tost(100 * (M[o] - M["lwce"]), margin=1.0)
        c[f"tost.{o}.ci_lo"], c[f"tost.{o}.ci_hi"] = r["ci90"]
        c[f"tost.{o}.p"], c[f"tost.{o}.sd"] = r["p"], r["sd"]
    c["tost.brlwce.p_half"] = tost(100 * (M["brlwce"] - M["lwce"]), margin=0.5)["p"]
    sd_seed = [100 * settings[s]["f1"]["lwce"].std(ddof=0) for s in names]
    c["tost.lwce_seed_sd_median.ddof0"] = float(np.median(sd_seed))
    c["tost.lwce_seed_sd_median.ddof1"] = float(
        np.median([100 * settings[s]["f1"]["lwce"].std(ddof=1) for s in names])
    )

    # Table 4: ablation averaged over the 26 settings.
    for loss in ABLATION:
        c[f"ablation.{loss}"] = M[loss].mean()

    # Section 4.5: inclusion versus selection. Proxy gain of the selected point over the
    # LWCE anchor, test difference against LWCE, and the "selection penalty" = gain - test.
    anchors = {
        "plwce": dict(alpha=1.0),
        "eslwce": dict(eps=0.0),
        "combined": dict(alpha=1.0, eps=0.0),
    }
    reduced = np.zeros(len(names), dtype=bool)
    for ext, anchor in anchors.items():
        gain, test, at_anchor = [], [], []
        for i, s in enumerate(names):
            tr, sel = settings[s]["trials"][ext], settings[s]["best"][ext]
            assert np.isclose(max(t["value"] for t in tr), trial_value(tr, **sel))
            gain.append(100 * (trial_value(tr, **sel) - trial_value(tr, **anchor)))
            test.append(100 * (M[ext][i] - M["lwce"][i]))
            at_anchor.append(all(np.isclose(sel[k], v) for k, v in anchor.items()))
        gain, test, at_anchor = map(np.array, (gain, test, at_anchor))
        reduced |= at_anchor
        c[f"selection.{ext}.proxy_gain"] = gain.mean()
        c[f"selection.{ext}.test_diff"] = test.mean()
        c[f"selection.{ext}.penalty"] = (gain - test).mean()
        c[f"selection.{ext}.n_anchor"] = int(at_anchor.sum())
        if ext == "plwce":
            c["selection.plwce.n_not_anchor"] = int((~at_anchor).sum())
            c["selection.plwce.pearson_r"] = stats.pearsonr(gain, test)[0]
    c["selection.n_settings_with_reduction"] = int(reduced.sum())

    # CICIDS2017: the selected exponent stretches the LWCE weight ratio to ratio**alpha.
    counts = np.array(_load("results", "network", "_class_counts.json")["CICIDS2017"], float)
    w = 1.0 / np.log1p(counts[counts > 0])
    alpha = settings["CICIDS2017"]["best"]["plwce"]["alpha"]
    i17 = names.index("CICIDS2017")
    c["cicids2017.alpha"] = alpha
    c["cicids2017.lwce_ratio"] = w.max() / w.min()
    c["cicids2017.plwce_ratio"] = (w.max() / w.min()) ** alpha
    c["cicids2017.test_cost"] = 100 * (M["lwce"][i17] - M["plwce"][i17])

    # Sensitivity paragraph: selected PLWCE exponent across the 26 settings.
    a_star = np.array([settings[s]["best"]["plwce"]["alpha"] for s in names])
    grid = np.array(sorted(t["params"]["alpha"] for t in settings[names[0]]["trials"]["plwce"]))
    c["alpha.min"], c["alpha.max"], c["alpha.median"] = (
        a_star.min(),
        a_star.max(),
        np.median(a_star),
    )
    for d in DOMAINS:
        c[f"alpha.mean.{d}"] = a_star[dom == d].mean()
    c["alpha.n_lower_end"] = int(np.isclose(a_star, grid.min()).sum())
    c["alpha.n_upper_end"] = int(np.isclose(a_star, grid.max()).sum())
    c["alpha.n_at_one"] = int(np.isclose(a_star, 1.0).sum())
    c["alpha.n_interior"] = (
        len(a_star) - c["alpha.n_lower_end"] - c["alpha.n_upper_end"] - c["alpha.n_at_one"]
    )
    pw = np.array([settings[s]["best"]["pwce"]["alpha"] for s in names])
    c["pwce.n_below_0.3"] = int((pw < 0.3).sum())

    # Section 4.4: CIFAR-LT gaps of LWCE over WCE and over CE.
    for tag in ["CIFAR-10-LT", "CIFAR-100-LT"]:
        for ir in (10, 50, 100):
            i = names.index(f"{tag} rho={ir}")
            c[f"cifar.{tag}.{ir}.lwce_minus_wce"] = M["lwce"][i] - M["wce"][i]
            c[f"cifar.{tag}.{ir}.lwce_minus_ce"] = M["lwce"][i] - M["ce"][i]
    return c


# Numbers printed in the paper, with the number of printed decimals.
PAPER = {}
_main = {  # Image, Tabular, Network, Text, Mean rank (Table 2)
    "ce": (0.578, 0.714, 0.642, 0.412, 7.63),
    "wce": (0.556, 0.683, 0.622, 0.374, 9.23),
    "pwce": (0.581, 0.716, 0.666, 0.421, 5.63),
    "sqce": (0.577, 0.723, 0.660, 0.429, 5.73),
    "cb": (0.549, 0.701, 0.629, 0.399, 9.62),
    "focal": (0.559, 0.715, 0.663, 0.405, 7.77),
    "logitadj": (0.604, 0.720, 0.631, 0.421, 4.85),
    "ldam": (0.582, 0.721, 0.653, 0.403, 6.65),
    "lwce": (0.578, 0.730, 0.665, 0.419, 4.65),
    "plwce": (0.577, 0.721, 0.666, 0.434, 6.25),
    "eslwce": (0.578, 0.726, 0.669, 0.419, 5.21),
    "brlwce": (0.580, 0.729, 0.668, 0.415, 4.77),
}
for _l, _v in _main.items():
    for _d, _x in zip(DOMAINS, _v[:4], strict=True):
        PAPER[f"main.{_l}.{_d}"] = (_x, 3)
    PAPER[f"main.{_l}.rank"] = (_v[4], 2)
for _k, _v in {
    "logitadj.image": 1.2,
    "logitadj.network": 7.0,
    "plwce.text": 1.0,
    "plwce.image": 6.8,
    "lwce.tabular": 3.1,
    "lwce.image": 6.3,
}.items():
    PAPER[f"rank_by_domain.{_k}"] = (_v, 1)
PAPER["friedman.chi2"] = (64.6, 1)
for _o, (_p, _w) in {
    "brlwce": (1.000, "16/0/10"),
    "logitadj": (1.000, "11/0/15"),
    "eslwce": (1.000, "14/2/10"),
    "pwce": (1.000, "13/0/13"),
    "sqce": (1.000, "17/0/9"),
    "plwce": (1.000, "19/0/7"),
    "ldam": (0.143, "18/0/8"),
    "ce": (0.001, "20/0/6"),
    "focal": (0.008, "20/0/6"),
    "wce": (0.003, "20/0/6"),
    "cb": (0.003, "22/0/4"),
}.items():
    PAPER[f"stats.{_o}.p_holm"] = (_p, 3)
    PAPER[f"stats.{_o}.wtl"] = (_w, None)
for _o, (_lo, _hi) in {
    "brlwce": (-0.19, 0.39),
    "eslwce": (-0.51, 0.45),
    "pwce": (-1.29, 0.28),
    "logitadj": (-2.34, 0.58),
}.items():
    PAPER[f"tost.{_o}.ci_lo"], PAPER[f"tost.{_o}.ci_hi"] = (_lo, 2), (_hi, 2)
PAPER.update(
    {
        "tost.eslwce.p": (0.001, 3),
        "tost.pwce.p": (0.15, 2),
        "tost.logitadj.p": (0.45, 2),
        "tost.brlwce.p_half": (0.013, 3),
        "tost.pwce.sd": (2.4, 1),
        "tost.logitadj.sd": (4.4, 1),
        "tost.brlwce.sd": (0.9, 1),
        "tost.lwce_seed_sd_median.ddof1": (0.67, 2),
        "ablation.wce": (0.6231, 4),
        "ablation.lwce": (0.6630, 4),
        "ablation.plwce": (0.6600, 4),
        "ablation.eslwce": (0.6627, 4),
        "ablation.combined": (0.6620, 4),
        "selection.n_settings_with_reduction": (6, 0),
        "selection.plwce.n_not_anchor": (25, 0),
        "selection.plwce.proxy_gain": (4.35, 2),
        "selection.plwce.test_diff": (-0.31, 2),
        "selection.plwce.pearson_r": (-0.21, 2),
        "selection.plwce.penalty": (4.66, 2),
        "selection.combined.penalty": (4.59, 2),
        "selection.eslwce.penalty": (1.19, 2),
        "cicids2017.alpha": (3.34, 2),
        "cicids2017.lwce_ratio": (5.1, 1),
        "cicids2017.plwce_ratio": (229, 0),
        "cicids2017.test_cost": (3.5, 1),
        "alpha.min": (0.50, 2),
        "alpha.max": (5.62, 2),
        "alpha.median": (1.83, 2),
        "alpha.mean.image": (1.26, 2),
        "alpha.mean.tabular": (2.25, 2),
        "alpha.mean.network": (2.63, 2),
        "alpha.mean.text": (4.10, 2),
        "alpha.n_lower_end": (5, 0),
        "alpha.n_upper_end": (0, 0),
        "alpha.n_at_one": (1, 0),
        "alpha.n_interior": (20, 0),
        "pwce.n_below_0.3": (11, 0),
        "cifar.CIFAR-100-LT.10.lwce_minus_wce": (0.015, 3),
        "cifar.CIFAR-100-LT.100.lwce_minus_wce": (0.058, 3),
    }
)
# Statements checked as bounds rather than printed values.
BOUNDS = {
    "friedman.p": ("approx", 1.3e-9),
    "tost.brlwce.p": ("<", 0.001),
    **{f"cifar.CIFAR-10-LT.{ir}.lwce_minus_wce": ("abs<=", 0.006) for ir in (10, 50, 100)},
    **{
        f"cifar.{t}.{ir}.lwce_minus_ce": ("abs<=", 0.006)
        for t in ("CIFAR-10-LT", "CIFAR-100-LT")
        for ir in (10, 50, 100)
    },
}


def compare(c):
    rows = []
    for key, (paper, dec) in PAPER.items():
        val = c[key]
        if dec is None:
            ok = val == paper
            shown = val
        else:
            shown = round(float(val), dec)
            ok = np.isclose(shown, paper, atol=10 ** -(dec + 2))
        rows.append((key, paper, shown, "ok" if ok else "MISMATCH"))
    for key, (kind, bound) in BOUNDS.items():
        val = float(c[key])
        if kind == "<":
            ok = val < bound
        elif kind == "abs<=":
            ok = abs(val) <= bound  # the paper states |delta| <= 0.006 as an exact bound
        else:
            ok = f"{val:.1e}" == f"{bound:.1e}"
        rows.append((key, f"{kind} {bound}", f"{val:.4g}", "ok" if ok else "MISMATCH"))
    return rows


def main():
    c = compute(load_settings())
    rows = compare(c)
    width = max(len(r[0]) for r in rows)
    for key, paper, shown, status in rows:
        print(f"{key:{width}s}  paper={paper!s:>12}  recomputed={shown!s:>12}  {status}")
    bad = [r for r in rows if r[3] != "ok"]
    print(
        f"\n{len(rows) - len(bad)} of {len(rows)} printed values reproduced; {len(bad)} mismatches."
    )
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(
            dict(
                computed={k: (v if isinstance(v, str) else float(v)) for k, v in c.items()},
                comparison=[
                    dict(key=k, paper=str(p), recomputed=str(s), status=st) for k, p, s, st in rows
                ],
            ),
            f,
            indent=1,
            ensure_ascii=False,
        )
    print("wrote", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
