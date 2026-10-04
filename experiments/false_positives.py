"""How often does each test declare a winner between two models that are equally good?

Two decision trees identical except for their random seed have the same expected accuracy, so every
"significant" result is a false positive. A correct test at alpha = 0.05 should be fooled ~5% of the time.

    python experiments/false_positives.py      -> prints the table, writes results/false_positives.{csv,png}
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from modelduel import duel

TRIALS, SUBSAMPLE, ALPHA = 200, 0.6, 0.05
OUT = Path(__file__).resolve().parents[1] / "results"
DATASETS = {"breast_cancer": load_breast_cancer, "wine": load_wine, "digits": load_digits}


def wilson(k, n, z=1.96):
    c, h = (k + z * z / 2) / (n + z * z), z * np.sqrt(k * (n - k) / n + z * z / 4) / (n + z * z)
    return c - h, c + h


def main():
    OUT.mkdir(exist_ok=True)
    rows = []
    for name, load in DATASETS.items():
        X, y = load(return_X_y=True)
        for t in range(TRIALS):
            Xs, _, ys, _ = train_test_split(X, y, train_size=SUBSAMPLE, stratify=y, random_state=t)
            a = DecisionTreeClassifier(max_features="sqrt", random_state=2 * t)
            b = DecisionTreeClassifier(max_features="sqrt", random_state=2 * t + 1)
            r = duel(a, b, Xs, ys, alpha=ALPHA, random_state=t)
            rows.append({"dataset": name, "trial": t, "n": len(ys), "diff": r.diff,
                         "naive_p": r.naive_p_value, "corrected_p": r.p_value})
        print(f"{name}: done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "false_positives.csv", index=False)

    table = []
    for name, g in df.groupby("dataset", sort=False):
        for test, col in [("naive paired t-test", "naive_p"), ("corrected (modelduel)", "corrected_p")]:
            k = int((g[col] < ALPHA).sum())
            lo, hi = wilson(k, len(g))
            table.append({"dataset": f"{name} (n={g.n.iloc[0]})", "test": test, "false_positive_rate": k / len(g),
                          "ci_low": lo, "ci_high": hi})
    table = pd.DataFrame(table)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    table.to_csv(OUT / "false_positive_rates.csv", index=False)
    plot(table)


def plot(table):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = list(dict.fromkeys(table.dataset))
    x = np.arange(len(datasets))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (test, color) in enumerate([("naive paired t-test", "#d9480f"), ("corrected (modelduel)", "#2a6fdb")]):
        t = table[table.test == test].set_index("dataset").loc[datasets]
        rates = t.false_positive_rate * 100
        err = np.vstack([rates - t.ci_low * 100, t.ci_high * 100 - rates])
        ax.bar(x + (i - 0.5) * 0.38, rates, 0.36, yerr=err, capsize=4, color=color, label=test)
    ax.axhline(ALPHA * 100, color="#555", ls="--", lw=1, label="what a correct test should give (5%)")
    ax.set_xticks(x, datasets)
    ax.set(ylabel="Times a 'winner' was declared (%)",
           title=f"Two equally good models, {TRIALS} trials each: how often is noise called significant?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "false_positives.png", dpi=150)


if __name__ == "__main__":
    main()
