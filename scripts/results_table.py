"""Turn per-method, per-seed metric lists into a results table (CSV and
LaTeX) and an ablation bar chart, both with 95% confidence intervals.
Feeds the paper's results table once real baseline runs exist.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ci(v):
    v = np.array(v, float)
    return v.mean(), 1.96 * v.std() / np.sqrt(len(v))


def summarize(runs):
    """runs: {method: {metric: [values over seeds]}} -> a DataFrame, also
    written to results/table.csv, with a LaTeX version printed to stdout."""
    rows = []
    for m, mets in runs.items():
        row = {"method": m}
        for k, v in mets.items():
            mu, c = ci(v)
            row[k] = f"{mu:.3f}±{c:.3f}"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv("results/table.csv", index=False)
    print(df.to_string(index=False))
    print("\nLaTeX:\n", df.to_latex(index=False))
    return df


def ablation_bar(runs, metric="rescue_rate", out="results/ablation.png"):
    names = list(runs)
    mus = [ci(runs[n][metric])[0] for n in names]
    errs = [ci(runs[n][metric])[1] for n in names]
    plt.figure(figsize=(5, 3))
    plt.bar(names, mus, yerr=errs, capsize=4)
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print("saved", out)
