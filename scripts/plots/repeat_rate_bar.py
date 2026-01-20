from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import bootstrap_mean_ci, set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "redundancy_repeated" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "redundancy_repeated"]].copy()
    sub["redundancy_repeated"] = pd.to_numeric(sub["redundancy_repeated"], errors="coerce").fillna(0.0)
    if sub.empty:
        return

    rows = []
    for emo, group in sub.groupby("emotion_top1"):
        vals = (group["redundancy_repeated"] > 0).astype(float).to_numpy()
        if len(vals) == 0:
            continue
        mean = float(vals.mean())
        lo, hi = bootstrap_mean_ci(vals)
        rows.append((emo, len(vals), mean, lo, hi))
    if not rows:
        return
    rows.sort(key=lambda x: x[2])
    labels = [f"{r[0]} (n={r[1]})" for r in rows]
    means = np.array([r[2] for r in rows])
    lows = np.array([r[3] for r in rows])
    highs = np.array([r[4] for r in rows])
    errs = np.vstack([means - lows, highs - means])

    height = max(4.0, 0.35 * len(labels) + 1.5)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.errorbar(means, y, xerr=errs, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("P(repeated > 0)")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

