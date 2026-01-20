from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    counts = df["emotion_top1"].value_counts().sort_values(ascending=False)
    if counts.empty:
        return
    total = float(counts.sum())
    if total <= 0:
        return

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return

    raw_labels = counts.index.tolist()
    labels = [f"{lab} (n={int(cnt)})" for lab, cnt in zip(raw_labels, counts.values)]
    values = (counts.values / total) * 100.0

    score_data = []
    for lab in raw_labels:
        col = f"emo_{lab}"
        if col not in df:
            score_data.append([])
            continue
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
        score_data.append(vals)

    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [1, 1.6]})
    ax1, ax2 = axes

    y = np.arange(len(labels))
    ax1.barh(y, values, color="#4c78a8", alpha=0.9)
    ax1.set_title("Top-1 emotion")
    ax1.set_xlabel("Percent of episodes")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.grid(axis="x")

    ax2.violinplot(score_data, showmeans=False, showmedians=True, vert=False)
    ax2.set_title("Score distributions (emo_*)")
    ax2.set_yticks(np.arange(1, len(labels) + 1))
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("Score")
    ax2.grid(axis="x")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

