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
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [f"{lab} (n={int(cnt)})" for lab, cnt in zip(counts.index.tolist(), counts.values)]
    values = (counts.values / total) * 100.0
    y = np.arange(len(labels))
    ax.barh(y, values, color="#4c78a8", alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel("Percent of episodes")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

