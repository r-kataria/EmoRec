from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(pairs: List[Tuple[str, float]], out_path: Path, title: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    if not pairs:
        return
    df = pd.DataFrame(pairs, columns=["emotion_top1", "value"])
    stats = df.groupby("emotion_top1")["value"].agg(["count", "median"]).sort_values("median", ascending=True)
    labels = [f"{idx} (n={int(row['count'])})" for idx, row in stats.iterrows()]
    data = [df[df["emotion_top1"] == idx]["value"].to_numpy() for idx in stats.index.tolist()]

    height = max(4.0, 0.35 * len(labels) + 1.5)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, vert=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("#4c78a8")
        pc.set_alpha(0.7)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

