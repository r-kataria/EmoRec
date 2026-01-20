from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(corr_df: pd.DataFrame, emotions: List[str], metrics: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if corr_df.empty:
        return
    data = []
    for emo in emotions:
        row = []
        for metric in metrics:
            subset = corr_df[(corr_df["emotion"] == emo) & (corr_df["metric"] == metric)]
            row.append(np.nan if subset.empty else float(subset.iloc[0]["spearman_r"]))
        data.append(row)
    if not data:
        return

    mat = np.array(data, dtype=float)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(metrics) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels(emotions)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Spearman r")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

