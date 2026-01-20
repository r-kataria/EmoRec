from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ._common import pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, metrics: List[str], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df:
        return
    rows = []
    for metric in metrics:
        if metric not in df:
            continue
        sub = df[[metric, "stereotype_label"]].copy()
        sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
        sub = sub.dropna()
        if sub.empty:
            continue
        vals0 = sub[sub["stereotype_label"] == 0][metric].to_numpy()
        vals1 = sub[sub["stereotype_label"] == 1][metric].to_numpy()
        if len(vals0) == 0 or len(vals1) == 0:
            continue
        rows.append((metric, vals0, vals1))
    if not rows:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(rows) + 2.5))
    data = []
    labels = []
    for metric, vals0, vals1 in rows:
        data.append(vals0)
        data.append(vals1)
        labels.extend([f"{pretty_metric_name(metric)} (Non-biased)", f"{pretty_metric_name(metric)} (Biased)"])
    ax.violinplot(data, showmedians=True, vert=False, positions=np.arange(len(data)))
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Value")
    ax.set_title(title)
    ax.grid(axis="x")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

