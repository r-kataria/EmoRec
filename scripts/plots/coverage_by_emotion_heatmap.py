from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import metric_stats_by_emotion, pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    metrics = ["p01", "p05", "p10"]
    stats = {m: metric_stats_by_emotion(df, m) for m in metrics}
    emotions = set()
    for s in stats.values():
        emotions.update(s.index.tolist())
    if not emotions:
        return

    def count(emo: str) -> int:
        return max(int(stats[m].loc[emo, "count"]) if emo in stats[m].index else 0 for m in metrics)

    ordered = sorted(emotions, key=lambda e: count(e), reverse=True)
    data = []
    for emo in ordered:
        row = []
        for m in metrics:
            s = stats[m]
            row.append(np.nan if emo not in s.index else float(s.loc[emo, "mean"]))
        data.append(row)

    mat = np.array(data, dtype=float)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.0, 0.35 * len(ordered) + 2.2))
    cmap = plt.get_cmap("Blues").copy()
    cmap.set_bad("#eeeeee")
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([pretty_metric_name(m) for m in metrics], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels(ordered)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Mean coverage")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

