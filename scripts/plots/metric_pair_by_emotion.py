from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ._common import metric_stats_by_emotion, pretty_metric_name, set_plot_style


def plot(df: pd.DataFrame, metric_a: str, metric_b: str, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    stats_a = metric_stats_by_emotion(df, metric_a)
    stats_b = metric_stats_by_emotion(df, metric_b)
    if stats_a.empty and stats_b.empty:
        return

    emotions = set(stats_a.index).union(stats_b.index)
    if not emotions:
        return

    def count(emotion: str) -> int:
        ca = int(stats_a.loc[emotion, "count"]) if emotion in stats_a.index else 0
        cb = int(stats_b.loc[emotion, "count"]) if emotion in stats_b.index else 0
        return max(ca, cb)

    ordered = sorted(emotions, key=lambda e: (count(e), e), reverse=True)
    labels = [f"{emo} (n={count(emo)})" for emo in ordered]
    y = np.arange(len(labels))

    def series(stats_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        means: list[float] = []
        cis: list[float] = []
        for emo in ordered:
            if emo not in stats_df.index:
                means.append(np.nan)
                cis.append(0.0)
                continue
            row = stats_df.loc[emo]
            mean = float(row["mean"])
            std = float(row["std"]) if not math.isnan(float(row["std"])) else 0.0
            n = int(row["count"])
            se = std / math.sqrt(n) if n > 1 else 0.0
            cis.append(1.96 * se)
            means.append(mean)
        return np.array(means), np.array(cis)

    means_a, cis_a = series(stats_a)
    means_b, cis_b = series(stats_b)

    height = max(4.0, 0.33 * len(labels) + 1.5)
    set_plot_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, height), sharey=True)
    axes[0].errorbar(means_a, y, xerr=cis_a, fmt="o", color="#4c78a8", ecolor="#999999", capsize=2)
    axes[1].errorbar(means_b, y, xerr=cis_b, fmt="o", color="#f58518", ecolor="#999999", capsize=2)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel(pretty_metric_name(metric_a))
    axes[1].set_xlabel(pretty_metric_name(metric_b))
    axes[0].grid(axis="x")
    axes[1].grid(axis="x")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

