from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import bootstrap_mean_ci, pretty_metric_name, set_plot_style


def plot(
    df: pd.DataFrame,
    emotion_label: str,
    metric: str,
    out_path: Path,
    title: str,
    bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    col = f"emo_{emotion_label}"
    if col not in df or metric not in df:
        return
    x = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy()
    y = pd.to_numeric(df[metric], errors="coerce").to_numpy()
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if len(x) == 0:
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    ax.scatter(x, y, s=8, alpha=0.2, color="#4c78a8")

    edges = np.linspace(0.0, 1.0, bins + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    means: list[float] = []
    cis: list[tuple[float, float]] = []
    for i in range(bins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        vals = y[m]
        if len(vals) == 0:
            means.append(np.nan)
            cis.append((np.nan, np.nan))
        else:
            means.append(float(np.mean(vals)))
            cis.append(bootstrap_mean_ci(vals))
    means_arr = np.array(means, dtype=float)
    lows = np.array([c[0] for c in cis], dtype=float)
    highs = np.array([c[1] for c in cis], dtype=float)

    ax.plot(mids, means_arr, color="#e45756", linewidth=2)
    ax.fill_between(mids, lows, highs, color="#e45756", alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(f"{emotion_label} score")
    ax.set_ylabel(pretty_metric_name(metric))
    ax.grid(axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

