from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Serif",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "grid.color": "#e6e6e6",
            "grid.linestyle": "-",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def pretty_metric_name(metric: str) -> str:
    mapping = {
        "p01": "P@1%",
        "p05": "P@5%",
        "p10": "P@10%",
        "pi": "PI (rank utility)",
        "pop_mean_pct": "Mean popularity percentile",
        "cep": "CEP similarity",
        "uiop": "UIOP similarity",
        "genre_js": "Genre JS divergence",
        "genre_entropy": "Genre entropy",
        "year_decade_js": "Year/decade JS divergence",
        "year_decade_entropy": "Year/decade entropy",
        "mean_year": "Mean release year",
        "redundancy_new": "Redundancy: new items",
        "redundancy_repeated": "Redundancy: repeated items",
    }
    return mapping.get(metric, metric)


def emotion_counts(df: pd.DataFrame) -> Dict[str, int]:
    if "emotion_top1" not in df:
        return {}
    counts = df["emotion_top1"].dropna().value_counts()
    return {str(k): int(v) for k, v in counts.items()}


def metric_stats_by_emotion(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df or "emotion_top1" not in df:
        return pd.DataFrame([])
    sub = df[["emotion_top1", metric]].copy()
    sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
    sub = sub.dropna()
    if sub.empty:
        return pd.DataFrame([])
    stats_df = sub.groupby("emotion_top1")[metric].agg(["mean", "std", "count"])
    return stats_df[stats_df["count"] > 0]


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 500, seed: int = 7) -> tuple[float, float]:
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for _ in range(n_boot)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
