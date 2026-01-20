from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if "stereotype_label" not in df or "emotion_top1" not in df:
        return
    sub = df[["emotion_top1", "stereotype_label"]].dropna()
    sub = sub[sub["stereotype_label"].isin([0, 1])]
    if sub.empty:
        return

    counts = (
        sub.groupby(["emotion_top1", "stereotype_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=[0, 1], fill_value=0)
    )
    counts["total"] = counts.sum(axis=1)
    counts = counts[counts["total"] > 0].sort_values("total", ascending=False)
    if counts.empty:
        return

    overall = sub["stereotype_label"].value_counts().reindex([0, 1], fill_value=0)
    overall_total = int(overall.sum())
    if overall_total > 0:
        overall_row = pd.DataFrame(
            {0: [int(overall.get(0, 0))], 1: [int(overall.get(1, 0))], "total": [overall_total]},
            index=["All"],
        )
        counts = pd.concat([overall_row, counts], axis=0)

    labels = [f"{idx} (n={int(row.total)})" for idx, row in counts.iterrows()]
    p0 = (counts[0] / counts["total"]) * 100.0
    p1 = (counts[1] / counts["total"]) * 100.0

    height = max(4.0, 0.35 * len(labels) + 1.5)
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, height))
    y = np.arange(len(labels))
    ax.barh(y, p0, color="#54a24b", label="Non-biased")
    ax.barh(y, p1, left=p0, color="#e45756", label="Biased")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Percent of episodes")
    ax.set_title(title)
    ax.grid(axis="x")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

