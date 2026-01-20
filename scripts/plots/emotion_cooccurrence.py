from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._common import emotion_counts, set_plot_style


def plot(df: pd.DataFrame, out_path: Path, title: str, top_n: int = 3) -> None:
    import matplotlib.pyplot as plt

    emo_cols = [c for c in df.columns if c.startswith("emo_")]
    if not emo_cols:
        return

    raw_labels = sorted([c[len("emo_") :] for c in emo_cols])
    idx = {lab: i for i, lab in enumerate(raw_labels)}
    mat = np.zeros((len(raw_labels), len(raw_labels)), dtype=float)

    for _, row in df[emo_cols].iterrows():
        scores = []
        for col in emo_cols:
            val = row.get(col)
            if pd.isna(val):
                continue
            try:
                score = float(val)
            except Exception:
                score = 0.0
            if score <= 0:
                continue
            scores.append((col[len("emo_") :], score))
        if not scores:
            continue
        scores.sort(key=lambda x: x[1], reverse=True)
        top = [lab for lab, _ in scores[:top_n]]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a = idx[top[i]]
                b = idx[top[j]]
                mat[a, b] += 1
                mat[b, a] += 1

    if not mat.any():
        return

    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    im = ax.imshow(mat, cmap="Reds")
    ax.set_title(title)
    counts = emotion_counts(df)
    display_labels = [f"{lab} (n={counts.get(lab, 0)})" for lab in raw_labels]
    ax.set_xticks(np.arange(len(raw_labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(raw_labels)))
    ax.set_yticklabels(display_labels)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Co-occurrence count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

