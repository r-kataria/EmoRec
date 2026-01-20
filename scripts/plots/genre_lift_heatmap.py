from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np

from ._common import set_plot_style


def plot(stats: Dict[str, Any], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    overall = stats.get("overall_mean") or {}
    emo_mean = stats.get("emotion_mean") or {}
    if not overall or not emo_mean:
        return

    genres = sorted(overall.keys())
    counts = stats.get("emotion_counts") or {}
    emotions = sorted(emo_mean.keys(), key=lambda e: counts.get(e, 0), reverse=True)
    if not genres or not emotions:
        return

    mat = np.array(
        [[float(emo_mean.get(emo, {}).get(g, 0.0) - overall.get(g, 0.0)) for g in genres] for emo in emotions],
        dtype=float,
    )

    set_plot_style()
    fig, ax = plt.subplots(figsize=(0.6 * len(genres) + 3.5, 0.35 * len(emotions) + 2.5))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#eeeeee")
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(genres)))
    ax.set_xticklabels(genres, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(emotions)))
    ax.set_yticklabels([f"{emo} (n={counts.get(emo, 0)})" for emo in emotions])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Lift vs overall")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

