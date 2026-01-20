from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

from ._common import set_plot_style


def plot(stats: Dict[str, Any], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    dominant_counts = stats.get("dominant_counts") or {}
    emotion_counts = stats.get("emotion_counts") or {}
    if not dominant_counts:
        return

    emotions = sorted(emotion_counts.keys(), key=lambda e: emotion_counts.get(e, 0), reverse=True)
    genre_totals: Counter = Counter()
    for _, cnts in dominant_counts.items():
        genre_totals.update(cnts)
    genres = [g for g, _ in genre_totals.most_common()]
    if not emotions or not genres:
        return

    emo_total = float(sum(emotion_counts.values()))
    genre_total = float(sum(genre_totals.values()))
    if emo_total <= 0 or genre_total <= 0:
        return

    emo_pos: Dict[str, tuple[float, float]] = {}
    y = 0.0
    for emo in emotions:
        h = emotion_counts.get(emo, 0) / emo_total
        emo_pos[emo] = (y, y + h)
        y += h

    genre_pos: Dict[str, tuple[float, float]] = {}
    y = 0.0
    for g in genres:
        h = genre_totals.get(g, 0) / genre_total
        genre_pos[g] = (y, y + h)
        y += h

    set_plot_style()
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(Rectangle((0.05, 0), 0.1, 1, facecolor="#f0f0f0", edgecolor="#cccccc"))
    ax.add_patch(Rectangle((0.85, 0), 0.1, 1, facecolor="#f0f0f0", edgecolor="#cccccc"))

    cmap = plt.get_cmap("tab20")
    for i, emo in enumerate(emotions):
        y0, y1 = emo_pos[emo]
        ax.text(
            0.02,
            0.5 * (y0 + y1),
            f"{emo} (n={emotion_counts.get(emo, 0)})",
            ha="right",
            va="center",
            fontsize=8,
        )
        for g, count in dominant_counts.get(emo, {}).items():
            if count <= 0:
                continue
            g0, g1 = genre_pos.get(g, (0.0, 0.0))
            h = count / emo_total
            left_top = y1
            left_bot = y1 - h
            y1 = left_bot
            poly = Polygon(
                [(0.15, left_bot), (0.15, left_top), (0.85, g1), (0.85, g0)],
                closed=True,
                facecolor=cmap(i % 20),
                alpha=0.25,
                edgecolor=None,
            )
            ax.add_patch(poly)

    for g in genres:
        y0, y1 = genre_pos[g]
        ax.text(0.98, 0.5 * (y0 + y1), g, ha="left", va="center", fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

