from __future__ import annotations

from pathlib import Path

import pandas as pd


def plot(df: pd.DataFrame, out_path: Path, title: str) -> None:
    if "stereotype_score_1" not in df:
        return
    from .violin_by_emotion import plot as violin_by_emotion

    violin_by_emotion(df, "stereotype_score_1", out_path, title)

