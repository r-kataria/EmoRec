from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ._common import ensure_dir


def build_all_plots(
    *,
    redial_df: pd.DataFrame,
    cosrec_df: pd.DataFrame,
    redial_genre_stats: Dict[str, Any],
    cosrec_genre_stats: Dict[str, Any],
    redial_exposure_items: Dict[str, Any],
    cosrec_exposure_items: Dict[str, Any],
    redial_exposure_emotions: Dict[str, Any],
    cosrec_exposure_emotions: Dict[str, Any],
    out_dir: Path,
) -> None:
    from .exposure_emotion_curves import plot as exposure_emotion_curves
    from .genre_lift_heatmap import plot as genre_lift_heatmap
    from .stereotype_by_emotion import plot as stereotype_by_emotion

    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    stereotype_by_emotion(redial_df, out_dir / "stereotype_by_emotion_redial.png", "ReDial: Stereotype by emotion (top-1)")
    stereotype_by_emotion(cosrec_df, out_dir / "stereotype_by_emotion_cosrec.png", "CoSRec: Stereotype by emotion (top-1)")

    exposure_emotion_curves(
        redial_exposure_items["counts"],
        redial_exposure_emotions["counts"],
        redial_exposure_items["percentiles"],
        redial_exposure_emotions["episodes"],
        out_dir / "exposure_gini_redial.png",
        "ReDial: Exposure vs popularity (Gini)",
    )
    exposure_emotion_curves(
        cosrec_exposure_items["counts"],
        cosrec_exposure_emotions["counts"],
        cosrec_exposure_items["percentiles"],
        cosrec_exposure_emotions["episodes"],
        out_dir / "exposure_gini_cosrec.png",
        "CoSRec: Exposure vs popularity (Gini)",
    )

    genre_lift_heatmap(redial_genre_stats, out_dir / "genre_lift_heatmap_redial.png", "ReDial: Genre lift by emotion")
    genre_lift_heatmap(cosrec_genre_stats, out_dir / "genre_lift_heatmap_cosrec.png", "CoSRec: Category lift by emotion")

